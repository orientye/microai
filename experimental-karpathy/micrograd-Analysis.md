# micrograd 源码分析

对照说明：[`micrograd/README.md`](micrograd/README.md)。  
官方仓库：[karpathy/micrograd](https://github.com/karpathy/micrograd)。  
本文件按**这份克隆的真实源码**拆，不按讲座口头描述。

-  §1～§3，「它在算什么、文件在哪」。
- **§4～§7**：数据结构、前向传播、反向传播、链式法则，共用同一个手算例子。
- §8～§10 是 MLP / demo / 测试；若准备自己重写，最后看 §11。

---

## 1. 一句话

micrograd 在**标量**上做反向模式自动微分：前向时把每次 `+` / `*` / `**` / `ReLU` 建成一张动态 DAG，反向时拓扑排序后沿闭包把链式法则推回去。上面再叠约 50 行的 `Neuron` / `Layer` / `MLP`。一个神经元被拆成一堆标量加减乘，这就够训一个二分类 MLP。

它解决的是「autograd 到底在干什么」这件事，不是性能。DAG 只认 `float`，没有张量、没有广播、没有 GPU。

---

## 2. 和本仓库 microai 对照

本仓库的 [`microai/core.py`](../microai/core.py) 是 DeZero 路线：`Variable` 挂 `ndarray`，算子是独立的 `Function` 对象。micrograd 把算子的反向直接闭包在输出节点上。

| | **micrograd `Value`** | **microai `Variable`** |
|--|----------------------|------------------------|
| 数据 | 一个 Python `float` | `np.ndarray` / `cupy.ndarray` |
| 图怎么连 | `_prev` + 每节点一个 `_backward` 闭包 | `creator` 指向 `Function`，`Function.inputs/outputs` |
| 反向遍历 | DFS 拓扑序，再 `reversed` 调闭包 | 按 `generation` 弹出 `Function`，调 `f.backward` |
| 梯度存放 | `grad` 初始就是 `0` | `grad` 初始 `None`，第一次 backward 才填 |
| 高阶导 / `no_grad` | 没有 | `create_graph`、`Config.enable_backprop` |
| 广播、reshape、GPU | 没有 | 有 |

所以讲座里「神经网络就是一堆加减乘」在这里是字面意思：没有矩阵乘法原语，`Neuron` 的点积是 Python 循环里一次一次 `wi*xi`。

| | micrograd | microai | PyTorch |
|--|-----------|---------|---------|
| 学什么 | `dL/dθ`，标量图 | 同样是反向 AD，张量图 | 张量图 + 内核 |
| 图何时建 | 每次前向现场建 | 每次前向现场建 | 默认动态 |
| 反向局部导数 | **闭包里写死** | `Function.backward` | `autograd.Function` / C++ |
| 训练循环 | 手写 SGD：`p.data -= lr * p.grad` | `optimizers.py` | `optimizer.step()` |

公式直觉：多元复合函数的链式法则。一次 `loss.backward()` 得到**所有**叶子参数的梯度；正向模式要对每个参数各扫一遍。参数远多于输出维时，反向才划算——神经网络正好是这个形状。

---

## 3. 目录与读序

```text
experimental-karpathy/
  micrograd-Analysis.md   # 本文件
  microgpt-Analysis.md    # 相对本文的下一步：microgpt
  micrograd/              # 上游克隆（见 .gitignore）
    micrograd/
      engine.py           # Value：图 + 反向（~100 行）
      nn.py               # Module / Neuron / Layer / MLP（~50 行）
      __init__.py         # 空
    test/test_engine.py   # 和 torch.Tensor 对梯度
    demo.ipynb            # moons 二分类
    trace_graph.ipynb     # graphviz 画 DAG
    setup.py
```

依赖：标准库就能跑 `engine.py` / `nn.py`。`demo.ipynb` 要 `numpy` / `matplotlib` / `sklearn`；单测要 `torch`；画图要 `graphviz`。

建议读序：

1. `engine.py` 的 `Value.__init__` — §4 五个字段
2. `__add__` / `__mul__` — §5 前向怎么长边、闭包怎么挂
3. `backward()` — §6 拓扑序；对照 §7 把 `+=` 当成多路径求和
4. `nn.py` 的 `Neuron.__call__` — 点积是标量循环
5. `demo.ipynb` 的 `loss()` 和 SGD — hinge + `zero_grad`
6. `test/test_engine.py` — 用 PyTorch 当标准答案

下文四个维度共用这个例子（菱形：`a`、`b` 各被用两次）：

```python
a = Value(2.0)
b = Value(3.0)
c = a * b          # 6
d = a + b          # 5
e = c * d          # 30
e.backward()       # 没有这一行，a.grad / b.grad 仍是 0
```

解析式 `e = (a b)(a + b) = a²b + a b²`，手算：

- `∂e/∂a = 2 a b + b² = 12 + 9 = 21`
- `∂e/∂b = a² + 2 a b = 4 + 12 = 16`

`e.backward()` 后：
    `e.data == 30`、`a.grad == 21`、`b.grad == 16`。
前向只填 `data`；梯度是反向写进去的。

---

## 4. 数据结构

micrograd 只有一种运行时对象：`Value`。没有 `Function` 类、没有边对象、没有单独的计算图容器。图 = 一堆 `Value` 用 `_prev` 互相指着。

### 4.1 核心：`Value` 对象的五个字段

`Value` 不是「一个 float」，是计算图上的一个节点。`engine.py` 的构造：

```python
def __init__(self, data, _children=(), _op=''):
    self.data = data
    self.grad = 0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
```

五个字段各管一件事（`c = a * b` 时，构造里的 `_children` 就是 `(self, other)`，即 `(a, b)`）：

| 字段 | 存什么 | 例子里 |
|------|--------|--------|
| `data` | 该节点的**标量**数值 | `a.data = 2.0`，`c.data = 6.0` |
| `grad` | **最终输出**（调用了 `.backward()` 的那个根，训练时就是损失）相对于**该节点**的梯度。初始化为 `0` | `e.backward()` 之前全是 0；之后 `a.grad = 21` |
| `_prev` | 生成该节点的输入来源（代码里传入 `(self, other)`，再做成 `set`）。用来建 **DAG**、给拓扑 DFS 当邻接表 | `c._prev = {a, b}`；叶子 `a._prev = {}` |
| `_op` | 生成该节点的运算符字符串。**只用于调试和可视化**，反向不算它 | `c._op = '*'`，`d._op = '+'`，叶子是 `''` |
| `_backward` | 函数指针（闭包）。算当前节点相对其输入的**局部梯度**，再 **`+=` 累加**到输入节点的 `grad` 上 | `c._backward` 会做 `a.grad += b.data * c.grad` |

对照源码看每一项：

**`data`**  
前向的唯一产物。叶子由用户写入（`Value(2.0)`）；中间节点由算子写出（`self.data * other.data`）。反向时乘法 / 幂的局部导还要**再读**它，所以必须先 `backward` 再改 `p.data`（§5.2）。

**`grad`**  
含义是 `∂(backward 的根) / ∂(自己)`，不是「对下一层的局部导」。初始化 `0`（整数 `0`，和 `0.0` 同值）。前向不读、不写（新节点默认就是 0）。没有 `requires_grad`：每个节点都带这个字段。训练时根通常是 loss，但引擎并不知道「损失」这个词——谁调用 `.backward()`，`grad` 就是对谁的。

**`_prev`**  
方向是**指向输入**（拓扑更早的一方），不是指向消费者。`__add__` / `__mul__` 里写成：

```python
out = Value(..., (self, other), '+')   # _children = (self, other)
# __init__ 里：self._prev = set(_children)
```

所以文档和口播里说的「`(self, other)`」是构造参数；对象上的字段是 `set`。`set((a, a)) == {a}`，同一输入只留一份，见 §4.3、§7.4。叶子：`_children=()`，`_prev` 为空。

**`_op`**  
`'+'`、`'*'`、`f'**{other}'`、`'ReLU'`，或叶子的空串。`backward()` **不读** `_op`。`trace_graph.ipynb` 用它画算子盒子。

**`_backward`**  
叶子默认 `lambda: None`。中间节点在前向时被赋成一个闭包：读取 `out.grad`（上游已经填好的全导数），乘上本算子的局部导，`+=` 到 `_prev` 里那些输入的 `grad`。反向真正执行链式法则的是它，不是 `_prev`，也不是 `_op`。

五个字段按职责分层：

| 层 | 字段 | 前向 | 反向 | 画图 |
|----|------|------|------|------|
| 数值 | `data` | 读、写出新节点 | 乘法等局部导要读它 | 左格 |
| 数值 | `grad` | 不读 | 写 `d(根)/d(自己)` | 右格 |
| 图骨架 | `_prev` | 写入，之后前向不读 | 拓扑 DFS 的邻接表 | 边 |
| 图语义 | `_backward` | 写入一个闭包 | **真正执行链式法则** | 不用 |
| 标签 | `_op` | 写入 | 不用 | 算子盒子 |

没有 `generation` / `creator`（那是本仓库 `microai.Variable` 的拆法）。`)

### 4.2 图是 DAG，不是树

例子前向结束后，堆上是 5 个对象：

```text
        a(data=2, grad=0)          b(data=3, grad=0)
         叶子 _prev={}               叶子 _prev={}
              \     \               /     /
               \     \             /     /
                \     +-----------+     /
                 \                     /
              c=* (6)              d=+ (5)
              _prev={a,b}          _prev={a,b}
                    \                 /
                     \               /
                      e=* (30)
                      _prev={c,d}
```

`a` 有两个下游 `c`、`d`，所以是 **DAG**：有汇合，不是树。反向时 `a.grad` 必须把两条路加起来（§7）。

`_prev` 的方向是**指向输入**（拓扑更早的一方）。`trace_graph.ipynb` 画边时再翻转成「输入 → 输出」。不要把 `_prev` 理解成指向消费者。

`_prev` 是 `set`，不是 list：同一对象只出现一次。这影响画图和拓扑，**不影响**闭包里 `self`/`other` 各写一次梯度，见 §4.3、§7.4。

### 4.3 闭包才是边的语义

`+` 并不往图里塞一个 `Add` 对象。它做三件事：新建 `out`、把 `(self, other)` 放进 `_prev`、再把局部反向**冻**进 `out._backward`：

```python
out = Value(self.data + other.data, (self, other), '+')

def _backward():
    self.grad += out.grad
    other.grad += out.grad
out._backward = _backward
```

闭包捕获的是当时的 `self`、`other`、`out` 三个对象引用。反向执行时不再查表「这个 `+` 的输入是谁」——名字已经绑死了。

因此数据结构里其实有**两套邻接**：

| | 存在哪 | 反向用它吗 |
|--|--------|------------|
| `_prev` | `set` of `Value` | 只用它做拓扑序 |
| 闭包自由变量 | `self` / `other` / `out` | **真正写梯度** |

两套必须一致（同一对输入），但 `a + a` 时 `set((a,a)) == {a}`，骨架少一条，闭包仍是两次 `+=`。梯度以闭包为准。

### 4.4 `nn` 侧：参数是叶子列表

`Neuron` / `Layer` / `MLP` **不是**图节点。它们只是「哪些 `Value` 算参数」的目录：

```python
self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
self.b = Value(0)
# parameters() = self.w + [self.b]
```

一次 `__call__` 会新建一批中间 `Value`（每个 `*`、每个 `+`、可能一个 `ReLU`），但 `w`/`b` 一直是那些叶子。SGD 改的是叶子的 `.data`；下一轮前向再长出新的中间节点。图没有跨 step 复用。

`Module.zero_grad` 只扫 `parameters()`，不扫中间节点。中间节点每轮都是新对象，`grad` 从 0 起，无所谓。

### 4.5 和 `microai` 的结构差

| 角色 | micrograd | microai |
|------|-----------|---------|
| 数值节点 | `Value`（标量 + 图 + 闭包） | `Variable`（ndarray + `creator`） |
| 算子 | 没有类型，闭包顶替 | `Function` 子类，`inputs`/`outputs` |
| 边 | `_prev` + 捕获的引用 | `Function` 两端的列表（output 还是 weakref） |
| 参数 | 还是 `Value`，靠 `parameters()` 收集 | `Parameter(Variable)` 子类 |

同一套 AD，micrograd 把「节点 / 算子 / 边」压进一个类；`microai` 把算子拆出去，才能做广播、高阶导、`no_grad`。

---

## 5. 前向传播

前向的职责只有两件：**算出 `out.data`**，以及**把以后反向要用的东西挂到 `out` 上**。不碰任何 `.grad`（除了新节点默认的 0）。

### 5.1 一次运算 = 三个动作

以 `c = a * b` 为例：

1. 若 `b` 不是 `Value`，包一层 `Value(b)`（常数叶子，之后没有参数更新也会占一个节点）。
2. `out.data = a.data * b.data` → `6.0`。纯 Python `float` 乘法。
3. `out._prev = {a, b}`，`out._op = '*'`，`out._backward` 设成乘法的局部反向。
4. `return out`。调用方拿到的是**新对象**，`a`、`b` 不变。

`+=` 也是新建：`c += c + 1` 先算出右边那个新 `Value`，再把名字 `c` 指过去。旧节点若还在别人的 `_prev` 里，继续留在图上。

### 5.2 四个原语的前向

只有这四个自己写了 `_backward`，其余全转发过来。

| 代码 | `out.data` | `_prev` | 闭包稍后要用的前向量 |
|------|------------|---------|----------------------|
| `a + b` | `a.data + b.data` | `{a,b}` | 加法局部导恒为 1，不额外存 |
| `a * b` | `a.data * b.data` | `{a,b}` | 闭包读 `a.data`、`b.data` |
| `a ** n` | `a.data ** n` | `{a}` | `n` 和 `a.data`（`n` 是 Python 数） |
| `a.relu()` | `0 if a.data < 0 else a.data` | `{a}` | 用 `out.data > 0` 当门 |

乘法、幂的局部导数依赖**前向时的输入值**。闭包不拷贝一份 `float`，直接读对象上的 `.data`。训练时必须先 `backward` 再改 `p.data`；若先改 `data` 再 `backward`，乘法的局部导会用到**更新后**的值，梯度是错的。demo 的顺序是对的：backward → `p.data -= lr * p.grad`。

`**` 断言指数是 `int`/`float`，不能 `Value ** Value`。`n` 不是图上的节点，`∂(aⁿ)/∂n` 不算。

### 5.3 派生算子是前向宏

不写第二套反向：

| 写法 | 展开成的原语图 |
|------|----------------|
| `-x` | `x * (-1)`，多一个常数叶子 `-1` |
| `x - y` | `x + (y * -1)` |
| `x / y` | `x * (y ** -1)` |
| `2 + x` | `__radd__` → `x + 2` |
| `2 * x` | `__rmul__` → `x * 2` |

`e / 2` 会多出 `2 ** -1` 再乘。图比「看起来的表达式」更深一截，数值和梯度仍对，因为原语的反向已经对过。

### 5.4 逐步跑例子

```text
a = Value(2.0)     # 叶子
b = Value(3.0)     # 叶子
c = a * b          # 新建 c，data=6，闭包捕获 a,b,c
d = a + b          # 新建 d，data=5，闭包捕获 a,b,d
e = c * d          # 新建 e，data=30，闭包捕获 c,d,e
```

此时所有 `grad` 仍是 0。前向**不**把梯度清零以外的任何东西写进叶子——叶子的 `grad` 若上一轮没 `zero_grad`，会一直躺到这次 `+=`。

`Neuron.__call__` 是同一套前向的循环展开：

```python
act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
return act.relu() if self.nonlin else act
```

`nin=2` 时图是：`b + (w0*x0) + (w1*x1)`，再可选 ReLU。`sum(..., self.b)` 从 bias 那个 `Value` 起加，避免默认 `sum` 从整数 `0` 起（虽然 `__radd__` 也能接）。没有 matmul 节点；宽度 `n` 就有 `n` 次 `*` 和 `n` 次 `+`。

MLP 前向是层与层把 list of `Value` 往后传。最后一层 `nonlin=False`，输出一个标量打分。一次 `loss()` 对 100 个样本各建一张图，再 `sum` 成一个 `total_loss`——那是 100 张小 DAG 的汇合点，一次 `backward()` 从这一点灌回去。

### 5.5 前向的边界

- 动态图：控制流（Python `if`）走哪条，图就长成什么样。ReLU 的 `if` 在算 `data` 时已经定死，反向用 `out.data > 0` 复述这个决定，而不是再跑一遍 Python `if`。
- 不缓存「算子类型」以外的中间量（乘法直接读 `.data`）。
- 常数和参数在类型上无区别，都是 `Value`。区别只在你是否把它们放进 `parameters()` 去做 SGD。

---

## 6. 反向传播

反向的职责：在已经建好的 DAG 上，给每个节点填 `grad = ∂(调用了 backward 的那个根)/∂(该节点)`。实现就两步——**排序**，再**按闭包撒梯度**。

### 6.1 入口

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    self.grad = 1
    for v in reversed(topo):
        v._backward()
```

`self` 是要微分的标量（loss / 例子里的 `e`）。`self.grad = 1` 的含义：`∂e/∂e = 1`。这是**赋值**，不是 `+=`。

叶子的 `_backward` 是空函数：没有更上游的输入可写。叶子的 `grad` 全是下游闭包写进来的。

### 6.2 为什么要拓扑序

每个闭包都假设：**自己的 `out.grad` 已经是最终值**，再乘局部导送给输入。若先处理 `c` 再处理 `e`，`c.grad` 还是 0，乘法会把 0 送给 `a`、`b`。

后序 DFS：先递归所有 `_prev`，再 `topo.append(自己)` → 输入在列表里更早、输出更晚。`reversed(topo)` 就是输出 → 输入。

`visited` 保证每个节点只入序一次。菱形里 `a` 既是 `c` 的输入也是 `d` 的输入，只 append 一次——正确，因为 `a` 只要在 `c`、`d` 之前出现；`a` 的 `_backward` 是空的，真正累加发生在 `c._backward` 和 `d._backward` 里。

`_prev` 是 `set`，子节点遍历顺序不确定。任意合法拓扑序都可以：`[a,b,c,d,e]` 和 `[a,b,d,c,e]` 反向结果相同。

递归 DFS 的深度 ≈ 最长路径。Python 默认 1000 层，教学图没事，深 MLP 按标量展开有可能爆；生产级会改迭代。

### 6.3 例子逐步执行

一种可能的 `topo = [a, b, c, d, e]`，`reversed` 为 `[e, d, c, b, a]`。

| 步 | 调用 | 闭包做什么 | 之后的 grad |
|----|------|------------|-------------|
| 0 | `e.grad = 1` | 种子 | e=1；其余 0 |
| 1 | `e._backward()` | `c.grad += d.data * 1` → 5；`d.grad += c.data * 1` → 6 | c=5, d=6 |
| 2 | `d._backward()` | `a.grad += 6`；`b.grad += 6` | a=6, b=6 |
| 3 | `c._backward()` | `a.grad += 3*5` → 21；`b.grad += 2*5` → 16 | a=21, b=16 |
| 4–5 | `b`/`a` 的空 lambda | 无 | 不变 |

与手算 21、16 一致。步 2 和步 3 谁先都行，因为它们只读 `c.grad`/`d.grad`（已在步 1 写完），并往 `a`/`b` 上 `+=`。

### 6.4 `+=`、`zero_grad`、二次 `backward`

闭包一律 `+=`，因为一个节点可能被多个下游写（菱形的 `a`）。`grad` 初值 0，第一次反向看起来像赋值。

训练循环里上一轮叶子的 `grad` 还在。必须先 `model.zero_grad()`（写成 0），再 `backward()`，否则第 2 轮的 21 会加在第 1 轮的 21 上变成 42。这和 PyTorch 相同。

对**同一个**已填过梯度的图再调一次 `e.backward()`、中间不清零：

- 根 `e.grad` 被重新赋成 1（还好）
- `c`、`d`、`a`、`b` 再 `+=` 一遍 → 翻倍

demo 的契约：每轮新建图（中间节点是新的），叶子先清零，backward 一次。

`zero_grad` 清的是参数叶子。不要指望它清中间节点；中间节点本就不该活过一轮。

### 6.5 反向模式在干什么

对标量 `e(a,b)`，反向一次遍历，同时得到 `∂e/∂a` 和 `∂e/∂b`。正向 AD 要对 `a`、`b` 各扫一遍图。

神经网络：参数个数 ≫ 损失维数（标量）。所以训练用反向。micrograd 没有正向模式实现。

和 `microai` 的差别只在遍历结构：那边按 `Function.generation` 弹出算子，调 `f.backward(*gys)`；这边按节点拓扑调闭包。数学相同。

### 6.6 实现边界

- 只对**标量根**有定义。没有 `y.backward(gradient=...)` 这种向量-雅可比积接口；根被写死为 1。
- 没有 `retain_graph`：闭包还在，理论上可再跑，但 `+=` 会叠（见上）。
- 没有 `no_grad`：前向总是建图、挂闭包。
- `trace_graph.ipynb` 的 `draw_dot` 在 `backward()` **之后**调用才看得到非零 `grad`；它只读字段，不参与计算。

---

## 7. 链式法则

反向传播是链式法则的一种**求值顺序**。每个闭包只实现「这一跳」的局部导数；整条链靠拓扑序把局部导连乘（多路径则相加）。

### 7.1 一跳：局部导 × 上游梯度

对 `out = f(in)`，要的是 `∂L/∂in`，已知 `out.grad = ∂L/∂out`：

```text
in.grad += (∂out/∂in) * out.grad
```

代码里的 `+=` 右边就是这一项。`out.grad` 是上游已经算好的 `∂L/∂out`；括号里是只依赖本算子的局部导数。

这就是反向模式的向量-雅可比积在标量上的退化：雅可比是一个数。

### 7.2 四个原语的局部导数

从 `f` 的微积分直接抄进闭包：

| 算子 | ∂out/∂self | ∂out/∂other | 代码 |
|------|------------|-------------|------|
| `out = self + other` | 1 | 1 | `self.grad += out.grad` |
| `out = self * other` | `other.data` | `self.data` | `self.grad += other.data * out.grad` |
| `out = self ** n` | `n * self.data**(n-1)` | （n 不在图上） | 与公式相同 |
| `out = relu(self)` | `1 if out.data > 0 else 0` | — | `(out.data > 0) * out.grad` |

加法：`e = c * d` 里若看 `d = a + b`，`∂d/∂a = 1`，所以 `d` 的闭包把 `d.grad` **原样**加给 `a` 和 `b`。

乘法：积规则。`e = c * d`：`∂e/∂c = d`（前向值 5），`∂e/∂d = c`（前向值 6）。这就是 §6.3 步 1。

ReLU：分段线性。`data > 0` 时局部导 1，梯度透传；`≤ 0` 时局部导 0，把上游梯度挡掉。用 `out.data` 而不是再读输入，和前向的 `0 if self.data < 0 else self.data` 在 0 点一致（0 的子梯度取 0，对齐 PyTorch）。

除法没有自己的公式：`x / y = x * y**-1`，链式法则自动变成 `∂/∂x = 1/y`、`∂/∂y = -x/y²`。

### 7.3 多元：多条路径相加

`e` 依赖 `a` 有两条路：`a → c → e` 和 `a → d → e`。全导数是

```text
∂e/∂a = (∂e/∂c)(∂c/∂a) + (∂e/∂d)(∂d/∂a)
      = d · b + (∂e/∂d) · 1
      = 5 · 3 + 6
      = 21
```

反向求值顺序正好是先算括号里的上游（`∂e/∂c`、`∂e/∂d`），再乘局部导、**加**到 `a.grad`：

```text
a.grad += (∂c/∂a) * c.grad    # c._backward：3 * 5
a.grad += (∂d/∂a) * d.grad    # d._backward：1 * 6
```

若把闭包里的 `+=` 改成 `=`，后执行的那条路会盖掉先执行的，菱形必错。`test_more_ops` 里大量节点复用，改 `=` 会炸。

`b` 同理：`∂e/∂b = c 对 b 的路 + d 对 b 的路 = 2*5 + 6 = 16`。

### 7.4 `a + a`：同一输入出现两次

```python
c = a + a          # out = Value(a.data+a.data, (a, a), '+')
```

微积分：`c = 2a`，`∂c/∂a = 2`。

数据结构：`_prev = {a}` 只有一个。闭包仍是：

```python
self.grad += out.grad     # self is a
other.grad += out.grad    # other is a
```

两次 `+=` 合成 `a.grad += 2 * out.grad`，与 `∂(2a)/∂a = 2` 一致（已验证：输出 2）。

若有人「为了和图一致」改成对 `_prev` 里每个节点加一次 `out.grad`，这里会变成只加一次，**少一半**。这是 §4.3「骨架 vs 闭包」的测试用例。

### 7.5 再套一层 ReLU

```python
f = e.relu()       # e=30 > 0，f.data=30
f.backward()
```

多一跳：`∂f/∂e = 1`（正区），`f.grad = 1` → `e.grad = 1`，其后与 §6.3 相同，`a.grad` 仍是 21。

若 `e.data < 0`（例如 `a=-2, b=-3`：`e = 6 * (-5) = -30`），`∂f/∂e = 0`，`e.grad` 保持 0，后面所有叶子梯度都是 0。这就是「死 ReLU」：前向已经关掉的通道，链式法则把梯度乘零传不回去（已验证：负区 `a.grad`、`b.grad` 均为 0）。

### 7.6 从 `loss` 到某一个 `w`

`demo` 里 `total_loss` 是许多 hinge 项和 L2 项之和。对某个隐层权重 `w`：

```text
∂L/∂w = Σ_样本 (∂L/∂score)(∂score/∂w)  +  2 α w
```

第一项：样本的 score 图一路乘回来；`w` 被多个神经元/样本共用时，还是靠 `+=` 把路径加在一起。第二项：`reg_loss = α Σ p²`，`p*p` 的局部导 `2p`，再乘 `α`。

SGD `p.data -= lr * p.grad` 不在图里，是图外用已经算完的全导数做一步欧拉。链式法则到 `grad` 填完就结束。

### 7.7 一张表对上四个维度

还是 `e = (a*b)*(a+b)`，`a=2,b=3`：

| 维度 | 这个例子里看到什么 |
|------|-------------------|
| 数据结构 | 5 个 `Value`；`a`/`b` 被两个 `_prev` 共享；梯度语义在三个 `*`/`+` 闭包里 |
| 前向 | `c=6,d=5,e=30`；闭包捕获当时的节点；`.grad` 全 0 |
| 反向 | 拓扑保证先 `e` 后 `c`/`d` 后叶子；种子 `e.grad=1` |
| 链式法则 | `∂e/∂a = d·b + (∂e/∂d)·1 = 15+6=21`；`+=` 把两条路合并 |

`trace_graph.ipynb` 在 `backward()` 之后画：每个盒子 `data | grad`，就是这张表的可视化。

---

## 8. `nn`：神经元就是标量循环

### 8.1 `Module`

`zero_grad` 把每个参数的 `grad` 写成 0。`parameters()` 默认空列表。没有 `eval()` / `train()`，也没有保存权重。

### 8.2 `Neuron`

```python
self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
self.b = Value(0)
act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
return act.relu() if self.nonlin else act
```

点积是 `zip` 上逐项 `wi*xi`，`sum(..., self.b)` 从 bias 起加，避免 `sum` 默认从 `0` 起（虽然 `__radd__` 也能扛）。输入比 `nin` 长会被 `zip` 丢掉，短则少乘几项——没有形状检查。

`nonlin=True`（默认）接 ReLU；最后一层关掉，输出是打分，不是概率。

### 8.3 `Layer` / `MLP`

一层 = 一排 `Neuron`。单输出时 `__call__` 拆包成一个 `Value`，多输出保持 list。

```python
sz = [nin] + nouts
self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
```

`MLP(2, [16, 16, 1])`：输入 2，两层 16 的 ReLU 隐层，最后 1 个线性神经元。参数个数：

| 层 | 计算 | 个数 |
|----|------|------|
| 16 × (2+1) | 隐层 1 | 48 |
| 16 × (16+1) | 隐层 2 | 272 |
| 1 × (16+1) | 输出 | 17 |
| 合计 | | **337** |

和 `demo.ipynb` 打印的 `number of parameters 337` 对得上。每个参数都是独立 `Value`，一次前向会为每个加、每个乘各建一个中间节点。337 个参数的 MLP，图上的节点数是参数量的好几倍。

---

## 9. `demo.ipynb`：hinge + L2 + SGD

数据：`sklearn.datasets.make_moons`，100 个点，标签从 `{0,1}` 改成 `{-1,+1}`。

损失（SVM max-margin / hinge）：

```python
losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
data_loss = sum(losses) * (1.0 / len(losses))
reg_loss = alpha * sum((p*p for p in model.parameters()))  # alpha=1e-4
total_loss = data_loss + reg_loss
```

`max(0, 1 - y·score)`。`y` 和 `score` 同号且 `|score| ≥ 1` 时该项为 0。准确率只看符号：`(yi > 0) == (scorei.data > 0)`。不是交叉熵，输出层因此保持线性。

训练 100 step，全量 batch（`loss()` 不传 `batch_size` 就用全部 100 点）：

1. `total_loss, acc = loss()` — 建图
2. `model.zero_grad()` — 参数 `grad` 归零（**必须**，因为反向是 `+=`）
3. `total_loss.backward()`
4. `p.data -= learning_rate * p.grad`

学习率 `1.0 - 0.9*k/100`，从 1.0 线性降到 0.1。没有 momentum、没有梯度裁剪。notebook 里大约 step 40 训练集准确率到 100%。决策边界画在 `moon_mlp.png`。

`loss()` 里 `sum(losses)` 从 Python 的 `0` 起加，靠 `__radd__`。和 `Neuron` 里显式传入 `self.b` 不是同一条路，都能跑。

---

## 10. 测试：PyTorch 当标准答案

`test/test_engine.py` 把同一串标量表达式各算一遍，比 `data` 和叶子 `grad`：

- `test_sanity_check`：单变量，含 ReLU
- `test_more_ops`：双变量，覆盖 README 那个「故意写得很绕」的例子；容差 `1e-6`

```bash
cd micrograd
python -m pytest
```

测的是引擎和 `torch.Tensor`（0 维 double）是否同数值，**不**测 `nn.py`、不测训练。`nn` 的正确性间接来自：它只调用已经对过的算子。

---

## 11. 缺什么、自己重写时不要抄什么

这份克隆缺的：张量、广播、批量 matmul、GPU、`no_grad`、高阶导、优化器、DataLoader、checkpoint。README 指向的进阶版是 [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)，对照见 [`microgpt-Analysis.md`](microgpt-Analysis.md)（前向时存局部导数，不再为每个算子挂闭包）和配套 [explainer](https://karpathy.github.io/2026/02/12/microgpt/)。讲座版还会加 `tanh`，这份仓库的 `engine.py` **没有** `tanh`。

| 可借 | 不要当脚手架继续堆 |
|------|-------------------|
| 四个原语 + 拓扑序反向，用来建立直觉 | 把 matmul / conv 也拆成 Python 标量循环 |
| `grad +=`、菱形图、`a+a` 与 `set` 的关系 | 递归 DFS 当生产级拓扑序 |
| hinge 二分类 demo 验证「图是对的」 | 在这份 `Value` 上训 Transformer |
| 用 PyTorch 0 维 tensor 对梯度 | 闭包风格一直用到张量引擎（microgpt / microai 都改成算子对象或局部导数表） |

本仓库已经有张量版：[ `microai.Variable` ](../microai/core.py) + [`microai.models.MLP`](../microai/models.py)。读 micrograd 是为了看见 AD 的最小核；接下来的路是「同一个链式法则，换成 ndarray 上的 `Function`」，不是把 `Value` 扩成框架。

由易到难对照这份代码：

1. 手算 §4 的菱形例子，对 `a.grad==21`、`b.grad==16`；再改成 `a+a`，确认梯度是 2 倍上游。
2. 跑 `test/test_engine.py`，改一个闭包里的 `+=` 成 `=`，看哪条测试炸。
3. 自己写时只留「标量图 + 拓扑反向」当笔记；真正训练走 `microai` 或 PyTorch。
