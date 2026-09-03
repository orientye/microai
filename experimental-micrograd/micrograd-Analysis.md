# micrograd 源码分析

对照说明：[`micrograd/README.md`](micrograd/README.md)。  
官方仓库：[karpathy/micrograd](https://github.com/karpathy/micrograd)。  
本文件按**这份克隆的真实源码**拆，不按讲座口头描述。

阅读建议：

- 先扫 §1～§2，搞清「它在算什么、和本仓库 `microai` 差在哪」。
- 再按 §3 的读序进源码；§4～§7 是路径精读。
- 若准备自己重写，最后看 §9。

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
experimental-micrograd/
  micrograd-Analysis.md   # 本文件
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

1. `micrograd/engine.py` 的 `Value.__add__` / `__mul__` — 前向怎么长边
2. 同一个文件的 `backward()` — 拓扑序 + `grad +=`
3. `micrograd/nn.py` 的 `Neuron.__call__` — 点积是标量循环
4. `demo.ipynb` 的 `loss()` 和 SGD 循环 — hinge + `zero_grad`
5. `test/test_engine.py` — 用 PyTorch 当标准答案

`engine.py` 远重于 `nn.py`。先把 `a + a` 和菱形图的累加读通，比抠 MLP 层数有用。

---

## 4. 一次前向+反向数据怎么走

```text
叶子 Value（参数、输入）          grad 初值 0，_backward 是空 lambda
    ↓ 每次 + * ** relu 新建一个 Value
    ↓ 子节点放进 _prev；反向规则闭包进 _backward
输出 L
    ↓ L.backward()
DFS 建拓扑序（子先于父）
    ↓ L.grad = 1
从输出往叶子，逐个调 _backward()
    ↓ 链式法则写进父节点.grad（用 +=）
叶子上的 .grad 就是 dL/d(该叶子)
    ↓ SGD：p.data -= lr * p.grad
下一轮必须 zero_grad，否则 += 会把上一轮的梯度叠进来
```

图是动态的：下一轮前向是一批**新的**中间 `Value`。参数叶子是同一批对象，只改 `.data`。

`trace_graph.ipynb` 的 `draw_dot` 把每个节点标成 `data | grad`，边经过 `_op` 盒子。那是调试工具，不是训练路径。

---

## 5. `Value`：标量 DAG

### 5.1 节点里有什么

`engine.py`：

```python
self.data = data
self.grad = 0
self._backward = lambda: None
self._prev = set(_children)
self._op = _op
```

| 字段 | 作用 |
|------|------|
| `data` | 前向数值 |
| `grad` | `d(backward根)/d(自己)`，默认 0 |
| `_backward` | 把 `out.grad` 分配给父母；叶子是空函数 |
| `_prev` | 父母集合，给拓扑序和画图 |
| `_op` | 字符串，只给 graphviz |

没有 `requires_grad`：所有节点都带梯度。没有 `Function` 类：局部导数活在闭包里，闭包捕获前向时的 `self` / `other` / `out`。

### 5.2 四个原语，其余是语法糖

真正写了 `_backward` 的只有四个：

| 算子 | 前向 | 反向（累加到父母） |
|------|------|-------------------|
| `+` | `self.data + other.data` | `self.grad += out.grad`；`other` 同 |
| `*` | `self.data * other.data` | `self.grad += other.data * out.grad`；对称 |
| `**` | `self.data ** other` | `self.grad += other * self.data**(other-1) * out.grad` |
| `relu` | `0 if self.data < 0 else self.data` | `self.grad += (out.data > 0) * out.grad` |

Python 数会包成 `Value`：`other if isinstance(other, Value) else Value(other)`。所以 `2 * x`、`x + 1` 能跑。

派生算子**不**单独写反向，全走原语：

| 写法 | 实际 |
|------|------|
| `-self` | `self * -1` |
| `self - other` | `self + (-other)` |
| `self / other` | `self * other**-1` |
| `__radd__` / `__rmul__` | 转到正向，好让 `2 + x` 成立 |

`**` 的指数必须是 `int`/`float`，不能 `Value ** Value`。除法对 0、负数的分数次幂都没护栏。

`relu` 在 0 处用 `out.data > 0`，子梯度取 0，和 PyTorch 默认一致。

### 5.3 `backward()`：拓扑序 + 闭包

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

后序 DFS：父母先进 `topo`，自己后进。`reversed` 之后从输出走向叶子。每个 `_backward` 只知道「怎么把 `out.grad` 分给直接父母」，深层靠链式法则一层层传。

根上是**赋值** `self.grad = 1`，不是 `+=`。中间节点全靠闭包里的 `+=`。对同一个 `Value` 调两次 `backward()` 而不 `zero_grad`：根被重置成 1，中间却再累加一遍——不要这么用。demo 里每次都是 `zero_grad` 再 `backward` 一次。

`build_topo` 是递归的。Python 默认递归深度 1000，图很深会爆。教学规模没事。

### 5.4 累加、菱形、`a + a`

多元链式法则要求：同一节点若被多用，梯度要**加**起来。所以闭包一律 `+=`，不是 `=`。

菱形（`c = a * b; d = a + b; e = c * d`）靠拓扑序保证 `c`、`d` 都算完才动 `a`，再靠 `+=` 把两条路加在一起。

`c = a + a` 有个实现细节：

```python
out = Value(self.data + other.data, (self, other), '+')
self._prev = set(_children)   # set((a, a)) == {a}
```

`_prev` 只留一个 `a`，画图少一条边，拓扑序里 `a` 也只出现一次——这是对的。反向仍然正确，因为闭包捕获了 `self` 和 `other` 两个名字，对同一个对象 `+=` 两次。梯度走的是闭包，不是 `_prev`。若有人改成「遍历 `_prev` 均分梯度」，`a + a` 会少一半。

---

## 6. `nn`：神经元就是标量循环

### 6.1 `Module`

`zero_grad` 把每个参数的 `grad` 写成 0。`parameters()` 默认空列表。没有 `eval()` / `train()`，也没有保存权重。

### 6.2 `Neuron`

```python
self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
self.b = Value(0)
act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
return act.relu() if self.nonlin else act
```

点积是 `zip` 上逐项 `wi*xi`，`sum(..., self.b)` 从 bias 起加，避免 `sum` 默认从 `0` 起（虽然 `__radd__` 也能扛）。输入比 `nin` 长会被 `zip` 丢掉，短则少乘几项——没有形状检查。

`nonlin=True`（默认）接 ReLU；最后一层关掉，输出是打分，不是概率。

### 6.3 `Layer` / `MLP`

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

## 7. `demo.ipynb`：hinge + L2 + SGD

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

## 8. 测试：PyTorch 当标准答案

`test/test_engine.py` 把同一串标量表达式各算一遍，比 `data` 和叶子 `grad`：

- `test_sanity_check`：单变量，含 ReLU
- `test_more_ops`：双变量，覆盖 README 那个「故意写得很绕」的例子；容差 `1e-6`

```bash
cd micrograd
python -m pytest
```

测的是引擎和 `torch.Tensor`（0 维 double）是否同数值，**不**测 `nn.py`、不测训练。`nn` 的正确性间接来自：它只调用已经对过的算子。

---

## 9. 缺什么、自己重写时不要抄什么

这份克隆缺的：张量、广播、批量 matmul、GPU、`no_grad`、高阶导、优化器、DataLoader、checkpoint。README 指向的进阶版是 [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)（前向时存局部导数，不再为每个算子挂闭包）和配套 [explainer](https://karpathy.github.io/2026/02/12/microgpt/)。讲座版还会加 `tanh`，这份仓库的 `engine.py` **没有** `tanh`。

| 可借 | 不要当脚手架继续堆 |
|------|-------------------|
| 四个原语 + 拓扑序反向，用来建立直觉 | 把 matmul / conv 也拆成 Python 标量循环 |
| `grad +=`、菱形图、`a+a` 与 `set` 的关系 | 递归 DFS 当生产级拓扑序 |
| hinge 二分类 demo 验证「图是对的」 | 在这份 `Value` 上训 Transformer |
| 用 PyTorch 0 维 tensor 对梯度 | 闭包风格一直用到张量引擎（microgpt / microai 都改成算子对象或局部导数表） |

本仓库已经有张量版：[ `microai.Variable` ](../microai/core.py) + [`microai.models.MLP`](../microai/models.py)。读 micrograd 是为了看见 AD 的最小核；接下来的路是「同一个链式法则，换成 ndarray 上的 `Function`」，不是把 `Value` 扩成框架。

由易到难对照这份代码：

1. 手算 `e = (a * b + c).relu()`，再 `e.backward()`，对一下 `a.grad`。
2. 跑 `test/test_engine.py`，改一个闭包里的 `+=` 成 `=`，看哪条测试炸。
3. 自己写时只留「标量图 + 拓扑反向」当笔记；真正训练走 `microai` 或 PyTorch。
