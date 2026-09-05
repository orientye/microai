# microgpt 源码分析（相对 micrograd）

对照说明：[explainer](https://karpathy.github.io/2026/02/12/microgpt/)；gist：[microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)。  
micrograd 拆解：[`micrograd-Analysis.md`](micrograd-Analysis.md)。  
本文件按**这份克隆的真实源码**拆：[`microgpt/microgpt.py`](microgpt/microgpt.py)（约 200 行，无依赖）。

阅读建议：

- 已读过 micrograd 的 §4～§7（`Value` / 前向 / 反向 / 链式法则）再进本文。
- **§1～§3 先对引擎**：microgpt 改的是 AD，不是另起一套数学。
- §4～§7 才是 GPT：数据、结构、训练、采样。

---

## 1. 一句话

microgpt 用**同一套标量 DAG + 反向 AD**训一个 GPT-2 骨架：字符级词表、单层 4 头注意力、交叉熵、Adam，在 32k 个人名上做 next-token，再自回归采样新名字。

相对 micrograd，算法核没变（拓扑序 + `grad += 局部导 × 上游`）。变的是：局部导数**前向时存成数**，不再为每个算子挂闭包；上面不再是 `nn.MLP` + hinge，而是一个函数 `gpt(...)` + softmax 交叉熵。

---

## 2. 和 micrograd 对照

| | **micrograd** | **microgpt** |
|--|---------------|--------------|
| 文件 | `engine.py` + `nn.py` + demo | **单文件** `microgpt.py` |
| `Value` 图语义 | `_backward` 闭包 | `_local_grads` 元组（前向算死） |
| 邻接 | `_prev = set(...)` | `_children` **元组**（可重复） |
| 可视化 | `_op` + `trace_graph` | 没有 |
| 原语 | `+ * ** relu` | 再加 `log` / `exp`（为 softmax / CE） |
| 网络 | `Neuron`/`Layer`/`MLP` 对象 | 无类：`linear` / `softmax` / `rmsnorm` / `gpt` |
| 损失 | hinge + L2 | 逐步 `-log p(target)` 再平均 |
| 优化 | 手写 SGD：`p.data -= lr * p.grad` | Adam + 线性衰减；更新后 `p.grad = 0` |
| 任务 | moons 二分类 | 人名 next-token → 采样新名字 |
| 参数量 | 337 | **4192** |

micrograd README 里说的进阶版就是这一句：*storing local gradients at forward time instead of per-op backward closures*。源码对得上。

链式法则本身没换。microgpt 官方手算（explainer，不是我们文档里的菱形）：

```python
a = Value(2.0)
b = Value(3.0)
c = a * b       # 6
L = c + a       # 8
L.backward()
# a.grad == 4   （两条路：b + 1）
# b.grad == 2
```

和 [`micrograd-Analysis.md`](micrograd-Analysis.md) 的菱形是同一类图：`a` 被用两次，靠 `+=` 把路径加起来。

---

## 3. 目录与读序

```text
experimental-karpathy/
  micrograd-Analysis.md
  microgpt-Analysis.md      # 本文件
  micrograd/                # 上游 micrograd
  microgpt/
    microgpt.py             # 全文
    input.txt               # 首次运行下载 names.txt（gitignore）
```

```bash
cd experimental-karpathy/microgpt
python microgpt.py          # 无 pip；首次拉 names；约 1000 step
```

建议读序（对着 `microgpt.py` 行号）：

1. `Value`（约 L30–72）— 只看和 micrograd 的差：`_local_grads`、元组 `_children`、统一 `backward`
2. 数据 + tokenizer（L14–27）
3. `state_dict` / `params`（L74–90）
4. `linear` / `softmax` / `rmsnorm` / `gpt`（L94–144）
5. 训练循环 + Adam（L146–184）
6. 推理采样（L186–200）

`Value` 仍远重于后面的 GPT 公式。引擎没读通，注意力只是一堆 `*`/`+`。

---

## 4. 数据结构：相对 micrograd 改了什么

还是一个 `Value` 当节点。字段从五个换成四个，职责重分：

| micrograd | microgpt | 变了什么 |
|-----------|----------|----------|
| `data` | `data` | 同：标量前向值 |
| `grad` | `grad` | 同：初值 0，含义 `∂根/∂自己` |
| `_prev`（`set`） | `_children`（**tuple**） | 可重复；`a+a` 是 `(a,a)` 不是 `{a}` |
| `_backward` 闭包 | `_local_grads` 元组 | 局部导在**前向**算成 `float` 存下来 |
| `_op` | （删） | 不画图 |
| （无） | `__slots__` | 省 `__dict__`，图很大时省内存 |

### 4.1 前向把局部导冻住

micrograd 乘法：

```python
def _backward():
    self.grad += other.data * out.grad   # 反向时再读 .data
    other.grad += self.data * out.grad
```

microgpt 乘法：

```python
return Value(self.data * other.data, (self, other), (other.data, self.data))
```

`(other.data, self.data)` 是两个 **Python 数**，造节点时拷下来。`backward` 不再读输入的 `.data`。

后果：即使有人在 `backward()` **之前**改了 `p.data`，局部导仍是前向那一拍的。micrograd 会算错（见 micrograd 分析 §5.2）。microgpt 的训练顺序仍是先 backward 再改 `data`，但引擎不再依赖这个顺序。

### 4.2 反向不再调闭包

```python
self.grad = 1
for v in reversed(topo):
    for child, local_grad in zip(v._children, v._local_grads):
        child.grad += local_grad * v.grad
```

这就是链式法则的一行版：

```text
child.grad += (∂v/∂child) * v.grad
```

micrograd 每个算子自己写一遍 `+=`；这里所有算子共用这一循环。加原语 = 前向多返回一对 `(children, local_grads)`，不必再写 `_backward`。

| 算子 | 前向 `data` | `_local_grads` |
|------|-------------|----------------|
| `a + b` | `a+b` | `(1, 1)` |
| `a * b` | `a*b` | `(b, a)` |
| `a ** n` | `a**n` | `(n * a**(n-1),)` |
| `log(a)` | `ln a` | `(1/a,)` |
| `exp(a)` | `e^a` | `(e^a,)` |
| `relu(a)` | `max(0,a)` | `(float(a>0),)` |

`log` / `exp` 是为 softmax 和 `-log p` 新加的。`relu` 用 `max` + `float(data > 0)`，0 处子梯度仍是 0，和 micrograd 的 `out.data > 0` 同。

派生算子仍是宏：`-`、`/` 走 `*` 和 `**`，和 micrograd 一样。

### 4.3 `a + a`：不再靠闭包「写两次」

micrograd：`_prev = {a}`，闭包里 `self`、`other` 两个名字对同一对象各 `+=` 一次。

microgpt：`_children = (a, a)`，`_local_grads = (1, 1)`，`zip` 循环两次。骨架和语义对齐了。若有人改回 `set`，这里会少一半——和 micrograd 把反向改成「遍历 `_prev`」是同一个坑，方向相反。

拓扑 DFS 仍用 `visited`：同一个 `a` 只入序一次。累加发生在下游节点的 `zip` 里，不是叶子的 `_backward`（叶子没有局部导，元组为空）。

### 4.4 和 `microai` 的位置

| | micrograd | microgpt | microai |
|--|-----------|----------|---------|
| 局部导存在哪 | 闭包 | 节点上的数表 | `Function.backward` |
| 数据 | 标量 | 标量 | ndarray |
| 适合 | 看懂 AD | 在同一引擎上堆完整 GPT 循环 | 真训练 |

microgpt 的 `Value` 更接近「把闭包求值提前」；要张量仍然得换 `microai` / PyTorch。

---

## 5. 前向：从 MLP 打分到 next-token

### 5.1 数据与词表

首次运行若没有 `input.txt`，下载 [makemore/names.txt](https://github.com/karpathy/makemore)。约 32033 个名字，一行一个文档。

字符集排序后当 token id：`a–z` → `0..25`，再加一个 `BOS = 26`。`vocab_size = 27`。没有 BPE。

训练时 `"emma"` 变成 `[BOS, e, m, m, a, BOS]`。模型在每个位置预测下一个；第二个 `BOS` 表示「名字结束」。

### 5.2 参数是嵌套 list，不是 `Module`

```python
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
```

`state_dict` 里每张权重都是 `list[list[Value]]`。展平：

```python
params = [p for mat in state_dict.values() for row in mat for p in row]
```

没有 bias（GPT-2 简化）。`n_layer=1, n_embd=16, n_head=4, block_size=16`：

| 矩阵 | 形状 | 个数 |
|------|------|------|
| `wte` | 27×16 | 432 |
| `wpe` | 16×16 | 256 |
| `lm_head` | 27×16 | 432 |
| `attn_wq/k/v/o` | 4 × 16×16 | 1024 |
| `mlp_fc1` | 64×16 | 1024 |
| `mlp_fc2` | 16×64 | 1024 |
| 合计 | | **4192** |

和脚本打印的 `num params` 对得上。每个数仍是独立 `Value`，和 micrograd 的 337 个叶子同一套路，只是多了两个数量级的中间节点。

### 5.3 三个积木

`linear(x, w)`：对 `w` 每一行做 `sum(wi*xi)`——就是 micrograd `Neuron` 的点积，去掉 bias，批量到矩阵。

`softmax`：先减 `max(val.data)`（Python 数，为了 `exp` 不炸），再 `exp`、求和、相除。减 max 在图上是 `val - 常数`，梯度透传（平移不改 softmax）。

`rmsnorm`：`scale = (mean(x²) + 1e-5) ** -0.5`，再逐元乘。没有可学的 gain。嵌入后先做一次 rmsnorm；注释写 *not redundant due to backward pass via the residual connection*——残差旁路还握着未归一的 `x`，这条路上的梯度不经过后面的 norm。

### 5.4 `gpt`：一次一个 token

```text
token_id, pos_id, keys, values
    → wte[token] + wpe[pos]
    → rmsnorm
    → 每层：Attn（残差）→ MLP（残差）
    → lm_head → 27 维 logits
```

相对 GPT-2 的简化（文件头注释）：LayerNorm → RMSNorm，去掉 bias，GeLU → ReLU。位置编码仍是可学的 `wpe`，不是 RoPE。

注意力逐步走：

1. `q,k,v = linear(x, W*)`，`k`/`v` **append 进 KV cache**
2. 4 头切 `head_dim=4`
3. `attn_logits[t] = q·k_t / √d`，softmax，再对 `v` 加权
4. 拼 4 头，乘 `attn_wo`，加残差

MLP：`16 → 64 → ReLU → 16`，再残差。和 micrograd 的 `MLP(16,[64,16])` 同形，只是权重是矩阵而不是 `Neuron` 列表，且没有输出层的「最后一层线性」这种对象——线性就是 `linear`。

**没有时间并行、没有 batch。** 一个名字里每个位置调一次 `gpt`。KV cache 在训练里也显式存在；cache 里的 `k`/`v` 是活着的 `Value`，`loss.backward()` 会穿过它们。这和推理时「detach 后的 KV」不同。explainer 强调：并行实现里 cache 被藏在矩阵乘法里，这里一步一个 token，所以必须看见它。

---

## 6. 反向与损失：相对 hinge 换了什么

### 6.1 逐步交叉熵

```python
probs = softmax(logits)
loss_t = -probs[target_id].log()
loss = (1 / n) * sum(losses)
loss.backward()
```

`n = min(block_size, len(tokens)-1)`。随机猜 27 类：`-log(1/27) ≈ 3.3`。explainer 说 1000 step 后大约到 2.37。

micrograd demo 是 `max(0, 1 - y·score)`，输出一个标量打分。这里输出 27 维 logits，损失只盯**正确那个** token 的概率。数学上仍是一个标量根，一次 `backward()` 灌满所有 4192 个参数的 `grad`。

图比 demo 大得多：一个名字约 10 个位置 × 一层注意力（对过去每步都有 `q·k`）× 标量展开。还是递归 DFS 拓扑，Python 默认深度 1000 对「最长名 15」够用。

### 6.2 Adam，不是 SGD

micrograd：`p.data -= lr * p.grad`，`lr` 线性从 1.0 降到 0.1，另有 `zero_grad()`。

microgpt：

```python
m[i] = β1 m + (1-β1) g
v[i] = β2 v + (1-β2) g²
p.data -= lr_t * m̂ / (√v̂ + ε)
p.grad = 0
```

`β1=0.85, β2=0.99`（比常见的 0.9/0.999 更短记忆），`lr` 从 0.01 线性到 0。`m`/`v` 是普通 `float`，不进计算图。

没有 `Module.zero_grad`：更新完当场把叶子 `grad` 写成 0。中间节点每步都是新图，不用清。

### 6.3 推理不调 `backward`

从 `BOS` 起，softmax（可除 `temperature`）后 `random.choices`，采到 `BOS` 或 `block_size` 停。只用 `.data`。

没有 `no_grad`：前向照样建图、挂 `_local_grads`，只是从不 `backward`，图随循环丢掉。温度 0.5 把分布削尖，偏保守。

---

## 7. 一张表对上 micrograd 的四个维度

用 explainer 的分支例子 `L = a*b + a`（`a=2,b=3`）：

| 维度 | micrograd（我们的菱形） | microgpt（这份源码） |
|------|-------------------------|----------------------|
| 数据结构 | `_prev` set + `_backward` 闭包 | `_children` 元组 + `_local_grads` 数表 |
| 前向 | 算 `data`，闭包捕获对象 | 算 `data`，**同时写出局部导** `(b,a)`、`(1,1)` |
| 反向 | `reversed(topo)` 调闭包 | 同一拓扑，`zip` 做 `g += local * v.grad` |
| 链式法则 | `∂e/∂a = 15+6=21`（菱形） | `∂L/∂a = b+1=4`；`+=` 仍然是多路径求和 |

网络层：micrograd 把同一引擎接在 hinge MLP 上；microgpt 接在「字符 GPT + CE + Adam + 采样」上。引擎升级是 `_local_grads`；其余是任务。

---

## 8. 缺什么、不要抄什么

gist 是完整算法，不是完整系统。缺的和 explainer「Real stuff」一节一致：BPE、张量/GPU、FlashAttention、RoPE/GQA/MoE、大 batch、混合精度、SFT/RL、推理分页。

| 可借 | 不要当脚手架继续堆 |
|------|-------------------|
| `_local_grads` + 统一 `backward`（比闭包干净） | 在这份标量 `Value` 上加层数 / 加 vocab 当「真 GPT」 |
| 显式 KV cache（训练也穿过图） | 把 cache 写成 detach 再当训练实现 |
| 逐步 CE + Adam 的最小循环 | 从 `microgpt.py` 抄并行、checkpoint、tokenizer |
| 和 micrograd 对同一菱形 / `a+a` | 闭包风格和新的数表混在一个 `Value` 里 |

由易到难：

1. 把 micrograd 菱形和 explainer 的 `L=a*b+a` 都在 **两个** `Value` 上跑一遍，确认 21/16 与 4/2。
2. 只读 `Value` 到 `backward` 结束，挡住 GPT 细节，直到能默写 `child.grad += local * v.grad`。
3. 再读 `gpt` + 训练循环。真要训语言模型，换 PyTorch / 本仓库 `microai`，不要扩这个文件。
