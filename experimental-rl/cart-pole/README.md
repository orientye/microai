# CartPole + Double DQN 示例解析

本目录用 **Double DQN** 训练 Gymnasium 的 `CartPole-v1`，并提供可视化测试脚本。

| 文件 | 作用 |
|------|------|
| `cartpole_train.py` | 训练：经验回放 + Double DQN + 软更新目标网络 |
| `cartpole_test.py` | 加载 `dqn_cartpole.pth`，纯贪心策略渲染 5 局 |
| `dqn_cartpole.pth` | 已训练好的策略网络权重 |
| `../rl001.py` | 对照：随机策略基线（无学习） |

```bash
# 训练（会弹出奖励曲线图）
python cartpole_train.py

# 测试（需已有 dqn_cartpole.pth，建议在本目录下运行）
python cartpole_test.py
```

依赖：`gymnasium[classic-control]`、`torch`、`numpy`、`matplotlib`。

---

## 1. 问题：CartPole 在学什么
https://gymnasium.farama.org/environments/classic_control/cart_pole/
CartPole-v1 是极简 MDP：小车上立一根杆，每步向左或向右推车，目标是尽量不让杆倒下。

| 概念 | 本例取值 |
|------|----------|
| 状态 `s` | 4 维：`[车位置, 车速度, 杆角度, 杆角速度]` |
| 动作 `a` | 2 个：`0` 左推，`1` 右推 |
| 奖励 `r` | 杆仍立着 → 每步 `+1` |
| `terminated` | 杆偏角过大 / 车出界 |
| `truncated` | 撑满 **500** 步（单局满分） |
| 通关判定 | 近 10 局均分 ≥ **450**（仅打印提示，当前不早停） |

`rl001.py` 用 `env.action_space.sample()` 随机动作，通常只能撑几十步。本例用神经网络近似 `Q(s,a)`，再贪心选动作。

---

## 2. 算法：从 DQN 到本例的 Double DQN

经典 Q-learning 目标：

```text
Q(s,a) ← r + γ · max_{a'} Q(s', a')
```

本例不用查表，而用 `QNet` 近似 `Q`，并叠加三项稳定技巧：

1. **经验回放（Replay Buffer）**  
   打断相邻样本的强相关，近似 i.i.d. 小批量采样。

2. **目标网络（Target Network）**  
   用较慢更新的 `target_net` 估计下一状态价值，避免「用正在变化的网络当标签」。

3. **Double DQN**  
   用 `policy_net` **选择** `a* = argmax_a Q_policy(s', a)`，用 `target_net` **估计** `Q_target(s', a*)`，减轻 `max` 带来的过估计。

数据流示意：

```text
环境 (s, r, s', done)
        │
        ▼
  ReplayBuffer ──sample──► batch
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         policy_net     policy_net     target_net
         Q(s,a)         选 a*          估 Q(s',a*)
              │              │              │
              └──────► MSE Loss ◄───────────┘
                         │
                       Adam
                         │
              policy_net ──软更新 τ──► target_net
```

代码里 `done = terminated or truncated`，目标里用 `(1 - dones)` 关掉 bootstrap。  
对 **terminated**（杆倒/出界）这是正确的：真实终局，没有未来价值。  
对 **truncated**（撑满 500 步）则偏保守：杆其实还立着，按严格 MDP 仍应 bootstrap；本例把截断也当终局处理，实现简单，但会略微低估「满分局」的后续价值。

对应核心逻辑（`cartpole_train.py`）：

```python
q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

with torch.no_grad():
    best_actions = self.policy_net(next_states).argmax(dim=1)
    max_next_q_values = self.target_net(next_states).gather(
        1, best_actions.unsqueeze(1)
    ).squeeze(1)
    expected_q_values = rewards + GAMMA * max_next_q_values * (1 - dones)

loss = F.mse_loss(q_values, expected_q_values)
# ... backward + step ...

# 目标网络软更新
for tp, p in zip(self.target_net.parameters(), self.policy_net.parameters()):
    tp.data.copy_(TAU * p.data + (1.0 - TAU) * tp.data)
```

---

## 3. 代码模块拆解

### 3.1 `QNet`：状态 → 两个 Q 值

结构：`4 → 64 → 64 → 2`，中间层 ReLU。对 CartPole 足够。  
输出是 **Q 值**，不是动作概率（后者属于策略梯度路线）。

#### Q 值是什么

`Q(s, a)` ≈「在状态 `s` 下选动作 `a`，从这一步起往后大概能攒多少分」。

口语里的「打分函数」在这里就是 **Q 函数（动作价值）**：给「局面 + 动作」打分。  
不要和 **状态价值 `V(s)`** 混淆——`V` 只问「到了这个局面大概能得多少」，不问左还是右；本仓库 PPO 的 Critic 学的是 `V`，DQN 的 `QNet` 学的是 `Q`。贪心时大约有 `V(s) ≈ max_a Q(s,a)`。

CartPole 每步奖励 `+1`，可粗想成「这样推下去还能撑多久（带折扣 γ）」。网络输出例如 `[2.0, 5.0]`，表示它认为当前局面下右推比左推更值；贪心时就 `argmax` 选右。

注意：

- 这是**估计值**，一开始乱猜，训练中被 TD 目标纠正。
- 有 `γ=0.99`，满分局的 Q 不会飙到 500，更常落在约一百量级（`k=0..499` 的折扣和 `Σ γ^k ≈ 100`）。
- Q **不是**这一步的即时奖励（即时奖励恒为 `+1`）；看的是未来一串回报。

`train_step` 只拧「真实走过的那个动作」对应的那一维：走了右，就主要让 `Q(s, 右)` 靠近老师给的目标 `y`。

#### 网络输入 / 输出

```text
输入: 状态，长度 4
      [车位置, 车速度, 杆角度, 杆角速度]
         │
      QNet（4 → 64 → 64 → 2，ReLU）
         │
输出: [Q(左), Q(右)]   ← 两个分数，无 softmax，不必加起来为 1
```

| 场景 | 怎么用这两个数 |
|------|----------------|
| 选动作（多数时候） | 谁大选谁（`argmax`） |
| 探索 | 不管分，随机推（ε-greedy） |
| 学习 | `gather` 取出真实动作那一维，与目标 `y` 做 MSE |

网络**不直接输出「推右」**，只打分；「推右」是对分数取最大得来的。

维 0 / 维 1 与动作编号绑定：环境里 `0=左推`、`1=右推`，`argmax` / `gather` 用同一编号。叫「Q(左)/Q(右)」是约定；有意义是因为训练时只纠正「当时那个动作」对应的那一维。

#### 层怎么把数吐出来（4 维小例子）

没有手写「左 = 位置×? + 角度×?」的公式。每一层都是：

```text
每个输出槽 = Σ (输入各维 × 对应权重) + 偏置
中间层再过 ReLU（负数变 0）
```

真实结构是 `4 → 64 → 64 → 2`。下面用缩小版 **`4 → 2 → 2`** 手算（规则相同，只是中间更窄）。

状态：

```text
s = [车位置, 车速度, 杆角度, 杆角速度]
  = [ 0.1 ,  0.0  ,  0.2  ,  -0.1   ]
```

第 1 层（假设权重，仅演示）：

```text
              中间0用的权重    中间1用的权重
车位置            1.0              0.0
车速度            0.0              1.0
杆角度            2.0              0.5
杆角速度          0.0             -1.0
偏置              0.0              0.1

raw0 = 0.1×1 + 0×0 + 0.2×2 + (-0.1)×0 + 0 = 0.5  → ReLU → h0=0.5
raw1 = 0.1×0 + 0×1 + 0.2×0.5 + (-0.1)×(-1) + 0.1 = 0.3 → ReLU → h1=0.3
```

输出层（`h → [Q左, Q右]`，再假设一组权重）：

```text
q0(左) = 0.5×2.0 + 0.3×0.0 + 0.1 = 1.1
q1(右) = 0.5×1.0 + 0.3×4.0 + 0.0 = 1.7
→ 输出 [1.1, 1.7]，贪心选右
```

真实代码对应：

```text
x = ReLU(W1·s + b1)     # fc1: 4→64
x = ReLU(W2·x + b2)     # fc2: 64→64
out = W3·x + b3         # out: 64→2
```

「矩阵连乘」口语上就是：**一层一层做加权求和**（中间夹 ReLU）；不是先把三个矩阵乘成一个再乘状态。

#### 在学习什么？参数有多少？

学习的不是「怎么做乘法」（算法写死），而是 **权重表里那些小数 `W、b`**，让吐出的 `[Q左, Q右]` 越来越准。

- 概念上：学一个打分函数 `s → [Q(s,左), Q(s,右)]`，即 **动作价值函数 Q**（不是 `V(s)`）
- 实现上：改几千个参数；打分准了，`argmax` 看起来就像会平衡杆

一份 `QNet`（`4→64→64→2`）参数量为 **4610**：

| 参数 | shape | 个数 |
|------|-------|------|
| `fc1.weight` | (64, 4) | 256 |
| `fc1.bias` | (64,) | 64 |
| `fc2.weight` | (64, 64) | 4096 |
| `fc2.bias` | (64,) | 64 |
| `out.weight` | (2, 64) | 128 |
| `out.bias` | (2,) | 2 |
| **合计** | | **4610** |

训练时还有一份同结构的 `target_net`（慢更新副本）；`dqn_cartpole.pth` 只保存 `policy_net`。  
`cartpole_test.py` 加载后会打印上述总量与逐层个数，可自行核对。

#### 为什么有两个网（`policy_net` / `target_net`）

结构都是 `QNet`，角色不同：

| 名字 | 角色 |
|------|------|
| `policy_net` | 正在学的打分员：选动作、算当前 `Q(s,a)` |
| `target_net` | 慢更新的「标准答案」：参与算 TD 目标 `y`（本例为软更新） |

一步学习在做什么：

1. 当前分：`q = Q_policy(s, a)`
2. 老师分（Double DQN，且 `no_grad`）：policy 在 `s'` 上选 `a*`，target 估值 → `y = r + γ · Q_target(s', a*)`（`done` 则切断）
3. 损失 `(q - y)²`，只更新 `policy_net`，让 `q` **靠近** `y`
4. 再用很小的 `τ` 把 policy 渗一点进 target

若目标也用正在更新的同一张网算，标签与被更新对象绑在一起，容易抖、容易过估计。`target_net` 故意慢半拍，等于作业答案先固定一会儿。

推杆 / 存盘 / 测试用的都是 `policy_net`；`target_net` **不参与选动作**。

更细的手推数字与公式见 [`../cart-pole-deep-dive.adoc`](../cart-pole-deep-dive.adoc) §1、§7。

### 3.2 `ReplayBuffer`

- 容量 `MEMORY_SIZE = 10000`，`deque(maxlen=...)` 满则丢最旧。
- `push` 存五元组 `(s, a, r, s', done)`。
- `sample` 均匀随机抽 `BATCH_SIZE` 条，转成 tensor。

未使用 Prioritized Experience Replay；对本玩具环境通常够用。

### 3.3 `DQNAgent`

| 方法 | 作用 |
|------|------|
| `choose_action` | ε-greedy：以 ε 随机探索，否则 `argmax Q` |
| `train_step` | 池子 ≥ `MIN_MEMORY_SIZE` 才学习；Double DQN 目标；MSE；按步衰减 ε；软更新 target |

探索率按 **成功执行的 `train_step` 次数**（不是按 episode，也不是按环境步）指数衰减：

```text
ε = ε_end + (ε_start − ε_end) · exp(−t / EPS_DECAY_STEPS)
```

本例：`1.0 → 0.01`，衰减常数 `5000`。  
热身阶段（buffer < 1000）`train_step` 直接 return，此时 **ε 不衰减、target 也不软更新**，ε 一直保持 `1.0`。比「每回合降一点」更平滑，也更早进入利用阶段。

### 3.4 训练主循环

每个 episode：

1. `env.reset()` → 循环 `choose_action` → `env.step`
2. 每步：`memory.push` + `train_step`（交互与学习交织）
3. 每 10 局：打印近 10 局均分与当前 ε；≥450 打印通关；若均分刷新历史最佳则覆盖保存 `dqn_cartpole.pth`

「保存最佳」只在每 10 局检查时触发。若全程文件仍不存在（例如从未进入过该分支），收尾会兜底保存**训练结束时**的 `policy_net`（不是「只含最后一局」的含义）。

### 3.5 `cartpole_test.py`

加载同结构 `QNet`，**不做探索**（纯 `argmax`），`render_mode="human"` 演示 5 局。  
加载成功后会打印参数总量（应为 **4610**）及各层 `weight`/`bias` 的 shape 与个数。  
注意：训练/测试中的 `QNet` 是复制粘贴的两份，修改结构时需两边同步。

---

## 4. 超参数含义

| 参数 | 值 | 直觉 |
|------|-----|------|
| `LR` | `1e-3` | Adam 对小网络偏激进，CartPole 上通常能收敛 |
| `GAMMA` | `0.99` | 很在乎远期回报；满分 500 步时远期仍有权重 |
| `BATCH_SIZE` | `64` | 梯度噪声与稳定性的折中 |
| `MEMORY_SIZE` | `10000` | 回放池容量 |
| `MIN_MEMORY_SIZE` | `1000` | 先乱玩攒经验，再开始更新 |
| `TAU` | `0.005` | 每步 target 只跟随 0.5%，比周期性硬拷更稳 |
| `TARGET_UPDATE` | `10` | **软更新启用后实际未使用**（遗留常量） |
| `EPS_START / END` | `1.0 / 0.01` | 探索率上下界 |
| `EPS_DECAY_STEPS` | `5000` | 按成功 `train_step` 次数指数衰减的时间常数 |
| `MAX_EPISODES` | `500` | 本环境上通常几十到一两百局即可学好 |

通关阈值用 450（而非 500）留了余量，降低偶发满分导致的误判感。

---

## 5. 一次成功训练大致经历的阶段

1. **热身（约前 1000 条 transition / 环境步）**  
   只往 buffer 写数据，不反传、不衰减 ε、不软更新。ε 保持 `1.0`，几乎全随机，局分很低。

2. **探索 + 学习**  
   ε 指数下降；Q 开始区分「杆往哪倒、车该往哪推」。局分从几十爬到一两百。

3. **利用主导**  
   ε→0.01；策略接近对角度/角速度的非线性「比例控制」。近 10 局均分冲向 400+。

4. **保存最佳**  
   按近 10 局均分存盘，减轻「最后一局未必最好」的问题。

目录中若已有 `dqn_cartpole.pth`，说明至少成功训练过一次。

---

## 6. 与随机基线对照

| | `rl001.py` | `cartpole_train.py` |
|--|-----------|---------------------|
| 策略 | `action_space.sample()` | 学出的 `Q` + ε-greedy |
| 记忆 | 无 | ReplayBuffer |
| 学习信号 | 无 | `r + γ · Q_target(s', a*)` |
| 期望表现 | 几十步 | 接近 500 步 |

CartPole 难的不是动力学本身，而是：

- **信用分配**：早期动作影响很晚才是否倒下；
- **相关数据导致训练不稳定**。

Replay + Target Network + Double DQN 主要就是针对这两点。

---

## 7. 当前实现的亮点与已知不足

**亮点**

- Double DQN + 软更新，比「裸 DQN + 硬拷贝」更稳。
- ε 按训练步衰减，与真实交互量对齐。
- 按历史最佳均分存盘，而非只存最后一局。
- Windows 下强制 stdout/stderr UTF-8，避免日志里 emoji 触发编码错误。

**已知不足（尚未改代码）**

- `TARGET_UPDATE` 为死代码，易误导读者以为仍在硬更新。
- `QNet` 在 train/test 各写一份，结构易漂移。
- 通关后不早停，会继续跑满 `MAX_EPISODES`。
- 每环境步都调用 `train_step`（池满后每步都反传）：对本环境可接受；更大环境常改为每 N 步学一次。
- `done` 把 `truncated` 与 `terminated` 同等对待，满分截断时不 bootstrap。
- 损失用 MSE；Huber（Smooth L1）对大 TD 误差通常更稳。
- 未固定随机种子，曲线不易复现。
- `torch.load` 未指定 `map_location` / `weights_only`。
- 训练与测试都用相对路径 `dqn_cartpole.pth`，依赖当前工作目录。

这些可作为后续可选优化，当前文档对应的是**现有代码行为**。

---

## 8. 可选优化清单（暂未落地）

若以后要改，建议优先级：

1. 通关后早停  
2. 抽出公共 `QNet` 供 train/test 共用  
3. 删除或真正启用 `TARGET_UPDATE`（硬/软更新可切换）  
4. `torch.load(..., map_location="cpu", weights_only=True)`；权重路径相对 `__file__`  
5. 固定 seed；Huber loss；梯度裁剪  
6. 学习频率可配置（每 N 步学 1 次 / 每步学 K 次）  

对 CartPole 来说，Prioritized Replay、Dueling、NoisyNet、多环境并行等收益有限、复杂度偏高，一般不必为这个例子上。换成 PPO 则是另一条算法路线，不属于「优化当前这段 DQN」。

---

策略梯度 / 简化 PPO 对照示例见 [`../cart-pole-ppo/`](../cart-pole-ppo/)。

公式推导、信用分配、`train_step` / PPO 逐行精读见 [`../cart-pole-deep-dive.adoc`](../cart-pole-deep-dive.adoc)（AsciiDoc）。零基础请先读文中 **「新手速通」** 再进后面章节。
