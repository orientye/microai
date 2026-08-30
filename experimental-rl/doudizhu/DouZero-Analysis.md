# DouZero 源码分析

对照说明、读序、能跑什么：[`README.md`](README.md)。  
官方仓库：[kwai/DouZero](https://github.com/kwai/DouZero)（ICML 2021，[论文](https://arxiv.org/abs/2106.06135)）。  
本文件按**这份克隆的真实源码**拆，不按论文口头描述。

阅读建议：

- 先扫 §1～§2，搞清「它在学什么、和 DQN/PPO 差在哪」。
- 再按 §3 的读序进源码；§4～§7 是路径精读。
- 若准备自己重写，最后看 §9。

---

## 1. 一句话

DouZero 用 **Deep Monte-Carlo**：自博弈打完整局，把每个 `(s, a)` 的 Q 值回归到**同一局的终局回报 G**，动作用 54 维牌矩阵编进输入，不对两万维动作头做 `max Q` 或 softmax。

它解决的是斗地主叠在一起的四件事：变长合法集、终局稀疏奖励、部分可观测、三人合作/对抗。算法本身比本仓库的 PPO 更朴素；难的是规则、特征和并行自博弈工程。

---

## 2. 和 Gym 主线对照

| 主线里已有 | 斗地主上变成 | DouZero 的做法 |
|------------|--------------|----------------|
| GridWorld 动作掩码 | 每回合合法集不同，规模到 2 万+ | 只对**当前合法动作**各打一个 `Q(s,a)` |
| CartPole DQN：`Q(s)` 出 2 维 | 输出维不能等于动作数 | 动作编成 54 维，和状态拼在一起 |
| CartPole PPO：Categorical | 合法集每步都在变 | 不学 π，ε-greedy 选 Q 最大的合法动作 |
| MountainCar 终局才成功 | 中间步 `r=0` | 整局同一个 G，γ=1，不 bootstrap |
| 单智能体 | 地主 vs 两农民 | 三位置三套网，农民 target = −地主 G |

所以论文里 DQN、A3C 几乎打不过规则 Bot：不是实现差，是假设不对——动作太多时 `max Q` 会高估，固定维策略头也撑不住变长合法集。

| | CartPole DQN | **DouZero DMC** | CartPole PPO |
|--|--------------|-----------------|--------------|
| 学什么 | `Q(s)` 出 2 个头 | **`Q(s,a)` 标量** | `π(a\|s)` |
| 目标 | `r + γ max Q'`（TD） | **整局同一个 G**（MC） | GAE + clip |
| 动作怎么进网络 | 输出维 = 动作数 | **54 维牌矩阵拼进输入** | Categorical |
| 探索 | ε-greedy | 合法下标上 ε=0.01 | 采样 + 熵 |
| 信用分配 | TD 一步步渗 | 整局每步背同一个锅 | GAE 按步衰减 |

公式直觉见 [`../cart-pole/cart-pole-deep-dive.adoc`](../cart-pole/cart-pole-deep-dive.adoc)「整局 return」；这里拟合的是 Q，不是 `log π`。

---

## 3. 目录与读序

```text
DouZero/
  train.py / evaluate.py / generate_eval_data.py
  douzero/
    env/          # 规则 + 特征（该先读）
    dmc/          # Deep Monte-Carlo 训练
    evaluation/   # 固定牌谱对打
  baselines/      # 空的，预训练权重需另下
```

没有叫牌。`reset()` 直接洗牌：地主 20 张、两农民各 17 张。依赖：`torch`；评估里的规则 Bot 还要 `rlcard`。

建议读序：

1. `env/game.py` — 规则、合法出牌、`InfoSet`
2. `env/env.py` — `_cards2array`、`get_obs`、`_get_reward`、`DummyAgent`
3. `evaluation/deep_agent.py` — 对合法动作逐个打 Q，argmax
4. `dmc/models.py` — 地主 / 农民两套 LSTM+MLP，输出 1 维
5. `dmc/utils.py` 的 `act()` — 终局回填同一个 G
6. `dmc/dmc.py` 的 `learn()` — MSE + 权重拷回 Actor

`env/`（规则+特征）远重于 `dmc/` 的公式。先把 `MovesGener` 和 `_cards2array` 读通，比抠 `free_queue` 有用。

---

## 4. 一局里数据怎么走

```text
GameEnv（规则、合法出牌）
    ↓ DummyAgent 隔离「谁在想」和「牌桌」
Env.reset / step          ← 类 Gym
    ↓ get_obs(infoset)    ← 部分可观测 + 每个合法动作编成一行
Actor: Q(s,a) 对合法动作打分，ε-greedy
    ↓ 终局才有 r
每个 (s,a) 的 target = 同一局的最终回报 G   （γ=1）
Learner: MSE(Q, G)，三位置三套网、三个 RMSprop
```

`DummyAgent` 不是算法：外面先 `set_action`，引擎再 `act()`，好让训练长得像 `env.step(action)`。评估走另一条路：`GameEnv` 直接挂真 Agent，见 §8。

`InfoSet` 里其实有**完美信息**（`all_handcards` 三家手牌都在），但 `get_obs()` **不用**各家分开的手牌，只用「另外两家手牌的并集」。这是部分可观测；PerfectDou 才会在训练时把分开的手牌喂给 V。当前 DMC 没用这个钩子。

---

## 5. 规则引擎

### 5.1 牌的表示

整数，无花色：

| 值 | 牌 |
|----|----|
| `3`–`14` | 3–A |
| `17` | 2 |
| `20` / `30` | 小王 / 大王 |

一副 54 张。出牌顺序：`landlord → landlord_down → landlord_up`。

### 5.2 合法动作

`GameEnv.get_legal_card_play_actions()`：

1. `MovesGener` 按手牌枚举能出的牌型（单、对、三、顺、飞机、炸弹……）。
2. `move_selector` 按「桌上要压的那手」过滤同型更大者。
3. 补炸弹、王炸和 `[]`（pass）。

桌上为空时，不能 pass，必须出一手。只有一手合法时（常见是只能 pass），`DeepAgent` 不跑网络。

### 5.3 终局奖励（地主视角）

`Env._get_reward()`，只在 `done` 调用：

| `--objective` | 地主赢 | 农民赢 |
|---------------|--------|--------|
| `wp` | +1 | −1 |
| `adp`（默认） | `+2^炸弹数` | `−2^炸弹数` |
| `logadp` | `炸弹数+1` | 相反 |

Actor 回填时农民拿**地主回报的相反数**：

```python
episode_return = env_output['episode_return'] if p == 'landlord' else -env_output['episode_return']
target_buf[p].extend([episode_return for _ in range(diff)])
```

两个农民共用同一个零和目标，没有单独的「队友奖励」。一局里该位置每一步的 target 都是**同一个**终局 G。中间 `r=0` 且 γ=1，这就是整轨 Monte Carlo。

---

## 6. 特征：动作是输入，不是分类头

### 6.1 `_cards2array`：54 维

一手牌（或手牌、已出牌）编成 54 维：

- `4×13` 点数矩阵，列是 3–2，行是**unary（热度计）编码**，不是 one-hot：0～4 张 → `[0,0,0,0]` … `[1,1,1,1]`（前 *n* 位置 1）。张数越多，1 越多，数值上单调。
- 按列展开（Fortran order），再拼大小王各 1 维。
- **丢掉花色**，只留张数。

动作、手牌、已出牌都走这套。论文 Figure 2。

### 6.2 `get_obs`：每个合法动作一行

对**每个合法动作复制一行状态**，只改最后 54 维的「这一手」：

| 张量 | 形状直觉 | 用途 |
|------|----------|------|
| `x_batch` | `[合法数, 状态+动作]` | 决策：一次前向出一串 Q |
| `z_batch` | `[合法数, 5, 162]` | 最近 15 手，每 3 手拼成 162 维，进 LSTM |
| `x_no_action` | `[状态维]` | 训练 buffer 里不存动作，更新时再拼 |
| `z` | `[5, 162]` | 同上，非 batch |

15 手不够则前面补空。三手一轮，所以 `15 / 3 = 5` 个时间步。

决策时用的是展开后的 `x_batch`（合法集每一手一行）。训练 buffer **不存**这整张表，只存该步的 `x_no_action` 和 Actor **实际打出的**那一手 `obs_action`（54 维）。`learn()` 里再 `cat` 成一行完整 `x`。两套形状不要混：一边是「对所有合法动作打分」，一边是「只回归选中的 `(s,a)`」。

### 6.3 地主 vs 农民维度

| 位置 | `x_no_action` | + 动作 54 | 网络第一层 |
|------|---------------|-----------|------------|
| 地主 | 319 | 373 | `LandlordLstmModel`：`Linear(373+128, 512)` |
| 农民 | 430 | 484 | `FarmerLstmModel`：`Linear(484+128, 512)` |

地主 319 ≈ 手牌 54 + 其余两家并集 54 + 上一手 54 + 上家已出 54 + 下家已出 54 + 上家剩余 17 + 下家剩余 17 + 炸弹 one-hot 15。

农民多出来的是队友通道：地主最近一手、队友最近一手、地主/队友剩余张数。地主没有队友这个角色，只看两个农民的已出牌和剩余张数——不是「缺了一块本该有的信息」，是位置特征不对称。

### 6.4 决策

`evaluation/deep_agent.py`：

```python
obs = get_obs(infoset)
y_pred = model.forward(z_batch, x_batch, return_value=True)['values']  # [合法数, 1]
best_action = infoset.legal_actions[argmax(y_pred)]
```

这是 **Q(s,a) 标量 + 动作编码**，不是 27472 维 softmax。合法集变了，只是 batch 行数变了，网络结构不用改。

训练时 `models.py` 的 `forward`：`return_value=True` 出 Q；否则在合法下标上 ε-greedy（默认 `exp_epsilon=0.01`）或 argmax。探索是对**当前合法集**均匀乱选，不是对两万全集。

---

## 7. DMC 训练

### 7.1 算法核（很短）

`utils.py` 的 `act()`：一局结束，对该位置本局每一步写入同一个 G。

`dmc.py` 的损失：

```python
def compute_loss(logits, targets):
    return ((logits.squeeze(-1) - targets) ** 2).mean()
```

`learn()`：把 `obs_x_no_action` 和 `obs_action` 拼回完整 `x`，MSE，梯度裁剪，立刻把该位置权重拷回所有 Actor。

没有 target 网、没有 bootstrap、没有 advantage、没有 clip。优化器是三个 RMSprop（`lr=1e-4`），三位置分开更新。梯度裁剪 `max_grad_norm=40`（本仓库 CartPole PPO 是 `0.5`）：ADP 下 G 随炸弹数指数涨，范数上限要比 ±1 奖励的任务大。

`Model.forward` 的形参叫 `training`，传给子模型时对上的是 `return_value`（命名错位，不影响运行）。`learn()` 直接调子模型，不经过这个包装。

### 7.2 工程（很长）

TorchBeast 风格：

| 件 | 默认 | 作用 |
|----|------|------|
| Actor 进程 | 每设备 `num_actors=5` | 自博弈，写共享 buffer |
| buffer 段数 | `num_buffers=50` | 每段 `unroll_length=100` 步 |
| Learner 线程 | `num_threads=4` × 三位置 | 从 queue 取 batch，`batch_size=32` |
| `total_frames` | `1e11` | 等于一直跑到你停 |
| 存盘 | 每 30 分钟 | `douzero_checkpoints/` |

Actor 网 `share_memory()` + `eval()`；Learner 一份可训练副本。Windows 上 GPU tensor 不能 multiprocessing，官方写法：

```bash
python train.py --actor_device_cpu --training_device cpu
```

本机这样训，曲线没有对照意义；完整训练按天计。

### 7.3 为什么 MC 而不是 TD

斗地主中间几乎没有奖励，TD 的 `r + γ max Q'` 里 `max` 又要在巨大、变长的合法集上取，过估计很重。整局 G 方差大，但目标无偏，且不依赖「下一步 max Q」。用并行 Actor 堆样本量来压方差——这是 DouZero 能打过当时 Bot 的工程答案，不是更精巧的算法。

---

## 8. 评估和训练不是同一条路

| | 训练 | 评估 |
|--|------|------|
| 环境 | `Env` + DummyAgent | `GameEnv` + 真 Agent |
| 发牌 | 每局随机 | `generate_eval_data.py` 先生成固定 `.pkl` |
| 决策 | 带 ε 的 `Model.forward` | `DeepAgent` 纯贪心 |

```bash
python generate_eval_data.py --num_games 1000
python evaluate.py --landlord baselines/douzero_ADP/landlord.ckpt --landlord_up random --landlord_down random
```

`baselines/` 里没有权重，需按 `DouZero/README.md` 的 Evaluation 另下（Google Drive / 百度网盘）。

评估打印的 WP / ADP 是「地主这边 vs 农民那边」的合计，**不是**论文里「同一副牌换座位打两局」的完整协议。换座要对打，得改 `evaluation/simulation.py`。

基线：`random`、`rlcard`（规则）、SL（人类数据监督，权重另下）、DouZero-ADP / DouZero-WP。

---

## 9. 缺什么、自己重写时不要抄什么

这份克隆缺的：预训练权重、叫牌、论文里的 SL 人类数据、Demo GUI（在 [rlcard-showdown](https://github.com/datamllab/rlcard-showdown)）。

| 可借 | 不要当第一课脚手架 |
|------|-------------------|
| 牌编成点数张数（54 或先 15 维） | 多进程 Actor、共享 CUDA buffer |
| 先枚举合法动作，再给网络打分 | 三位置三套网从第一天自博弈 |
| 终局才给分 | 从 `train.py` 往下抄并行 |
| `InfoSet.all_handcards` 给集中式 Critic | 把规则写进 PPO 文件 |

若准备用新方式实现：规则层独立（三人轮转 + 完整合法动作接口），算法用「候选动作打分 + 合法集 softmax + PPO/GAE」，Critic 训练时可以看三家牌、执行时不能。不要复刻 `MSE(Q, G)` 这一条当主线——那是 DouZero 的选择，不是斗地主的唯一解。

由易到难对照这份代码：

1. 单测规则：随机一手牌，打印 `legal_actions`，对一下会不会漏顺子/飞机。
2. 装上预训练权重，看 `DeepAgent` 对一局的 Q 排序。
3. 自己写时只留编码和合法集接口，训练循环用本仓库已有的 PPO，不要抄 `dmc/`。
