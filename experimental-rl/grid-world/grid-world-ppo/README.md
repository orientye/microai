# GridWorld + CNN PPO 示例解析

本目录用 **CNN Actor-Critic + PPO** 训练 **每局障碍都可能变化** 的 5×5 GridWorld。Agent 必须**看见整张地图**（墙在哪、终点在哪、自己走过哪）才能换图泛化。

**MDP** 仍是格子世界怎么转、怎么给分；与 DQN 的差别在 **学什么、怎么更新**：这里不学 `Q(s,a)`，而学 **策略 `π(a|s)`**（各动作概率）和 **状态价值 `V(s)`**（这盘图大概还能拿多少分），用 **on-policy rollout + GAE + PPO clip** 更新，而不是经验回放 + Bellman TD。

**本例需要深度学习。** 依赖 `torch`；建议顺序：[`../grid-world-qlearning/`](../grid-world-qlearning/) → [`../grid-world-dqn/`](../grid-world-dqn/) → 本目录。

| 文件 | 作用 |
|------|------|
| `grid_world_env.py` | 随机可解布局；观察为 **4 通道** 拼成的向量（与 DQN 一致） |
| `ppo_train.py` | CNN Actor-Critic + GAE + clip；课程学习 + 塑形 + 动作掩码 |
| `ppo_test.py` | **随机多种子**评估；可选对照表格 Q |
| `inspect_fails.py` | 打印失败局的路径（排查卡死 / 绕圈） |
| `ppo_random_layout.pth` | 最佳策略网络权重 |
| `ppo_reward_history.png` | 训练曲线 |

```bash
cd experimental-rl/grid-world/grid-world-ppo
python ppo_train.py
python ppo_test.py               # 最终成绩：随机 5 种子 × 每种子 200 张图
python inspect_fails.py          # 排查失败路径（可自改 SEED）
```

依赖：`gymnasium`、`numpy`、`matplotlib`、`torch`。

环境与观察细节与 [`../grid-world-dqn/`](../grid-world-dqn/) 共用同一套 `grid_world_env.py` 设计；值函数 / 策略梯度见 [`../../cart-pole/cart-pole-ppo/`](../../cart-pole/cart-pole-ppo/)。

---

## 1. 问题：GridWorld 在学什么

Agent 从「能到 G 的随机起点」走到右下角终点 `G`，中间有 **3 个随机障碍 `#`**（训练时课程会从更少障碍起步）。目标仍是 **尽快到终点**。

与 [`../grid-world-qlearning/`](../grid-world-qlearning/) 不同：**每局 `reset()` 重新采样障碍**（BFS 保证有路），不能只背格子 id。与 [`../grid-world-dqn/`](../grid-world-dqn/) 相同：**状态是整张 4 通道图**；差别是输出与算法（见 §3、§7）。

| 概念 | 本例取值 |
|------|----------|
| 状态 `s` | **4 通道 5×5 网格**（flatten 成长度 100 的向量） |
| 动作 `a` | `0`↑ `1`→ `2`↓ `3`← |
| 奖励 `r` | 见下表（环境原始回报；训练时还会加距离塑形） |
| `terminated` | 踩到 `G` |
| `truncated` | 超过 **50** 步仍未到达 |
| 撞边界 / 撞障碍 | **位置不变**，`bump_penalty` |
| 重访格子 | 额外 `revisit_penalty × 已访问次数` |

奖励（环境 `step()` 的 **base reward**，不含塑形）：

| 情况 | 奖励 |
|------|------|
| 到终点 | `+1.0` |
| 普通一步（成功移动） | `-0.01` |
| 撞墙 / 出界 | `-0.05` |
| 重访格子 | 额外 `-0.03 × prev_visits` |

观察（`4 × 5 × 5`，网络内部 reshape 成卷积输入）：

| 通道 | 含义 |
|------|------|
| `agent` | 当前位置为 `1` |
| `obstacle` | 障碍格为 `1` |
| `goal` | 终点格为 `1` |
| `visited` | 本局访问次数 ÷ 5，clip 到 `[0,1]` |

PPO 在这里 **直接学怎么走**，而不是先学 Q 再 argmax：

| | 表格 Q-learning | CNN-DQN | **本例 CNN-PPO** |
|--|----------------|---------|------------------|
| 输出 | `Q[25,4]` | `QNet → 4 个 Q 值` | **Actor logits + Critic `V(s)`** |
| 选动作（训练） | ε-greedy on Q | ε-greedy on Q | **按 π 采样**（合法动作掩码后 Categorical） |
| 选动作（测试） | argmax Q | argmax Q（+掩码） | **argmax logits**（+掩码 + `avoid_revisit`） |
| 数据 | 每步改表 | off-policy replay | **on-policy rollout** |
| 探索 | ε | ε | **策略熵**（+ 随机采样） |

---

## 2. 环境怎么建模（`grid_world_env.py`）

与 DQN 目录 **同构**（随机布局、4 通道、合法动作、重访惩罚）。若已读过 [`../grid-world-dqn/README.md`](../grid-world-dqn/README.md) 第 2 节，可跳过；这里只保留要点。

### 2.1 随机布局与可达性

```text
随机抽 k 个障碍 → BFS 保证 (0,0) 与 (4,4) 连通
→ 从 G 反搜「能到 G」的格子 → random_start 只从这里采样
```

### 2.2 观察向量

```text
obs = [ agent | obstacle | goal | visited ]  → shape (100,)
网络: view(-1, 4, 5, 5)
```

### 2.3 合法动作与 `avoid_revisit`

- `legal_actions()`：只含不会立刻撞墙/出界的动作。  
- `legal_actions(avoid_revisit=True)`：优先走向 **访问更少** 的邻格（评估 / 测试防 A↔B 绕圈）。

训练 rollout：`legal_actions()` + **随机采样**。  
评估 / 测试：`greedy=True`（logits argmax）+ `avoid_revisit=True`。

---

## 3. 算法：CNN Actor-Critic + PPO

### 3.1 策略与价值是什么

- **`π(a|s)`（Actor）**：看见状态 `s`（整图）后，四个方向各自有多大概率。本实现输出 **logits**，对 **掩码后的合法动作** 做 `Categorical` 分布。  
- **`V(s)`（Critic）**：「从这盘图的状态出发，按当前策略玩下去，大概能拿多少折扣回报」。供 **GAE** 算 advantage，不直接用来选动作。

网络结构（`ppo_train.py` 里 `ActorCritic`）：

```text
输入 (4, 5, 5)
  → Conv 4→32 → Conv 32→64 → Flatten
  → Linear → 128 (shared)
  → actor head  → 4 logits
  → critic head → 1 标量 V(s)
```

与 DQN 的 `QNet` 相比：卷积骨干类似，但 **一条 shared 特征分两头**——一头定概率，一头定「这局好坏」。

### 3.2 On-policy：一次 update 在干什么

PPO 的训练单位是 **update**，不是 DQN 的「每个 env step 都可能 `train_step`」：

```text
1) collect_rollout：用「当前策略」连续交互 ROLLOUT_STEPS=2048 步
   每步存 (s, a, log π_old(a|s), r_塑形, done, V(s), legal)
   episode 中途结束就 reset，继续凑满 2048 步

2) compute_gae：用 rollout 内 V 和 r，算 advantage A_t 与 return 目标

3) ppo_update：对同一批数据重复 PPO_EPOCHS=4 轮
   每轮打乱后按 MINI_BATCH=256 切 mini-batch
   clip 策略损失 + MSE 价值损失 − 熵奖励
```

**On-policy 含义**：用来更新的 `(s, a, log_prob)` 必须是 **采集时那个策略** 产生的；更新完这批数据就丢弃，下一 update 再重新 rollout。不能 like DQN 那样把半年前的 transition 混进 batch（除非做 importance sampling，本例未做）。

### 3.3 GAE 优势（Generalized Advantage Estimation）

对每个 rollout，从后往前扫（`compute_gae`）：

```text
δ_t = r_t + γ · V_{t+1} · (1 − done_t) − V_t
A_t = δ_t + γ · λ · (1 − done_t) · A_{t+1}        λ = GAE_LAMBDA = 0.95
returns_t = A_t + V_t                              → Critic 回归目标
```

- `r_t` 是 **塑形后的** `train_r`（与 DQN 一样，只进学习信号，不改变 env 真实分）。  
- rollout **最后一步**之后：用当前状态的 `V(s_last)` bootstrap；若最后一条 transition 已 `done`，则 `last_value = 0`。  
- batch 内对 `advantages` 做 **零均值、单位方差** 归一化，再进 PPO loss。

直觉：`A_t > 0` 表示「这一步比 Critic 预期的好」，增大该动作概率；`A_t < 0` 则减小。

### 3.4 PPO clip 损失

Importance ratio（新策略相对采集时旧策略）：

```text
ratio = exp( log π_new(a|s) − log π_old(a|s) )
L_clip = − min( ratio·A,  clip(ratio, 1−ε, 1+ε)·A )     ε = CLIP_EPS = 0.2
L_value = MSE( V(s), returns )
L = L_clip + VALUE_COEF·L_value − ENTROPY_COEF·entropy(π)
```

对应代码要点：

```python
ratio = torch.exp(new_lp - mb_old_lp)
surr1 = ratio * mb_adv
surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
policy_loss = -torch.min(surr1, surr2).mean()
value_loss = F.mse_loss(values_pred, mb_ret)
loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
```

- **Clip**：限制 `ratio` 偏离 1 太远，避免一次 update 把策略改崩。  
- **Entropy**：`ENTROPY_COEF=0.01`，鼓励合法动作上保留一点随机性（替代 DQN 的 ε-greedy）。  
- **动作掩码**：rollout 时每个 `Transition` 存当时 `legal`；更新时对 **每条样本** 重新 mask logits，再算 `log_prob` 与 entropy（合法集随状态变）。

### 3.5 距离塑形

与 DQN 相同，只作用于 **写入 rollout 的 reward**（日志里的 episode return 仍是 base reward 之和）：

```text
Φ = −manhattan / max_dist,   max_dist = 2×(size−1) = 8
train_r = base_r + 0.1 × (γ·Φ(s') − Φ(s))
```

### 3.6 课程学习（按 update，不是按 episode）

| Update 范围 | 每局障碍数 `n_obstacles` |
|-------------|--------------------------|
| `1 … 80` | `1` |
| `81 … 180` | `2` |
| `181 … 300` | `3` |

评估始终 **3 障碍 + 随机起点**（`EVAL_SEED` 固定 300 张图），与 DQN 训练评估一致。

---

## 4. 训练循环在干什么（`ppo_train.py`）

```text
初始化 ActorCritic + Adam(lr=3e-4)
for update = 1 .. MAX_UPDATES(300):
    n_obs = curriculum_obstacles(update)
    batch, last_value, ep_returns = collect_rollout(2048 步, n_obs)
    losses = ppo_update(batch, last_value)   # 4 epoch × 若干 mini-batch
    累积 ep_returns 用于曲线
    每 10 次 update：
        固定 EVAL_SEED 上 300 张 3-障碍图，greedy + avoid_revisit
        score = success_rate + 0.001×mean_return → 更优则存 ppo_random_layout.pth
保存 ppo_reward_history.png
```

### 4.1 单局何时结束

| 条件 | 结果 |
|------|------|
| 踩到 `G` | `terminated=True` |
| **50** 步未到 | `truncated=True` |

GAE 里 `done=1` 时不再向未来 bootstrap（与 DQN、CartPole PPO 示例相同的简化）。

### 4.2 整个训练何时停止

| 项 | 值 | 说明 |
|----|-----|------|
| `MAX_UPDATES` | `300` | 共 300 次 PPO update |
| 环境步量级 | 约 `300 × 2048 ≈ 614k` | 含跨 episode 拼接 |
| Early stopping | **无** | 跑满 300 update |

没有 DQN 目录那种「6000 episode + 4000 finetune」；PPO 一次 update 已经吃掉 2048 步。

### 4.3 超参数速查

| 常量 | 值 | 含义 |
|------|-----|------|
| `LR` | `3e-4` | Adam |
| `GAMMA` | `0.99` | 折扣 |
| `GAE_LAMBDA` | `0.95` | GAE λ |
| `CLIP_EPS` | `0.2` | PPO clip |
| `ENTROPY_COEF` | `0.01` | 熵奖励系数 |
| `VALUE_COEF` | `0.5` | 价值损失权重 |
| `GRAD_CLIP` | `0.5` | 全局梯度裁剪 |
| `ROLLOUT_STEPS` | `2048` | 每次 update 采集步数 |
| `MINI_BATCH` | `256` | PPO 内 mini-batch |
| `PPO_EPOCHS` | `4` | 同一 rollout 重复优化轮数 |
| `EVAL_EVERY` | `10` | 每 10 update 评估一次 |
| `EVAL_LAYOUTS` | `300` | 训练内评估图数 |
| `EVAL_SEED` | `12345` | 固定评估流 |
| `SHAPE_COEF` | `0.1` | 距离塑形 |

---

## 5. 如何读训练 / 测试结果

### 5.1 训练日志 vs 最终测试

| | 训练中 `evaluate()` | `ppo_test.py` |
|--|---------------------|---------------|
| 种子 | **固定** `EVAL_SEED=12345` | 每次 **随机 5 种子** |
| 图数 | 300 / 次 | 5 × 200 = **1000** 局 |
| 动作 | greedy + legal + `avoid_revisit` | 同左 |
| 用途 | 存 best、看收敛 | 证明泛化 |

训练里 `success=300/300` 只说明固定 300 张考题全过；`ppo_test.py` 的 `overall success` 与 `per-seed mean±std` 更严。

### 5.2 控制台示例

```text
update= 30  n_obs=1  episodes≈412  rollout_return=-0.180
  pi=0.012  v=0.045  H=0.892
  eval_mean=0.520  success=90% (270/300)  min=-0.450
  saved ppo_random_layout.pth (success=90%, mean=0.520)
```

- `rollout_return`：本 update 刚采样的若干 **完整 episode** 的 base return 均值（探索性，会抖）。  
- `pi` / `v` / `H`：最近一次 `ppo_update` 的策略损失、价值损失、策略熵（标量日志）。  
- `episodes≈`：累计完成的 episode 数（用于曲线横轴近似）。  
- `eval_*`：固定 300 张 **3 障碍** 贪心成绩。

### 5.3 `ppo_reward_history.png`

- 训练回报滑动平均（on-policy 采样，仍可能抖）。  
- 评估点横轴为 **近似 episode 计数**，纵轴为 eval mean return（3 障碍）。

---

## 6. 测试脚本（`ppo_test.py`）

```bash
python ppo_test.py
```

**不再训练**；加载 `ppo_random_layout.pth`，`net.eval()`，全程 **greedy** + 掩码 + `avoid_revisit`。

流程与 [`../grid-world-dqn/dqn_test.py`](../grid-world-dqn/dqn_test.py) 同构：

```text
1) 加载 ActorCritic
2) 随机 5 个 seed
3) seeds[0] 上 5 局 demo + render
4) 每 seed 200 局 → per-seed / overall 汇总
5) 若有 ../grid-world-qlearning/q_table.npy → 同种子表格 Q 对照
```

表格 Q 仍 **不含墙通道**，随机布局上成功率通常明显低于 PPO/DQN。

### 6.1 排查失败（`inspect_fails.py`）

默认 `SEED=42`，`N_LAYOUTS=200`，列出失败局的起点、障碍、路径、访问次数 Top5，并重放终局地图。最多打印前 10 条失败；改文件顶部常量可换种子。

---

## 7. 和 Q-learning / DQN / CartPole PPO 的关系

```text
grid-world-qlearning   grid-world-dqn        grid-world-ppo (本目录)   cart-pole-ppo
────────────────────   ──────────────        ───────────────────────   ─────────────
Q 表                   Q 网络                π + V 网络                π + V 网络
TD + max Q             TD + replay           GAE + clip                同左
ε-greedy               ε-greedy              采样 + 熵                  采样 + 熵
固定地图               随机图 + CNN          随机图 + CNN              连续 4 维状态
```

建议认知顺序：

1. [`../grid-world-qlearning/`](../grid-world-qlearning/)：MDP、TD、Q 表  
2. [`../grid-world-dqn/`](../grid-world-dqn/)：看图、replay、Double DQN  
3. **本例**：同一 GridWorld，换 **策略梯度 + PPO**  
4. [`../../cart-pole/cart-pole-ppo/`](../../cart-pole/cart-pole-ppo/)：PPO 骨架更细（可与本例对照 GAE / clip 公式）

### 7.1 和同级 DQN 怎么比（实践向）

| 维度 | DQN | PPO |
|------|-----|-----|
| 样本效率 | replay 常更省环境步 | 每 update 固定 2048 新步，重复 4 epoch 学同一批 |
| 稳定性 | 需 target / clip 等 | clip + 归一化 advantage，GridWorld 上通常较稳 |
| 探索 | ε 衰减 | 熵 + 随机采样 |
| 离散小动作 | 很合适 | 合适；本例与 DQN 共用环境，可直接比 `success` |
| 扩展 | 离散 Q 为主 | 连续动作、约束策略更自然 |

同一 checkpoint 标准下，两者都应证明 **换种子换墙仍能到 G**；若 DQN 略高或 PPO 略高都正常，取决于训练步数与随机种子。

---

## 8. 常见问题

**Q: PPO 在这里学的是什么，和 DQN 一句话区别？**  
A: DQN 学「每个动作值多少 Q」再 greedy；PPO 学 **直接输出动作分布 π(a|s)** 和 **V(s)**，用优势 `A=r+γV'−V` 告诉 Actor 哪步好、哪步差。

**Q: 为什么没有 replay buffer？**  
A: 标准 PPO 是 **on-policy**；旧策略下的 `(s,a)` 若用新策略算 loss，需要 importance ratio + clip，本实现只保留 **当前 rollout** 内的 `log π_old`。

**Q: 训练还在随机探索吗？**  
A: 是。`select_action(..., greedy=False)` 从 **掩码后的 Categorical** 采样；评估 / 测试才 `greedy=True`（argmax logits）。

**Q: 塑形、visited 通道、avoid_revisit 和 DQN 一样吗？**  
A: **环境相同**；塑形公式与系数相同；测试时同样用 `avoid_revisit` 防绕圈。PPO 额外把 **visited 写进 obs**，Actor 可以自己学，不必只靠测试启发式。

**Q: `update` 和 `episode` 怎么对应？**  
A: 一次 update 固定采 **2048 步**，中间可能结束多个 episode；日志 `episodes≈` 是累计 episode 数，不是 update 数。

**Q: 训练 `success` 和 `ppo_test.py` 差很多？**  
A: 与 DQN 相同原因：训练用 **固定** 300 张图；测试用 **新随机种子** × 1000 局。以测试为准看泛化。

**Q: 只有 300 个 update，够吗？**  
A: 教学默认值；约 60 万环境步。想更高成功率可加大 `MAX_UPDATES` 或 `ROLLOUT_STEPS`（显存与时间会增加）。

---

## 9. 下一步可扩展

- 与 DQN **同种子对照**实验（两目录测试脚本已对齐格式）  
- Ablation：去掉塑形 / 去掉 visited / 不用 `avoid_revisit` 评估  
- Dueling 式 critic、value clip、Separate conv encoders  
- 更大网格或连续动作（PPO 更自然）  
- 共享 `grid_world_env` 的 IMPALA / A2C 等 on-policy 变体
