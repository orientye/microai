# CartPole + 简化 PPO 示例解析

本目录用 **策略梯度 / 简化 PPO** 训练 Gymnasium 的 `CartPole-v1`，并提供可视化测试脚本。实现为**单文件、教学向**：与「极简草稿」相比，实际代码加入了 **分离 Actor/Critic、GAE、多 epoch 更新、梯度裁剪** 等组件，才在 CartPole 上稳定收敛。

| 文件 | 作用 |
|------|------|
| `ppo_train.py` | 训练：on-policy rollout + GAE + PPO clip 多 epoch 更新 |
| `ppo_test.py` | 加载 `ppo_cartpole.pth`，贪心策略渲染 5 局 |
| `test_ppo_helpers.py` | 辅助函数单元测试（`compute_returns`、`normalize_adv`） |
| `ppo_cartpole.pth` | 训练过程中按近 10 局均分保存的最佳权重 |
| `reward_history.png` | 训练结束后绘制的每局奖励曲线 |

```bash
# 训练（在本目录下运行）
python ppo_train.py

# 测试（需已有 ppo_cartpole.pth）
python ppo_test.py

# 辅助函数测试
python test_ppo_helpers.py
```

依赖：`gymnasium[classic-control]`、`torch`、`numpy`、`matplotlib`。

---

## 1. 问题：CartPole 在学什么

CartPole-v1 是极简 MDP：小车上立一根杆，每步向左或向右推车，目标是尽量不让杆倒下。

| 概念 | 本例取值 |
|------|----------|
| 状态 `s` | 4 维：`[车位置, 车速度, 杆角度, 杆角速度]` |
| 动作 `a` | 2 个：`0` 左推，`1` 右推 |
| 奖励 `r` | 杆仍立着 → 每步 `+1` |
| `terminated` | 杆偏角过大 / 车出界 |
| `truncated` | 撑满 **500** 步（单局满分） |
| 通关判定 | 近 10 局均分 ≥ **450**（仅打印提示，**不早停**） |

与 [`../cart-pole/`](../cart-pole/) 的 Double DQN 示例对照：同一环境、同一通关阈值，但学习范式从 **值函数 + ε-greedy** 换成 **直接学策略 `π(a|s)`**。CartPole 环境细节见 [`../cart-pole/README.md`](../cart-pole/README.md) 第 1 节。

---

## 2. 算法：本例实现的简化 PPO

### 2.1 与「超极简草稿」的差异

最早设计草稿可能是「单网络、Monte Carlo return、一次更新」的 REINFORCE 风格。实际 `ppo_train.py` 为在 CartPole 上稳定训练，采用了更接近工业 PPO 的骨架：

| 草稿设想 | 本例实现 |
|----------|----------|
| 单头网络输出 logits + value | **分离** `Actor` / `Critic`，经 `ActorCritic` 组合 |
| ReLU 隐层 | **Tanh** 64×64 MLP |
| 全轨迹 Monte Carlo return | **GAE(λ=0.95)** 优势 + bootstrap |
| 每批数据更新 1 次 | 同一 rollout **重复 `UPDATE_EPOCHS=4` 次** |
| 仅 policy gradient | **Clip surrogate** + **value MSE** + **entropy bonus** |
| 无梯度裁剪 | **`clip_grad_norm_(..., 0.5)`** |

`compute_returns` 仍保留在代码与测试中（折扣回报、遇 `done` 截断），但**训练主循环用的是 `compute_gae`**，不是 Monte Carlo return。

### 2.2 On-policy 数据流

与 DQN 的 off-policy 回放不同，PPO 每次更新只用**刚采出来的**固定长度 rollout，更新完即丢弃：

```text
环境交互 ROLLOUT_STEPS=128 步
        │
        ▼
  (s, a, log π_old, r, done, V(s))  batch
        │
        ▼
  compute_gae ──► advantages, returns
        │
        ▼
  normalize_adv(advantages)
        │
        ▼
  ppo_update × UPDATE_EPOCHS 次
    ├─ ratio = exp(log π_new - log π_old)
    ├─ policy_loss = -min(ratio·A, clip(ratio)·A)
    ├─ value_loss = MSE(V, returns)
    ├─ entropy bonus
    └─ Adam + grad clip 0.5
```

**On-policy**：128 步内的 `(s, a, log_prob)` 是「旧策略」下的样本；多 epoch 时 ratio 衡量新策略相对旧策略的变化，clip 限制单步更新幅度。

### 2.3 GAE 与 bootstrap

对每个 rollout 末尾，用**当前状态**的 `V(s_last)` 作为 bootstrap：

```text
δ_t = r_t + γ · V_{t+1} · (1 − done_t) − V_t
A_t = δ_t + γλ · (1 − done_t) · A_{t+1}
```

`returns = A + V`，作为 Critic 的回归目标。  
`done = terminated or truncated`：episode 结束时 GAE 不再向未来 bootstrap（与 DQN 侧相同的简化处理）。

### 2.4 PPO 损失（代码对应）

```python
ratio = torch.exp(new_log_probs - old_log_probs)
surrogate = ratio * advantages
clipped_surrogate = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
policy_loss = -torch.min(surrogate, clipped_surrogate).mean()
value_loss = F.mse_loss(values, returns)
loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy
```

- **Clip**：`CLIP_EPS=0.2`，限制 importance ratio，避免策略一步走太远。
- **Value**：纯 MSE，**无 value clip**（与完整 PPO 的差别之一）。
- **Entropy**：鼓励探索，系数 `ENT_COEF=0.01`。

---

## 3. 代码模块拆解

### 3.1 `Actor` / `Critic` / `ActorCritic`

| 模块 | 结构 | 输出 |
|------|------|------|
| `Actor` | `4 → 64 → Tanh → 64 → Tanh → 2` | 动作 logits（未 softmax，由 `Categorical` 处理） |
| `Critic` | `4 → 64 → Tanh → 64 → Tanh → 1` | 标量 `V(s)` |
| `ActorCritic` | 组合上述两者 | `forward` 返回 `(logits, value)`；`act` 采样动作并返回 `(action, log_prob, value)` |

训练时 `act()` 在 `no_grad` 下采样；测试时 `ppo_test.py` 对 logits 取 **`argmax`（贪心）**，不再采样。

### 3.2 `collect_rollout`

- 固定采集 **`ROLLOUT_STEPS=128`** 个环境步（**不是** 128 个完整 episode）。
- 一步内 episode 结束则 `env.reset()`，继续填 batch，直到凑满 128 步。
- 返回 tensor batch + 更新后的 `(state, episode_reward)`，供下一 rollout 衔接。

### 3.3 `compute_gae` / `normalize_adv` / `compute_returns`

| 函数 | 用途 |
|------|------|
| `compute_gae` | **训练用**：GAE 优势 + bootstrapped returns |
| `normalize_adv` | 对 batch 内 advantages 做零均值、单位方差（`unbiased=False`） |
| `compute_returns` | **未在训练主循环调用**；保留供理解与 `test_ppo_helpers.py` 测试 |

### 3.4 `ppo_update`

对同一 rollout 数据循环 **`UPDATE_EPOCHS=4`** 次：前向 → 算 clip policy loss + value MSE + entropy → 反传 → **`clip_grad_norm_(max_norm=0.5)`** → Adam step。返回最后一次的 loss 标量供日志打印。

### 3.5 训练主循环

```text
for update in 1..MAX_UPDATES:
    collect_rollout(128 steps)
    compute_gae + normalize_adv
    ppo_update (4 epochs)
    if len(reward_history) >= 10:
        近 10 局均分刷新 best → 保存 ppo_cartpole.pth
        avg >= 450 → 打印通关提示（仅一次）
        update % 25 == 0 → 打印进度
收尾：若无 best 文件则保存当前权重；绘制 reward_history.png
```

**不早停**：即使均分 ≥450，仍跑满 `MAX_UPDATES=500`。

### 3.6 `ppo_test.py`

- 权重路径相对 **`__file__`** 解析，比 DQN 测试脚本更稳健。
- `torch.load(..., map_location="cpu")`。
- 无法 `render_mode="human"` 时回退无渲染模式。
- `Actor`/`Critic`/`ActorCritic` 与训练文件**复制粘贴**两份，改结构需同步。

### 3.7 `test_ppo_helpers.py`

- `test_compute_returns_stops_at_done`：验证 `done` 处折扣回报截断。
- `test_normalize_adv_zero_mean_unit_std`：验证 advantage 标准化。

---

## 4. 超参数含义

| 参数 | 值 | 直觉 |
|------|-----|------|
| `LR` | `3e-3` | 比 DQN 的 `1e-3` 略高；on-policy 小 batch 上常用稍大学习率 |
| `GAMMA` | `0.99` | 远期回报折扣，与 DQN 一致 |
| `GAE_LAMBDA` | `0.95` | GAE 偏差-方差折中；越接近 1 越接近 Monte Carlo |
| `CLIP_EPS` | `0.2` | PPO clip 半径；ratio 被限制在 `[0.8, 1.2]` |
| `VF_COEF` | `0.5` | value loss 权重 |
| `ENT_COEF` | `0.01` | 熵正则，减缓策略过早塌缩到确定性 |
| `ROLLOUT_STEPS` | `128` | 每次更新前的 on-policy 步数 |
| `UPDATE_EPOCHS` | `4` | 同一 rollout 重复优化次数 |
| `MAX_UPDATES` | `500` | 外层更新轮数（非 episode 数） |
| Grad clip | `0.5` | 全局梯度范数上限，抑制 value/policy 偶发大梯度 |
| 通关阈值 | 450 | 与 DQN 示例一致，近 10 局均分 |

---

## 5. 一次成功训练大致经历的阶段

1. **随机策略期（前若干 update）**  
   初始 Actor 近似均匀随机，Critic 未校准；单局步数低，reward 曲线波动大。

2. **Critic 跟上 + 策略微调**  
   GAE 依赖 `V(s)` 质量；value MSE 下降后，advantage 信号变可靠，clip 内的 policy 更新开始「推杆方向」。

3. **局分爬升**  
   无需 ε-greedy：采样本身带随机性，entropy 进一步维持探索。近 10 局均分从几十 → 一两百 → 400+。

4. **接近满分与存盘**  
   均分 ≥450 打印提示；历史最佳均分持续刷新时覆盖 `ppo_cartpole.pth`。训练仍继续至 500 updates。

5. **收尾**  
   保存 `reward_history.png`；若从未触发 best 保存逻辑，兜底写入当前权重。

---

## 6. 与 Double DQN（`../cart-pole/`）对照

| | Double DQN (`cart-pole`) | 简化 PPO（本目录） |
|--|--------------------------|-------------------|
| 范式 | Off-policy 值函数 | On-policy 策略梯度 |
| 网络 | 单 `QNet`，输出 `Q(s,a)` | 分离 Actor + Critic |
| 动作选择 | ε-greedy on Q | 训练：`π` 采样；测试：argmax |
| 数据 | ReplayBuffer 1 万条 | 每 update 仅最近 128 步 |
| 目标 / 优势 | TD：`r + γ · Q_target(s', a*)` | GAE + returns |
| 稳定手段 | Target net 软更新 + Double Q | PPO clip + 多 epoch + grad clip |
| 探索 | ε 衰减 | 随机采样 + entropy bonus |
| 更新触发 | 每环境步（池满后） | 每 128 步一批，批内 4 epoch |
| 学习率 | `1e-3` | `3e-3` |
| 训练上限 | `MAX_EPISODES=500` | `MAX_UPDATES=500` |
| 存盘 | 近 10 局 best | 近 10 局 best |
| 早停 | 无 | 无 |

CartPole 上两者都能收敛；PPO 更贴近「直接优化期望回报」的叙述，DQN 更贴近「先学 Q 再贪心」的经典 RL 课程顺序。

---

## 7. 当前实现的亮点与已知不足

**亮点**

- 分离 Actor/Critic + Tanh MLP，比单头共享更利于 value 拟合。
- GAE + advantage 标准化，比纯 Monte Carlo return 方差更小。
- PPO clip + 多 epoch，在 on-policy 数据上可重复利用同一 batch。
- 梯度裁剪降低训练尖峰。
- 按历史最佳近 10 局均分存盘；Windows UTF-8 stdout/stderr。
- `ppo_test.py` 权重路径基于 `__file__`；`map_location="cpu"`。

**已知不足 / 非目标（尚未改代码）**

- **无 observation normalization**；CartPole 状态尺度小，尚可接受。
- **无 value clip**（完整 PPO 常对 value 更新也做 clip）。
- **无** Prioritized Replay、**无** 多环境并行向量 env（本例本来就不需要 replay）。
- **无** 学习率衰减、**无** KL 早停 / adaptive KL。
- `done` 把 `truncated` 与 `terminated` 同等对待，截断处不 bootstrap。
- `compute_returns` 与训练路径脱节，易让读者误以为在用 MC return。
- `ActorCritic` 在 train/test 各写一份，结构易漂移。
- 通关后不早停，跑满 `MAX_UPDATES`。
- 未固定随机种子，曲线不易复现。
- `torch.load` 未设 `weights_only=True`（测试脚本）。
- 损失与日志较简：无单独 policy/value/entropy 分项打印。

以上对应**现有代码行为**，非遗漏待办清单。

---

## 8. 可选优化清单（暂未落地）

若以后要改，建议优先级：

1. 通关后早停，或达到目标后降低 `LR`  
2. 抽出公共 `ActorCritic` 供 train/test 共用  
3. 增加 **value clip** 或 Huber value loss  
4. Observation running mean/std（或简单缩放）  
5. 固定 seed；KL 散度监控；分项 loss 日志  
6. `torch.load(..., weights_only=True)`  
7. 向量环境并行采集 rollout（提速，非 CartPole 必需）  
8. 删除或明确标注 `compute_returns` 仅为测试/对照用途  

对 CartPole 而言，GAE、分离网络、clip、多 epoch 已是「能训起来」的最小增量集；再堆 IMPALA、LSTM、PopArt 等属于过度工程。若只需对比算法思想，[`../cart-pole/`](../cart-pole/) 的 DQN 与本目录 PPO 已足够。
