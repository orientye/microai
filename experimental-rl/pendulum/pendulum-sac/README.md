# Pendulum + Soft Actor-Critic (SAC)

Gymnasium **`Pendulum-v1`**：连续力矩控制。相对同目录的 PPO，这里是 **off-policy + 熵正则** 的连续控制路线。

| 文件 | 作用 |
|------|------|
| `sac_train.py` | 双 Q + Target 软更新 + Replay + tanh 重参数化 + 自动 α |
| `sac_test.py` | 确定性均值动作评估 + `human` 渲染 |

```bash
cd experimental-rl/pendulum/pendulum-sac
python sac_train.py
python sac_test.py
```

依赖：`gymnasium`、`torch`、`numpy`、`matplotlib`；动画需 `pygame`。

---

## 1. 环境（与 PPO 课相同）

| 概念 | 取值 |
|------|------|
| 观察 | `[cos θ, sin θ, θ̇]`，3 维连续 |
| 动作 | 力矩 ∈ `[-2, 2]`，**1 维连续** |
| 奖励 | `-(θ² + 0.1·θ̇² + 0.001·力矩²)` |
| 截断 | 默认 200 步 |

验收：确定性评估回报约 **≥ -200**（越好越接近 0）。

---

## 2. SAC 在学什么

目标：最大化回报的同时保持策略熵（别学死）：

\[
\max_\pi \; \mathbb{E}\big[R\big] + \alpha \, \mathcal{H}(\pi)
\]

| 组件 | 作用 |
|------|------|
| Actor | 高斯 → `tanh` 压到 `[-2,2]`；重参数化采样 |
| 双 Q | \(Q_1, Q_2\)；target 取 **min**，减轻高估 |
| Target 网 | `θ ← (1-τ)θ + τθ'` 软更新 |
| Replay | 经验反复用，样本效率通常高于 PPO |
| 自动 α | 学 `log α`，把熵推向 `-\|A\|` |

Bootstrap：`terminated` 才当真正结束；`truncated`（到 200 步）不当 terminal。

---

## 3. 和连续 PPO 差在哪

| | `pendulum-ppo` | 本目录 |
|--|----------------|--------|
| 数据 | on-policy rollout，用完丢 | off-policy replay |
| Critic | V(s) + GAE | 双 Q(s,a) |
| 探索 | 采样 +（可选）熵系数 | **熵进目标** + 自动温度 |
| 动作边界 | mean 上 `tanh`，采样后 clamp | `tanh` 变换 + **log_prob 修正** |

---

## 4. 验收

- 日志里 `train_avg20` / `eval_mean` 绝对值变小  
- 达到约 **≥ -200** 会打印 threshold；最佳 Actor 写入 `sac_pendulum.pth`  
- `sac_test.py` 窗口里摆应能拧起并大致稳住  

---

## 5. 非目标

- 不做观测归一化、多环境并行、优先经验回放  
- 不做 TD3 / DDPG（确定性策略另一条线）  
- 不做与 PPO 的严格同 seed 公平对比脚本  
