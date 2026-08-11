# Pendulum + 连续动作 PPO

Gymnasium **`Pendulum-v1`**：给倒立摆施加**连续力矩**，把它摆到竖直向上并稳住。  
相对 CartPole / CliffWalking，这里的动作不再是离散 `argmax`，而是高斯策略采样。

| 文件 | 作用 |
|------|------|
| `ppo_train.py` | 连续 Actor-Critic + GAE + PPO clip；存最佳权重 |
| `ppo_test.py` | 用均值动作确定性评估 + `human` 渲染 |

```bash
cd experimental-rl/pendulum/pendulum-ppo
python ppo_train.py
python ppo_test.py
```

依赖：`gymnasium`、`torch`、`numpy`、`matplotlib`；动画需 `pygame`。

---

## 1. 环境在学什么

| 概念 | 取值 |
|------|------|
| 观察 | `[cos θ, sin θ, θ̇]`，3 维连续 |
| 动作 | 力矩 ∈ `[-2, 2]`，**1 维连续** |
| 奖励 | `-(θ² + 0.1·θ̇² + 0.001·力矩²)`，正立静止时接近 `0` |
| 截断 | 默认 200 步 |

目标不是「坚持不倒」的整数步数，而是把回报从很负（乱扭）拉到 **-200 左右或更好**。

---

## 2. 和离散 PPO（CartPole）差在哪

| | CartPole PPO | 本目录 |
|--|--------------|--------|
| 策略分布 | `Categorical(logits)` | `Normal(μ(s), σ)` |
| 选动作 | `sample` / 测试 `argmax` | `sample` / 测试用 **μ** |
| 动作边界 | 离散 id | `tanh` 把均值压进 `[-2,2]`，采样后再 `clamp` |
| `log_prob` | 单维 categorical | 对动作维 `sum` |

其余骨架对齐：GAE(λ)、多 epoch + minibatch、ratio clip、value MSE。  
训练时把环境奖励乘以 `REWARD_SCALE=0.1` 再进 GAE（日志 / 评估仍用原始回报），避免价值损失淹没策略梯度。

---

## 3. 验收

- 训练日志里 `eval_mean` 逐步升高（绝对值变小）  
- 达到约 **≥ -200** 会打印 threshold；**历史最佳**写入 `ppo_pendulum.pth`（后期若回报回落，仍以最佳权重为准）  
- `ppo_test.py` 窗口里摆应能被拧起来并大致稳住（允许抖动）

---

## 4. 非目标

- 不做观测归一化、多环境并行、学习率退火  
- 不做 `MountainCarContinuous`（可作下一课）  

连续控制下一课：[`../pendulum-sac/`](../pendulum-sac/)（SAC）。 
