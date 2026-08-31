# 合法集 PPO（B 方案第 2 步）

只训**地主**；两个农民固定为 **random**。奖励 **WP ±1**。  
动作用 DouZero 的「候选打分 + 合法集 softmax」，训练循环是本仓库 PPO（GAE / clip / 熵）。

规则层：[`../doudizhu-env/`](../doudizhu-env/)。评估尺：[`../eval-ruler/`](../eval-ruler/)。本步**不做**完美信息 Critic、三位置自博弈、ADP。

## 和 CartPole PPO / DouZero DMC

| | CartPole PPO | DouZero DMC | 本目录 |
|--|--------------|-------------|--------|
| 策略 | 固定维 Categorical | 不学 π，ε-greedy Q | **合法集 softmax** |
| 目标 | GAE + clip | 整局 G 的 MSE | GAE（γ=1）+ clip |
| 农民 | — | 三套网自博弈 | **随机，不更新** |
| Buffer | `(s,a)` | 只存选中 `(s,a)` | **存全部合法 `x_batch`** |

农民赢时终局发生在农民步上：把 ±1 记到**地主最后一步**（`apply_opponent_terminal`）。

## 命令

```bash
cd experimental-rl/doudizhu/doudizhu-ppo
python test_ppo.py
python ppo_train.py
python ppo_test.py
```

验收：贪心地主对随机农民，评估 WP 约 **≥ 0.42**（高于「瞎出」的地主胜率，不是对打 DouZero）。对打官方权重是第 5 步。
