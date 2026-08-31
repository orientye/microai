# 完美信息 Critic（B 方案第 3 步）

相对 [`../doudizhu-ppo/`](../doudizhu-ppo/)：Actor 仍只看部分观测；**训练时** Critic 额外吃三家手牌（`all_handcards` → 3×54 维）。

执行时 `act(..., deterministic=True)` **不传**完美信息，避免泄漏。

| | 第 2 步 PPO | 本目录 |
|--|-------------|--------|
| Actor | 合法集 softmax，部分观测 | 相同 |
| Critic | `x_no_action`（319） | **319 + 162**（三家手牌） |
| 农民 | random | random |
| 奖励 | WP | WP |

## 命令

```bash
cd experimental-rl/doudizhu/doudizhu-ppo-critic
python test_critic.py
python ppo_train.py
python ppo_test.py
```
