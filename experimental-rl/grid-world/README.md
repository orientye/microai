# GridWorld 示例

本目录包含三个递进例子（表格对照课见同级 [`../cliff-walking/`](../cliff-walking/)）：

| 子目录 | 内容 |
|--------|------|
| [`grid-world-qlearning`](grid-world-qlearning/) | 固定地图 + 表格 Q-learning（先看这个） |
| [`grid-world-dqn`](grid-world-dqn/) | 随机地图 + CNN Double DQN（看见整张图才能换墙） |
| [`grid-world-ppo`](grid-world-ppo/) | 同一随机环境 + CNN Actor-Critic PPO（学策略 π） |

```bash
# 1) 固定地图表格法
cd grid-world-qlearning
python q_learning_train.py
python q_learning_test.py

# 2) 随机布局 DQN
cd ../grid-world-dqn
python dqn_train.py
python dqn_test.py          # 随机多种子评估（默认 5×200）

# 3) 随机布局 PPO（环境与 DQN 同构）
cd ../grid-world-ppo
python ppo_train.py
python ppo_test.py
```

- Q-learning：离散状态查表，策略可画成箭头图  
- DQN：每局换障碍；学 Q，最终用**随机多种子**测泛化  
- PPO：同一观察与奖励；学 π + V，on-policy + clip  
