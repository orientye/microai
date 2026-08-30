# experimental-rl 总览

单智能体、教学向的 Gymnasium 例子：每个子目录一份可跑的 `*_train.py` / `*_test.py`，对照「同一环境换算法」或「同一算法换环境」。

## 1. 推荐顺序

先把 MDP 和 TD 看清，再上网络；先离散后连续；连续里先稠密奖励，再稀疏。

```text
1  表格
   grid-world/grid-world-qlearning      固定小图，Q-learning，策略可画成箭头
   cliff-walking/cliff-walking-q-sarsa  同一悬崖，Q-learning vs SARSA

2  值函数近似
   cliff-walking/cliff-walking-dqn      同一悬崖，Q 换成 one-hot MLP
   cart-pole/cart-pole-dqn              连续状态 + Double DQN（可先跑 rl001.py 随机基线）

3  离散策略
   cart-pole/cart-pole-ppo              离散 Actor-Critic + GAE + clip
   grid-world/grid-world-dqn            随机地图 + CNN；看见墙才能换图
   grid-world/grid-world-ppo            同一随机环境，学 π 而不是 Q
   lunar-lander/lunar-lander-ppo        更难的离散 PPO（Box2D）

4  连续控制
   pendulum/pendulum-ppo                高斯策略（先看这个）
   pendulum/pendulum-sac                同一摆，off-policy + 熵
   mountain-car-continuous/...-sac      稀疏成功 + 欺骗性「少出力」
   lunar-lander/lunar-lander-continuous-sac  同一着陆，2 维油门
```

公式与 CartPole 上 DQN / PPO 的逐行对照：[`cart-pole/cart-pole-deep-dive.adoc`](cart-pole/cart-pole-deep-dive.adoc)。

## 2. 目录

| 目录 | 环境 | 例子 |
|------|------|------|
| [`grid-world/`](grid-world/) | 自定义 5×5 格子 | [Q-learning](grid-world/grid-world-qlearning/) · [CNN DQN](grid-world/grid-world-dqn/) · [CNN PPO](grid-world/grid-world-ppo/) |
| [`cliff-walking/`](cliff-walking/) | `CliffWalking-v1` | [Q vs SARSA](cliff-walking/cliff-walking-q-sarsa/) · [one-hot DQN](cliff-walking/cliff-walking-dqn/) |
| [`cart-pole/`](cart-pole/) | `CartPole-v1` | [随机基线](cart-pole/rl001.py) · [DQN](cart-pole/cart-pole-dqn/) · [PPO](cart-pole/cart-pole-ppo/) |
| [`lunar-lander/`](lunar-lander/) | `LunarLander-v3` | [离散 PPO](lunar-lander/lunar-lander-ppo/) · [连续 SAC](lunar-lander/lunar-lander-continuous-sac/) |
| [`pendulum/`](pendulum/) | `Pendulum-v1` | [连续 PPO](pendulum/pendulum-ppo/) · [SAC](pendulum/pendulum-sac/) |
| [`mountain-car-continuous/`](mountain-car-continuous/) | `MountainCarContinuous-v0` | [SAC](mountain-car-continuous/mountain-car-continuous-sac/) |
| [`doudizhu/`](doudizhu/) | 斗地主（非 Gym ） | [DouZero](doudizhu/DouZero/)（第三方，DMC + 自博弈） |

各课在自己的 README 里写训练命令、超参和验收阈值。公共依赖：`gymnasium`、`numpy`、`matplotlib`；网络课还要 `torch`；`human` 渲染要 `pygame`。LunarLander 额外：`pip install "gymnasium[box2d]"`。

## 3. 概念对照

| | 表格 | DQN | PPO | SAC |
|--|------|-----|-----|-----|
| **离散** | GridWorld、CliffWalking | CartPole、CliffWalking、GridWorld CNN | CartPole、LunarLander、GridWorld CNN | — |
| **连续** | — | — | Pendulum | Pendulum、MountainCar、LunarLander |

| 想看清… | 去哪 |
|----------|------|
| `max Q` vs 真实下一步 `Q` | CliffWalking Q vs SARSA |
| 值函数 vs 直接学 π | CartPole DQN ↔ PPO；GridWorld CNN DQN ↔ PPO |
| 离散 Categorical vs 连续高斯 | CartPole PPO → Pendulum PPO |
| on-policy rollout vs off-policy replay | Pendulum PPO ↔ SAC |
| 稠密奖励 vs 稀疏 / 欺骗奖励 | Pendulum SAC → MountainCar SAC |
| 同一任务、两套动作 | LunarLander 离散 PPO ↔ 连续 SAC |
| 换墙仍要会走（泛化） | GridWorld 随机布局 + 多种子评估 |
