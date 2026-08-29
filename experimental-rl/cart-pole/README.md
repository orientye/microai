# CartPole 示例

总览：[`../README.md`](../README.md)。

经典控制课：小车上立杆，离散左右推。

https://gymnasium.farama.org/environments/classic_control/cart_pole/

| 子目录 / 文件 | 内容 |
|---------------|------|
| [`rl001.py`](rl001.py) | 随机策略基线（无学习；先看这个） |
| [`cart-pole-dqn`](cart-pole-dqn/) | **Double DQN**（值函数 + ε-greedy） |
| [`cart-pole-ppo`](cart-pole-ppo/) | **离散 PPO**（直接学 π） |
| [`cart-pole-deep-dive.adoc`](cart-pole-deep-dive.adoc) | DQN / PPO 公式与逐行对照 |

```bash
python rl001.py

cd cart-pole-dqn
python cartpole_train.py
python cartpole_test.py

cd ../cart-pole-ppo
python ppo_train.py
python ppo_test.py
```

- 随机基线：不学习时能撑几步  
- DQN：连续 4 维状态，经验回放 + 目标网络  
- PPO：同一环境、同一通关阈值，换成 on-policy clip  

后续对照：离散格子 [`../grid-world/`](../grid-world/)；更难的离散 PPO [`../lunar-lander/lunar-lander-ppo/`](../lunar-lander/lunar-lander-ppo/)；连续动作 [`../pendulum/`](../pendulum/)。
