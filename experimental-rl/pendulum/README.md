# Pendulum 示例

连续控制课：倒立摆 + 力矩。

https://gymnasium.farama.org/environments/classic_control/pendulum/

| 子目录 | 内容 |
|--------|------|
| [`pendulum-ppo`](pendulum-ppo/) | **连续动作 PPO**（高斯策略；先看这个） |

```bash
cd pendulum-ppo
python ppo_train.py
python ppo_test.py
```

前置对照：

- 离散 PPO：[`../cart-pole/cart-pole-ppo/`](../cart-pole/cart-pole-ppo/)  
- 离散格子 / 悬崖：[`../grid-world/`](../grid-world/)、[`../cliff-walking/`](../cliff-walking/)  
