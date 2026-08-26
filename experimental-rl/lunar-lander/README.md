# LunarLander 示例

Box2D 着陆课：同一降落任务，离散 / 连续两套动作。

https://gymnasium.farama.org/environments/box2d/lunar_lander/

| 子目录 | 内容 |
|--------|------|
| [`lunar-lander-ppo`](lunar-lander-ppo/) | **离散 PPO**（4 个开关；先看这个） |
| [`lunar-lander-continuous-sac`](lunar-lander-continuous-sac/) | **连续 SAC**（2 维油门） |

```bash
cd lunar-lander-ppo
python ppo_train.py
python ppo_test.py

cd ../lunar-lander-continuous-sac
python sac_train.py
python sac_test.py
```

前置对照：

- 离散 PPO：[`../cart-pole/cart-pole-ppo/`](../cart-pole/cart-pole-ppo/)  
- 连续 SAC：[`../pendulum/pendulum-sac/`](../pendulum/pendulum-sac/)、[`../mountain-car-continuous/`](../mountain-car-continuous/)  

依赖额外需要 Box2D：`pip install "gymnasium[box2d]"`（Windows 上可用预编译 `box2d` wheel）。
