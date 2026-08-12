# LunarLander 示例

Box2D 着陆课：离散点火降落。

https://gymnasium.farama.org/environments/box2d/lunar_lander/

| 子目录 | 内容 |
|--------|------|
| [`lunar-lander-ppo`](lunar-lander-ppo/) | **离散 PPO**（本课主线） |

```bash
cd lunar-lander-ppo
python ppo_train.py
python ppo_test.py
```

前置对照：

- 离散 PPO：[`../cart-pole/cart-pole-ppo/`](../cart-pole/cart-pole-ppo/)  
- 连续控制：[`../pendulum/`](../pendulum/)、[`../mountain-car-continuous/`](../mountain-car-continuous/)  

依赖额外需要 Box2D：`pip install "gymnasium[box2d]"`（Windows 上可用预编译 `box2d` wheel）。
