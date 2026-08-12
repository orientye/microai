# MountainCarContinuous 示例

连续控制课：谷底小车 + 连续推力，目标是右侧山顶。

https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/

| 子目录 | 内容 |
|--------|------|
| [`mountain-car-continuous-sac`](mountain-car-continuous-sac/) | **SAC**（双 Q + 自动温度；本课主线） |

```bash
cd mountain-car-continuous-sac
python sac_train.py
python sac_test.py
```

前置对照：

- 稠密奖励连续控制：[`../pendulum/`](../pendulum/)（PPO / SAC）  
- 离散格子 / 悬崖：[`../grid-world/`](../grid-world/)、[`../cliff-walking/`](../cliff-walking/)  
