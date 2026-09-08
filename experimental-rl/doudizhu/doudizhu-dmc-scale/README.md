# DMC 并行采样（超过 DouZero 第 1 刀）

同一套 `get_obs`、同一个 `MSE(Q, G)`。多 Actor 灌 replay，Learner 更新后把权重写回 Actor。崩了用 `--resume` 接着跑。

验收：每小时局数上去；对官方 ADP 的换座 WP **离开 0.07**。这一刀不换特征。

## 命令

```bash
cd experimental-rl/doudizhu/doudizhu-dmc-scale
python test_scale.py
python scale_train.py --actors 4 --max_updates 200 --vs_douzero
python scale_train.py --resume dmc_scale_last.pth --actors 4 --vs_douzero
```

初始化默认读 [`../doudizhu-dmc/dmc_adp.pth`](../doudizhu-dmc/dmc_adp.pth)。best 存 `dmc_scale.pth`，续训存 `dmc_scale_last.pth`。
