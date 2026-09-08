# DMC 对照（同一尺子、同一特征）

不是第 6 步 PPO。算法是 DouZero 的 **Deep Monte-Carlo**：合法集上打 `Q(s,a)`，ε-greedy，每步 target 是**整局同一个 G**，`MSE(Q, G)`，样本进 replay 可反复用。

特征仍是 DouZero 的 `get_obs`（54 维编码）。网络宽度对齐本仓库 PPO scorer（256），方便对照「换公式」而不是换特征。

「更好」仍只认第 0 步尺子：**换座 WP>0.5 且 ADP>0**。

## 本机短训（和 PPO 同一把尺子）

| 权重 | 对手 | 副数 | WP | ADP |
|------|------|------|-----|------|
| `dmc_landlord.pth` 地主 | random 农民 | 留出 200 | **0.89** | **+1.29** |
| `dmc_adp.pth` 三人组 | random | 200 副换座 | **0.80** | **+0.87** |
| `dmc_adp.pth` 三人组 | 官方 DouZero-ADP | 200 副换座 | **0.068** | **−1.33** |

和 `doudizhu-adp` 的 PPO 一样：打得过 random，打不过官方权重。短训 DMC 没有把 WP 从 0.07 抬出去。

## 命令

```bash
cd experimental-rl/doudizhu/doudizhu-dmc
python test_dmc.py
python dmc_curriculum.py
python dmc_train.py
python eval_vs.py --side_a dmc --side_b random --max_deals 200
python eval_vs.py --side_a dmc --dmc dmc_adp.pth --side_b ../DouZero/baselines/douzero_ADP --max_deals 200
```
