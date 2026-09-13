# 斗地主（B 方案）

不完全信息三人牌：地主 vs 两农民。目标是在**同一把评估尺**上超过 DouZero。

官方仓库：[kwai/DouZero](https://github.com/kwai/DouZero)（ICML 2021）。源码拆解：[`DouZero-Analysis.md`](DouZero-Analysis.md)。

| 子目录 | 内容 |
|--------|------|
| [`eval-ruler`](eval-ruler/) | **第 0 步：评估尺**（固定牌谱 + 换座 WP/ADP） |
| [`doudizhu-env`](doudizhu-env/) | **第 1 步：规则接口** |
| [`doudizhu-ppo`](doudizhu-ppo/) | **第 2 步：合法集 PPO** |
| [`doudizhu-ppo-critic`](doudizhu-ppo-critic/) | **第 3 步：完美信息 Critic** |
| [`doudizhu-ppo-selfplay`](doudizhu-ppo-selfplay/) | **第 4 步：三位置自博弈（WP）** |
| [`doudizhu-adp`](doudizhu-adp/) | **第 5 步：ADP 课程（地主 vs random）+ 自博弈 + 换座对打** |
| [`doudizhu-dmc`](doudizhu-dmc/) | **对照：同一尺子 + 同一特征的 DMC**（`MSE(Q, G)`，不是 PPO） |
| [`doudizhu-dmc-scale`](doudizhu-dmc-scale/) | **加量第 1 刀：多 Actor + replay + 断点续训** |
| [`doudizhu-ppo-feat`](doudizhu-ppo-feat/) | **C：公开牌型特征 + oracle V**（PPO，评估不泄密） |
| `DouZero/` | 上游克隆（gitignore，不入库） |

```bash
cd eval-ruler && python test_eval_ruler.py
cd ../doudizhu-env && python test_env.py
cd ../doudizhu-ppo && python test_ppo.py
cd ../doudizhu-ppo-critic && python test_critic.py
cd ../doudizhu-ppo-selfplay && python test_selfplay.py
cd ../doudizhu-adp && python test_adp.py
cd ../doudizhu-dmc && python test_dmc.py
cd ../doudizhu-dmc-scale && python test_scale.py
cd ../doudizhu-ppo-feat && python test_feat.py
```

准备：`git clone --depth 1 https://github.com/kwai/DouZero.git` 到本目录下的 `DouZero/`；`pip install torch numpy`。官方 ADP 权重需自行下载到 `DouZero/baselines/douzero_ADP/`。

## 尺子上的数

同一把换座尺子（50 副 / 100 局，牌谱 seed=0）。官方 ADP 是大规模 DMC 训出来的权重。

| 做法 | 对 random | 对官方 DouZero-ADP |
|------|-----------|---------------------|
| PPO 加量自博弈 `ppo_adp_scale.pth` | WP 0.90 | WP ~0.07 |
| 短训 DMC `dmc_adp.pth` | WP 0.80 | WP 0.068 |
| 公开牌型 PPO `ppo_feat.pth` | WP 0.97 | WP 0.14 |
| DMC 4 Actor × 200 次 `dmc_scale.pth`（短加量） | WP 0.82 | WP 0.09 |
| 同上续到约 8650 次（best） | WP 0.87 | WP **0.20** |

「更好」（WP>0.5 且 ADP>0）对官方：**都没有。**

短训换公式（PPO → DMC）不动位置；加公开牌型到 0.14；堆几千局到 0.20。本机短训到此为止，缺的是官方那种量级，不是再开一个算法目录。
