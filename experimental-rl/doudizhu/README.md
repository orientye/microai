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
| [`doudizhu-adp`](doudizhu-adp/) | **第 5 步：ADP + 换座对打** |
| `DouZero/` | 上游克隆（gitignore，不入库） |

```bash
cd eval-ruler && python test_eval_ruler.py
cd ../doudizhu-env && python test_env.py
cd ../doudizhu-ppo && python test_ppo.py
cd ../doudizhu-ppo-critic && python test_critic.py
cd ../doudizhu-ppo-selfplay && python test_selfplay.py
cd ../doudizhu-adp && python test_adp.py
```

准备：`git clone --depth 1 https://github.com/kwai/DouZero.git` 到本目录下的 `DouZero/`；`pip install torch numpy`。官方 ADP 权重需自行下载到 `DouZero/baselines/douzero_ADP/`。
