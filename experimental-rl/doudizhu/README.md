# 斗地主（B 方案）

不完全信息三人牌：地主 vs 两农民。目标是在**同一把评估尺**上超过 DouZero。

官方仓库：[kwai/DouZero](https://github.com/kwai/DouZero)（ICML 2021）。源码拆解：[`DouZero-Analysis.md`](DouZero-Analysis.md)。

| 子目录 | 内容 |
|--------|------|
| [`eval-ruler`](eval-ruler/) | **第 0 步：评估尺**（固定牌谱 + 换座 WP/ADP；先看这个） |
| `DouZero/` | 上游克隆（gitignore，不入库） |

```bash
cd eval-ruler
python test_eval_ruler.py
python generate_eval_data.py --num_games 1000 --seed 0
python eval_ruler.py --side_a random --side_b random
```

尚未开目录：规则接口、合法集 PPO、完美信息 Critic、三位置自博弈。

准备：`git clone --depth 1 https://github.com/kwai/DouZero.git` 到本目录下的 `DouZero/`；`pip install torch numpy`（规则 Bot 再加 `rlcard`）。
