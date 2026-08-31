# 评估尺（B 方案第 0 步）

固定牌谱 + **换座** + 同报 **WP / ADP**。以后「比 DouZero 更好」只认这把尺子。

本步**不写 PPO**。上游克隆在 [`../DouZero/`](../DouZero/)；分析见 [`../DouZero-Analysis.md`](../DouZero-Analysis.md)。

## 口径

每副牌打两局：

1. A 地主，B 两个农民  
2. B 地主，A 两个农民  

| 指标 | 定义 |
|------|------|
| **WP** | 该侧在 2N 局里的胜率 |
| **ADP** | 论文口径：该局地主赢 `+2^k`、农民赢 `-2^k`（k=炸弹数）；换座后对 A 取平均 |

A 更好：这把尺子上 **WP > 0.5 且 ADP > 0**。

DouZero 自带 `evaluate.py` **不换座**，只报「这一座位上的地主 vs 农民」，不能直接当对打结论。

## 命令

在本目录执行：

```bash
cd experimental-rl/doudizhu/eval-ruler
python test_eval_ruler.py
python generate_eval_data.py --num_games 1000 --seed 0
python eval_ruler.py --side_a random --side_b random
```

本机 seed=0 的 1000 副牌（2000 局）随机对随机：

```text
side A (random)  WP=0.4995  ADP=0.0075
side B (random)  WP=0.5005  ADP=-0.0075
```

对打官方 DouZero-ADP：从 [Google Drive](https://drive.google.com/drive/folders/1NmM2cXnI5CIWHaLJeoDZMiwt6lOTV_UB) 或 [百度网盘](https://pan.baidu.com/s/18g-JUKad6D8rmBONXUDuOQ)（提取码 4624）下载权重，放到例如 `../DouZero/baselines/douzero_ADP/`（需含 `landlord.ckpt`、`landlord_up.ckpt`、`landlord_down.ckpt`）：

```bash
python eval_ruler.py --side_a ../DouZero/baselines/douzero_ADP --side_b random
```

`--side_a` / `--side_b` 只能是 `random`、`rlcard`，或**三份权重所在目录**。换座时农民位必须用 `landlord_up` / `landlord_down` 网络，不能复用地主头。

## 文件

| 文件 | 作用 |
|------|------|
| `generate_eval_data.py` | 可复现牌谱（seed=0） |
| `eval_ruler.py` | 换座 WP / ADP |
| `test_eval_ruler.py` | 聚合、发牌、手牌不被原地改掉 |

`GameEnv` 会原地 `remove` 手牌；尺子里每局 `deepcopy`，同一 `.pkl` 才能打两遍。
