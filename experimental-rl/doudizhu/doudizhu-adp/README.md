# ADP + 换座对打（B 方案第 5 步）

自博弈奖励改成 **ADP**（`±2^炸弹数`），梯度裁剪 **40**（对齐 DouZero）。  
对打走第 0 步尺子：同一副牌换座，同报 WP / ADP。

先 **课程**：只训地主、农民 random，best 按固定牌谱的地主 WP/ADP 存。打过 random 再 `ppo_train.py` 三家自博弈。

「更好」：**WP > 0.5 且 ADP > 0**。没下官方权重时，先对 `random`；权重放到目录后把 `--side_b` 换成该目录。

## 命令

```bash
cd experimental-rl/doudizhu/doudizhu-adp
python test_adp.py
python ppo_curriculum.py
python ppo_train.py
python eval_vs.py --side_a ppo --side_b random --max_deals 50
python eval_vs.py --side_a ppo --side_b ../DouZero/baselines/douzero_ADP --max_deals 1000
```
