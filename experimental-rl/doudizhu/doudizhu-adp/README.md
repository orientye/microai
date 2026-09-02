# ADP + 换座对打（B 方案第 5 步）

自博弈奖励改成 **ADP**（`±2^炸弹数`），梯度裁剪 **40**（对齐 DouZero）。  
对打走第 0 步尺子：同一副牌换座，同报 WP / ADP。

加量自博弈：`ppo_scale.py` 从 `ppo_adp.pth` 接着训（每更新 64 局，早停 WP≥0.90），存 `ppo_adp_scale.pth`。

「更好」：**WP > 0.5 且 ADP > 0**。没下官方权重时，先对 `random`；权重放到目录后把 `--side_b` 换成该目录。

## 命令

```bash
cd experimental-rl/doudizhu/doudizhu-adp
python test_adp.py
python ppo_curriculum.py
python ppo_train.py
python ppo_scale.py
python ppo_clone.py
python ppo_vs_douzero.py
python eval_vs.py --side_a ppo --side_b random --max_deals 50
python eval_vs.py --side_a ppo --ppo ppo_adp_bc.pth --side_b ../DouZero/baselines/douzero_ADP --max_deals 200
```
