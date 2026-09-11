# PPO + 公开牌型特征（C 第一刀）

同一把换座尺子。Actor 看 `get_obs` + 25 维公开牌型；Critic 训练时再加三家手牌 162 维。评估不传完美信息。

```bash
cd experimental-rl/doudizhu/doudizhu-ppo-feat
python test_feat.py
python feat_train.py --max_updates 200 --min_games 8 --vs_douzero
python eval_vs.py --side_a feat --feat ppo_feat.pth --side_b ../DouZero/baselines/douzero_ADP --max_deals 50 --eval_start 800
```

best：`ppo_feat.pth`。续训状态：`ppo_feat_last.pth`。
