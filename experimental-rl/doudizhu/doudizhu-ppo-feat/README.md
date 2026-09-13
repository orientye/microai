# PPO + 公开牌型特征（C 第一刀）

同一把换座尺子。Actor 看 `get_obs` + 25 维公开牌型；Critic 训练时再加三家手牌 162 维。评估不传完美信息。

信念头：公开观测预测另外两家手牌（108 维），训练标签来自 `all_handcards`，评估只用预测。best 存 `ppo_feat_belief.pth`。

```bash
cd experimental-rl/doudizhu/doudizhu-ppo-feat
python test_feat.py
python feat_train.py --max_updates 200 --min_games 8 --vs_douzero
python eval_vs.py --side_a feat --feat ppo_feat_belief.pth --side_b ../DouZero/baselines/douzero_ADP --max_deals 50 --eval_start 800
```

C 无信念：`ppo_feat.pth`，对官方 WP=0.14。信念头 200 次、50 副换座：对 random WP=0.91；对官方 **WP=0.11**。没过「更好」。
