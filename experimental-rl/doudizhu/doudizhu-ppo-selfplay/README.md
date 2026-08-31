# 三位置自博弈（B 方案第 4 步）

三个座位都用合法集 PPO + 完美信息 Critic。终局 **WP**：地主 +G，两个农民都是 **−G**（与 DouZero 相同零和，没有单独队友奖）。

相对第 2/3 步：农民不再随机出牌，而是各自一套网（`landlord_up` / `landlord_down` 输入 484 维）。本步仍用 WP，**不上 ADP**。

评估仍是：贪心地主 vs **随机**农民（和第 2 步同一口径，方便对照）。对打官方 DouZero 是第 5 步。

## 命令

```bash
cd experimental-rl/doudizhu/doudizhu-ppo-selfplay
python test_selfplay.py
python ppo_train.py
python ppo_test.py
```
