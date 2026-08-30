# DouDizhu

总览：[`../README.md`](../README.md)。

第三方对照：[`DouZero/`](DouZero/)（[kwai/DouZero](https://github.com/kwai/DouZero)，ICML 2021）。不是本仓库的教学课，没有 `*_train.py` / 验收阈值。

Gym 主线（表格 → DQN → PPO → SAC）见总览推荐顺序。这里只放完整斗地主系统，用来对照：

- 合法动作枚举 + 动作编码成牌矩阵
- Deep Monte-Carlo：整局回报拟合 `Q(s,a)`，不是 PPO / TD
- 三位置自博弈（地主 / 上家 / 下家）

先读 `DouZero/douzero/env/`（规则与 `get_obs`），再读 `DouZero/douzero/evaluation/deep_agent.py` 和 `DouZero/douzero/dmc/`。

预训练权重不在仓库里，见 `DouZero/README.md` 的 Evaluation。Windows 上 GPU Actor 不可用；完整训练按天计，不建议当下一课来跑。
