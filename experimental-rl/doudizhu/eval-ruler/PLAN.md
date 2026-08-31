# Dou Dizhu B 方案 — 第 0 步：评估尺

> 本文件只覆盖第 0 步。第 1～5 步（规则接口、合法集 PPO、完美 Critic、自博弈、对打 DouZero）另开，不在本步写 `ppo_train.py`。

**Goal:** 固定牌谱 + 换座 + 同报 WP/ADP，作为以后所有「比 DouZero 更好」的尺子。

**Architecture:** 不改 DouZero 源码。克隆放在 `../DouZero/`（gitignore）。本目录脚本生成可复现牌谱、对同一副牌打两局（A 地主 / B 地主），按论文口径汇总 WP 与 ADP。

**Tech Stack:** DouZero `GameEnv`、numpy、pickle；评估可不装 rlcard（random 基线）。

---

### Task 1: 可复现牌谱

- Create: `generate_eval_data.py`
- Test: `test_eval_ruler.py`

- [x] 固定 `SEED=0`，每局 20+17+17 张，写出 `eval_data.pkl`

### Task 2: 换座对打

- Create: `eval_ruler.py`
- Test: `test_eval_ruler.py`

- [x] 同一副牌：A 地主 vs B 农民，再 B 地主 vs A 农民
- [x] WP = 该侧在 2N 局里的胜率
- [x] ADP = 论文口径：地主赢 `+2^k`，农民赢 `-2^k`（k=炸弹数）；换座后对 A 取平均

### Task 3: 基线跑通

- [x] `random` vs `random` 换座，WP 应接近 0.5（本机 1000 副：WP=0.4995）
- [x] README 写明官方权重放到 checkpoint 目录后如何对打 DouZero-ADP
