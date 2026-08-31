# Dou Dizhu 规则接口（B 方案第 1 步）

对 DouZero `GameEnv` 的薄封装：`reset` / `step` / `legal_actions` / `get_obs` / `all_handcards`。  
**不写 PPO**，不重写牌型引擎。

上游克隆：[`../DouZero/`](../DouZero/)。评估尺：[`../eval-ruler/`](../eval-ruler/)。

## 接口

| 成员 | 作用 |
|------|------|
| `reset(deal=None, seed=None)` | 固定牌谱或随机开局；返回部分观测 |
| `step(action)` | 出一手（牌点列表）；终局才有 `reward` |
| `legal_actions` | 当前合法集（空桌没有 pass） |
| `get_obs()` | DouZero `get_obs`：每个合法动作一行 |
| `all_handcards` | 三家手牌（完美信息；执行时 Actor **不能**看） |
| `objective` | `wp` / `adp` / `logadp` |

`reset(deal=)` 会 `deepcopy`，同一副牌可打多遍。

## 命令

```bash
cd experimental-rl/doudizhu/doudizhu-env
python test_env.py
```

单测覆盖：空桌不能 pass；炸弹 / 顺子 / 飞机 / 王炸在合法集里；`all_handcards` 凑齐 54 张；WP 终局 ±1。
