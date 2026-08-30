# DouDizhu

总览：[`../README.md`](../README.md)。

这里**不是** Gym 主线上的一课：没有本仓库写的 `*_train.py` / 验收阈值。
第三方对照：[`DouZero/`](DouZero/)（[kwai/DouZero](https://github.com/kwai/DouZero)，ICML 2021）。
完整斗地主把主线里拆开教的几件事叠在一起，用来对照，不建议当下一课从 `train.py` 开跑。

Windows 上 GPU Actor 不可用；完整训练按天计。预训练权重不在仓库里。

---

## 1. 和 Gym 主线怎么对上

| 主线里已有 | 斗地主上变成 |
|------------|----------------|
| GridWorld 动作掩码 | 每回合合法出牌集合不同，规模到 2 万+ |
| MountainCar 终局才成功 | 中间步 `r=0`，只在 `done` 给胜负 / 炸弹分 |
| 看不见墙外 | `get_obs` 只用「另外两家手牌的并集」 |
| 单智能体 | 三网：地主 / 上家 / 下家（农民还要合作） |

所以论文里 DQN、A3C 几乎打不过规则 Bot：动作太多时 `max Q` 会高估，策略头也撑不住变长的合法集。

DouZero 选的是 **Deep Monte-Carlo（DMC）**，不是 PPO / TD：

```text
玩完一整局
对轨迹里每个 (s, a)：
    目标 = 这局最终回报 G     （Monte Carlo，不 bootstrap）
    MSE 让 Qθ(s, a) 去拟合 G
下一步 ε-greedy 按 Q 在合法动作里选
```

| | CartPole DQN | **DouZero DMC** | CartPole PPO |
|--|--------------|-----------------|--------------|
| 学什么 | `Q(s)` 出 2 个头 | **`Q(s,a)` 标量** | `π(a\|s)` |
| 目标 | `r + γ max Q'`（TD） | **整局同一个 G** | GAE + clip |
| 动作怎么进网络 | 输出维 = 动作数 | **54 维牌矩阵拼进输入** | Categorical |
| 为何适合这里 | 动作太多时 `max Q` 炸 | 不对 2 万维做 `max` | 合法集每步都在变 |

公式直觉见 [`../cart-pole/cart-pole-deep-dive.adoc`](../cart-pole/cart-pole-deep-dive.adoc) 里「整局 return」那一节；这里拟合的是 Q，不是 `log π`。

---

## 2. 目录

```text
doudizhu/
  README.md                 # 本说明
  DouZero/
    train.py / evaluate.py / generate_eval_data.py
    douzero/
      env/          # 规则 + 特征（该先读）
      dmc/          # DMC 训练
      evaluation/   # 固定牌谱对打
    baselines/      # 空的，只有 put_pretrained_models_here
```

没有叫牌。`reset()` 直接洗牌：地主 20 张、两农民各 17 张。
依赖：`torch`；评估里的规则 Bot 还要 `rlcard`。见 `DouZero/requirements.txt`。

读序：

1. `DouZero/douzero/env/game.py` — 规则、合法出牌、`InfoSet`
2. `DouZero/douzero/env/env.py` — `_cards2array`、`get_obs`、`_get_reward`
3. `DouZero/douzero/evaluation/deep_agent.py` — 对合法动作逐个打 Q，argmax
4. `DouZero/douzero/dmc/models.py` — 地主 / 农民两套 LSTM+MLP，输出 1 维
5. `DouZero/douzero/dmc/utils.py` 的 `act()` — 终局回填同一个 G
6. `DouZero/douzero/dmc/dmc.py` 的 `learn()` — MSE + 权重拷回 Actor

`env/`（规则+特征）远重于 `dmc/` 的公式。先把 `MovesGener` 和 `_cards2array` 读通，比抠 `free_queue` 有用。

---

## 3. 一局里数据怎么走

```text
GameEnv（规则、合法出牌）
    ↓ DummyAgent 隔离「谁在想」和「牌桌」
Env.reset / step          ← 类 Gym
    ↓ get_obs(infoset)    ← 部分可观测 + 每个合法动作编成一行
Actor: Q(s,a) 对合法动作打分，ε-greedy
    ↓ 终局才有 r
每个 (s,a) 的 target = 同一局的最终回报 G   （γ=1）
Learner: MSE(Q, G)，三位置三套网、三个 RMSprop
```

`DummyAgent` 不是算法：外面先 `set_action`，引擎再 `act()`，好让训练长得像 `env.step(action)`。

`InfoSet` 里其实有完美信息（三家手牌都在），但 `get_obs()` **不用**各家分开的手牌，只用并集。
`all_handcards` 是以后接 PerfectDou 式「训练时看三家牌」的钩子；当前 DMC 没用它。

---

## 4. 特征与网络

`_cards2array` 把一手牌编成 **54 维**：`4×13` 点数矩阵按列展开，再拼大小王。花色丢掉，只留张数。手牌、已出牌、动作都走这套。

`get_obs()` 对**每个合法动作复制一行状态**，只改最后 54 维的「这一手」：

| 张量 | 含义 |
|------|------|
| `x_batch` | `[合法动作数, 状态+动作]` |
| `z_batch` | 最近 15 手，每 3 手拼成 162 维 → `5×162`，进 LSTM |
| `x_no_action` | 不含动作；训练时再和 `obs_action` 拼回去（省 buffer） |

| 位置 | `x_no_action` | + 动作 54 | 网络 |
|------|---------------|-----------|------|
| 地主 | 319 | 373 | `LandlordLstmModel` |
| 农民 | 430 | 484 | `FarmerLstmModel` |

农民多：地主最近一手、队友最近一手、地主/队友剩余张数（one-hot）。
地主看不到「队友」，只看两个农民的已出牌和剩余张数。

决策：对 `x_batch` 一次前向，取 argmax。只有一手合法时（常见是只能 pass）不跑网络。
这是 **Q(s,a) 标量 + 动作编码**，不是 27472 维 softmax。

---

## 5. 奖励

终局 `_get_reward()`（地主视角）：

| `--objective` | 地主赢 | 农民赢 |
|---------------|--------|--------|
| `wp` | +1 | −1 |
| `adp`（默认） | `+2^炸弹数` | `−2^炸弹数` |
| `logadp` | `炸弹数+1` | 相反 |

农民的 target 是地主回报的相反数（`utils.py` 里 `p == 'landlord' else -episode_return`）。
两个农民共用同一个零和目标，没有单独的「队友奖励」。
一局里该位置每一步的 target 都是**同一个**终局 G。

---

## 6. 能跑什么、别跑什么

在 `DouZero/` 下：

```bash
# 生成固定牌谱（评估用，不是训练）
python generate_eval_data.py --num_games 1000

# 评估（需先把预训练权重放到 baselines/）
# 权重见 DouZero/README.md 的 Evaluation（Google Drive / 百度网盘）
python evaluate.py --landlord baselines/douzero_ADP/landlord.ckpt --landlord_up random --landlord_down random
```

`evaluate.py` 默认路径是 `baselines/douzero_ADP/...`，仓库里没有这些 ckpt。
评估打印的 WP / ADP 是「地主这边 vs 农民那边」的合计，不是论文里「同一副牌换座位打两局」的完整协议。

训练（不建议在本机当课来跑）：

```bash
# Windows：只能 CPU Actor
python train.py --actor_device_cpu --training_device cpu
```

| | 训练 | 评估 |
|--|------|------|
| 环境 | `Env` + DummyAgent | `GameEnv` + 真 Agent |
| 发牌 | 每局随机 | `generate_eval_data.py` 先生成固定 `.pkl` |
| 决策 | 带 ε 的 `Model.forward` | `DeepAgent` 纯贪心 |

默认 `total_frames=1e11`，等于一直跑到你停。工程是 TorchBeast 风格：多进程 Actor + 共享 buffer + Learner 更新后立刻拷回。

---

## 7. 建议怎么读、不要怎么用

由易到难：

1. 单测规则：随机一手牌，打印 `legal_actions`，对一下会不会漏顺子/飞机。
2. 装上预训练权重，跑 `generate_eval_data.py` + `evaluate.py`，看 `DeepAgent` 对一局的 Q 排序。
3. 若要自己写最小核：只留「两人、单张/对子、`_cards2array`、MC 更新 Q(s,a)」。不要从 `train.py` 往下抄并行。

缺的：预训练权重、叫牌、论文里的 SL 人类数据、Demo GUI（在 [rlcard-showdown](https://github.com/datamllab/rlcard-showdown)）。

官方说明：[`DouZero/README.zh-CN.md`](DouZero/README.zh-CN.md)。
