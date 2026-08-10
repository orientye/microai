# GridWorld + 表格 Q-learning 示例解析

本目录用 **表格 Q-learning** 训练自定义的 **5×5 固定地图 GridWorld**，并可视化贪心策略。状态完全离散、可直接查表，适合在上神经网络（见同级目录 `grid-world-dqn`）之前把 MDP 与 TD 更新看清楚。

| 文件 | 作用 |
|------|------|
| `grid_world_env.py` | 固定布局环境，观察为格子 id |
| `q_learning_train.py` | ε-greedy + 表格 TD 更新 |
| `q_learning_test.py` | 全起点贪心评估 + 随机起点 demo |
| `q_table.npy` | 训练得到的 Q 表（`shape = (25, 4)`） |
| `reward_history.png` | 训练回报曲线 |
| `policy_map.png` | `V(s)=max_a Q(s,a)` 热力图 + 箭头 |

```bash
cd experimental-rl/grid-world/grid-world-qlearning
python q_learning_train.py
python q_learning_test.py
```

依赖：`gymnasium`、`numpy`、`matplotlib`（不需要 torch）。

随机障碍 / 看整张图的版本见：`../grid-world-dqn/`。

---

## 1. 问题：GridWorld 在学什么

Agent 从左上角起点 `S` 走到右下角终点 `G`，中间有障碍 `#`。目标不是「随便到终点」，而是 **尽快到终点**（每多走一步就多扣一点分）。

地图（第 0 行在上）：

```text
S . . # .
. . . # .
. . . . .
. # . . .
. . . . G
```

| 概念 | 本例取值 |
|------|----------|
| 状态 `s` | 格子坐标 `(row, col)`，编码成整数 `0..24` |
| 动作 `a` | `0`↑ `1`→ `2`↓ `3`← |
| 奖励 `r` | 普通一步 `-0.01`；到达 `G` 的那一步 `+1` |
| `terminated` | 踩到 `G` |
| `truncated` | 超过 `100` 步仍未到达 |
| 撞边界 / 撞障碍 | **位置不变**，但仍算走了一步，照样扣 `-0.01` |

和 CartPole 的关键差别：

| | CartPole | 本例 GridWorld |
|--|----------|----------------|
| 状态 | 连续 4 维向量 | **25 个离散格子** |
| Q 的表示 | 神经网络 | **一张表** `Q[25][4]` |
| 奖励形态 | 每步 `+1`（稠密） | 几乎全是小惩罚，终点一次大奖 |
| 可解释性 | 难直接「看懂策略」 | 每格画一个箭头就能看懂 |

---

## 2. 环境怎么建模（`grid_world_env.py`）

### 2.1 状态编码

格子 `(r, c)` 与整数 id 互转：

```text
id = r * size + c
(r, c) = (id // size, id % size)
```

例如 `size=5`：

| 格子 | id |
|------|----|
| `(0,0)` 起点 | `0` |
| `(0,4)` | `4` |
| `(4,4)` 终点 | `24` |

`observation_space = Discrete(25)`，`action_space = Discrete(4)`，完全符合 Gymnasium 接口：`reset()` / `step()` 返回标准五元组。

### 2.2 转移规则

```text
当前格子 (r,c) + 动作 a
        │
        ▼
  算出候选 (nr, nc)，再裁剪到 [0, size-1]
        │
        ▼
  若 (nr,nc) 是障碍 → 留在原地
  否则 → 移动到 (nr,nc)
        │
        ▼
  若到终点 → reward=+1, terminated=True
  否则     → reward=-0.01
  若步数用尽 → truncated=True
```

注意：**障碍格子本身永远不会成为 Agent 的位置**；撞上时状态 id 不变。这会让「朝墙走」在 Q 表里学到较低价值——因为白扣了逐步惩罚却没靠近终点。

### 2.3 为什么要逐步惩罚

若只有终点 `+1`、中间 `0`，很多条长度不同的路径回报一样（都是 `1`），Agent 没有动力找短路径。  
加上 `step_penalty=-0.01` 后，最优回报大致是：

```text
return ≈ 1 + (-0.01) × (到达前的普通步数)
```

本例最短可行路径 **8 步**：前 7 步各 `-0.01`，第 8 步到终点拿 `+1`，总回报 **0.93**。  
训练日志里 `greedy_mean=0.930` 就说明已经学到这条最短路（或同等长度的路）。

---

## 3. 算法：表格 Q-learning

### 3.1 Q 值是什么

`Q(s, a)` ≈「在格子 `s` 选动作 `a`，从这一步起往后能攒多少折扣回报」。

- 不是「这一步的即时奖励」（即时奖励只有 `-0.01` 或 `+1`）
- 是「这条决策链的长期好坏」
- 贪心策略：`π(s) = argmax_a Q(s, a)`
- 状态价值：`V(s) ≈ max_a Q(s, a)`（`policy_map.png` 画的就是它）

本例 Q 表形状：

```text
Q[state_id][action]   →   shape (25, 4)

例如 Q[0] = [Q(↑), Q(→), Q(↓), Q(←)]   # 在起点四个方向的分数
```

一开始全是 `0`；Agent 乱走、偶尔撞到终点后，成功经验会通过 TD 更新 **从终点往回传**。

### 3.2 更新公式

经典单步 Q-learning（离策略 TD）：

```text
TD 目标:  y = r + γ · max_{a'} Q(s', a')     （若 s' 是终局，则 y = r）
TD 误差:  δ = y − Q(s, a)
更新:     Q(s, a) ← Q(s, a) + α · δ
```

对应代码：

```python
td_target = reward
if not done:
    td_target += GAMMA * float(np.max(q[next_state]))
td_error = td_target - q[state, action]
q[state, action] += ALPHA * td_error
```

| 符号 | 本例取值 | 含义 |
|------|----------|------|
| `α` | `0.1` | 学习率：每次朝目标挪一点，别一次改太猛 |
| `γ` | `0.99` | 折扣：更在乎近期回报，但仍看得较远 |
| `max_{a'} Q(s',a')` | 对下一状态取最优动作价值 | **离策略**：不管下一步实际会不会乱探索，更新都按「以后会最优地玩」来估 |

对比你仓库里 CartPole 的 Double DQN：那里也是同一套 Bellman 目标，只是 `Q` 换成神经网络，并加了 replay / target net。GridWorld 用查表，**没有过估计爆炸、也无需经验回放**，所以实现极短。

### 3.3 为什么叫「离策略」

- **行为策略（behavior）**：ε-greedy，有时故意乱走，用来收集经验。
- **目标策略（target）**：更新里用 `max`，等价于假设以后永远贪心。

于是可以用「探索时踩到的样本」去学「不探索时的最优策略」。  
[`../../cliff-walking/cliff-walking-q-sarsa/`](../../cliff-walking/cliff-walking-q-sarsa/) 用 **SARSA**（用「实际下一步动作」的 Q 来更新）在悬崖地图上与 Q-learning 对照：两者策略会分叉。

### 3.4 ε-greedy 探索

```text
以概率 ε → 随机选 4 个方向之一
以概率 1−ε → 选当前 Q 最大的方向
```

本例 ε 按 episode **线性衰减**：

```text
ε: 1.0  ────────────────►  0.05
     episode 0 … 3000 … 4000
```

前期必须大探索：否则 Q 全 0，Agent 可能卡在局部、很久碰不到 `G`。  
后期降低探索：让策略稳定在已学到的短路径上。评估时用 **纯贪心**（`ε=0`），所以日志里 `train_return` 可能仍抖，而 `greedy_mean` 很稳。

---

## 4. 训练循环在干什么（`q_learning_train.py`）

```text
初始化 Q = 全零表 (25×4)
for episode = 1 .. 4000:
    reset → 回到起点
    while 未结束:
        用 ε-greedy 选 a
        env.step(a) → (s', r, done)
        用上面的公式更新 Q(s,a)
        s ← s'
    每隔 200 局：纯贪心跑 20 局，打印平均回报
保存 q_table.npy / reward_history.png / policy_map.png
```

一次成功到达终点后，价值大致这样回传（示意）：

```text
… → 邻近终点的格子  →  终点
        Q 变大              拿到 +1
           ▲
           │  γ · max Q
           │
更远的格子也会慢慢变「更值得朝终点方向走」
```

障碍附近的「朝墙走」会反复得到 `-0.01` 且状态不变，Q 偏低，于是箭头会避开墙。

---

## 5. 如何读训练 / 测试结果

### 5.1 为什么测试不再只从固定 `S` 出发

只重复固定起点的贪心轨迹，五局结果往往完全一样，**不能说明整张地图都会走**。  
本例改为：

| 阶段 | 做法 |
|------|------|
| 训练 | 每局 `random_start=True`，从任意可行格出发 |
| 训练中评估 / `q_learning_test.py` | 对**所有**非障碍、非终点格子做纯贪心，统计成功率与平均回报 |
| 额外 demo | 再随机抽几个起点打印路径（展示用，分数以 all-start 为准） |

说服力来自「起点覆盖全图」，不是来自「同一确定轨迹多跑几次」或「测试时乱选动作」。

### 5.2 控制台

训练一段时间后常见输出：

```text
episode= 200  eps=0.937  train_return=...  all_starts_mean=0.969  success=100%
...
Learned greedy policy (↑→↓←):
→ ↓ ↓ # ↓
→ → ↓ # ↓
↓ → → → ↓
↓ # ↓ ↓ ↓
→ → → → G
```

- `all_starts_mean` / `success`：对所有可行起点纯贪心评估。
- 策略图：从任意起点跟箭头走，应绕过障碍到达 `G`。

实测最短路径之一（8 步）：

```text
(0,0) → (0,1) → (1,1) → (1,2) → (2,2) → (2,3) → (2,4) → (3,4) → (4,4)
```

回报：`7 × (-0.01) + 1 = 0.93`。

### 5.3 `policy_map.png`

- 颜色：`V(s) = max_a Q(s,a)`，越靠近终点通常越高（折扣后仍呈「由远到近升高」的趋势）。
- 箭头：该格的贪心动作。
- `#` / `G`：障碍与终点（终点不画箭头）。

### 5.4 `reward_history.png`

- 训练回报滑动平均：因为训练时还有探索，会上下抖。
- all-starts 评估点：更能反映「整张图都会不会走」。

---

## 6. 测试脚本（`q_learning_test.py`）

1. `np.load("q_table.npy")` 读表  
2. 打印整图贪心策略  
3. 对**所有可行起点**做贪心评估（成功率 / 平均回报）  
4. 再随机抽几个起点打印路径（展示用）  

用来确认：不是只从角落 `S` 碰巧会走，而是整张图都会走。

---

## 7. 和 CartPole DQN / PPO 的关系

```text
本例 GridWorld          CartPole Double DQN           CartPole PPO
─────────────────       ─────────────────────         ────────────────
表格 Q(s,a)             网络近似 Q(s,a)               网络输出 π(a|s) 与 V(s)
ε-greedy 选动作         ε-greedy 选动作               按概率采样动作
TD + max 更新           TD + Double/target            策略梯度 + GAE
离散小状态              连续状态                      连续状态
无 replay               需要 replay                   on-policy rollout
```

建议的认知顺序：

1. **本例**：看清 `Q`、TD、探索、回报回传  
2. **`../../cliff-walking/cliff-walking-q-sarsa`**：同一套表，对比 Q-learning vs SARSA（离策略 / 在策略）  
3. **`../grid-world-dqn`**：随机地图 + 网格观察 + Double DQN  
4. **CartPole DQN / PPO**：连续状态上的值函数 / 策略梯度  

---

## 8. 常见问题

**Q: 为什么障碍格子也有 Q 行，但策略图不画它们？**  
A: 状态空间按 `size*size` 建表，障碍 id 理论上存在；但转移不会让 Agent 停在障碍上，那些行基本保持初值，可视化时直接标 `#`。

**Q: `terminated` 和 `truncated` 都令 `done=True`，更新时都不 bootstrap，对吗？**  
A: 对 `terminated`（到终点）正确：没有未来。  
对 `truncated`（超时）偏保守：严格说超时不是「世界终结」，还应 bootstrap；本例地图很小，正常学成后几乎不会超时，影响可忽略。

**Q: 训练回报偶尔很低，但 all_starts 评估已经很高？**  
A: 正常。训练仍在按 ε 随机探索；评估是纯贪心，且覆盖所有起点。

**Q: 如何确认「最短」？**  
A: 无障碍时曼哈顿距离是 `8`；有障碍后最短仍是 `8`。回报 `0.93` 即对应 8 步最优。若长期停在 `0.92`（9 步），说明还次优，可加长训练或检查障碍布局。

---

## 9. 地图布局不固定？

表格 Q 只有格子坐标，默认墙永远在同一处。布局一变就无法泛化。  
随机障碍 + 看见整张图的版本在同级目录：

```bash
cd ../grid-world-dqn
python dqn_train.py
python dqn_test.py
```

---
