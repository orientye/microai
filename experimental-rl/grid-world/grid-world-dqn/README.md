# GridWorld + CNN Double DQN 示例解析

本目录用 **CNN + Double DQN** 训练 **每局障碍都可能变化** 的 5×5 GridWorld。Agent 必须**看见整张地图**（墙在哪、终点在哪、自己走过哪）才能换图泛化。

**MDP** 仍是格子世界怎么转、怎么给分；**TD** 仍是「用 Bellman 目标修正 Q 估计」。与同级 [`../grid-world-qlearning/`](../grid-world-qlearning/) 的差别：那里状态只有格子 id，Q 是一张 `(25, 4)` 的表，墙永远固定；这里 **Q 存在神经网络权重里**，输入是 **4 通道 × 5×5 网格**，同一格子在不同地图上最优动作可以不同。

**本例需要深度学习。** 依赖 `torch`；建议先跑通表格 Q-learning，再读本文。

| 文件 | 作用 |
|------|------|
| `grid_world_env.py` | 随机可解布局；观察为 **4 通道** 拼成的向量 |
| `dqn_train.py` | CNN + 课程学习 + 距离塑形 + 动作掩码；结束自动微调 |
| `dqn_test.py` | **随机多种子**评估；可选对照表格 Q |
| `inspect_fails.py` | 打印失败局的路径（排查卡死 / 绕圈） |
| `dqn_random_layout.pth` | **训练产物**（`.gitignore`）：评估最优的策略权重 |
| `dqn_reward_history.png` | **训练产物**（`.gitignore`）：回报曲线；完整跑完后是**微调**阶段的图，见 §5.3 |

```bash
cd experimental-rl/grid-world/grid-world-dqn
python dqn_train.py              # 课程 6000 局 + 自动微调 4000 局（无开关跳过）
python dqn_train.py --finetune   # 仅微调（需已有 checkpoint）
python dqn_test.py               # 最终成绩：随机 5 种子 × 每种子 200 张图
python inspect_fails.py          # 排查失败路径（可自改 SEED）
```

依赖：`gymnasium`、`numpy`、`matplotlib`、`torch`。

固定地图 + 表格 Q 见：[`../grid-world-qlearning/`](../grid-world-qlearning/)。

---

## 1. 问题：GridWorld 在学什么

Agent 仍是从起点区域走到右下角终点 `G`，中间有 **3 个随机障碍 `#`**（训练时可先用更少障碍做课程）。目标仍是 **尽快到终点**。

与固定地图版不同：**每局 `reset()` 都会重新采样障碍位置**。连通检查是 **固定左上 `(0,0) → G`**（不是即将抽到的随机起点）；通过后再从「能到达 G 的格子」里随机抽起点。因此 Agent 不能只背「格子 7 该往右」，而要学会 **看墙绕路**。

这里的 **「有路」** 指：只走上下左右、不出界、不踩 `#`，存在至少一条格路径从 `(0,0)` 到 `G`。环境用 **BFS（广度优先搜索）** 在摆好障碍后做连通检查，不通就重采（最多 300 次；仍不通则本局 **零障碍**），避免整局根本到不了终点。流程见 §2.1。

| 概念 | 本例取值 |
|------|----------|
| 状态 `s` | **4 通道 5×5 网格**（flatten 成长度 100 的向量） |
| 动作 `a` | `0`↑ `1`→ `2`↓ `3`← |
| 奖励 `r` | 见下表（环境 **base reward**；训练 replay 另加距离塑形，§3.3） |
| `terminated` | 踩到 `G` |
| `truncated` | 超过 **50** 步仍未到达 |
| 撞边界 / 撞障碍 | **位置不变**，环境给 `bump_penalty`（DQN 训练/测试有动作掩码，几乎踩不到） |
| 重访格子 | 在 `step_penalty` 之外再扣 `revisit_penalty × prev_visits` |

奖励（环境 `step()` 返回的 **base reward**，不含塑形）：

| 情况 | 奖励 |
|------|------|
| 到终点 | `+1.0` |
| 普通一步（成功移动） | `-0.01` |
| 撞墙 / 出界 | `-0.05`（环境规则；带掩码的 DQN 轨迹里几乎不出现） |
| 重访格子 | 额外 `-0.03 × prev_visits`（访问越多扣越多；撞墙原地不动时也会叠上） |

**Agent 要优化的仍是上表 base reward**（尽快到 G、少步数、少绕圈）。训练脚本还会对写入 replay 的回报加 **距离塑形**，只为了给 TD 更稠密的「是否靠近 G」信号、**加快训练收敛**，不改变 `step()` 返回值，测试也不塑形；公式与动机见 §3.3。塑形 ≠ 重访惩罚：后者是环境真实扣分，评估也算。

观察（`4 × 5 × 5`，网络内部 reshape 成卷积输入）：

| 通道 | 含义 |
|------|------|
| `agent` | 当前位置为 `1`，其余为 `0` |
| `obstacle` | 障碍格为 `1` |
| `goal` | 终点格为 `1` |
| `visited` | 本局各格访问次数 ÷ 5，clip 到 `[0,1]`（打破 A↔B 来回；带掩码时几乎不会原地撞墙） |

和表格 Q-learning 的关键差别：

| | 表格 Q-learning | 本例 CNN-DQN |
|--|----------------|--------------|
| 状态 | 格子 id `0..24` | **整张图** 4 通道 |
| Q 存在哪 | `q_table.npy` | `dqn_random_layout.pth` |
| 查 Q | `q[state, action]` | 网络前向 `QNet(obs)[action]` |
| 墙的位置 | 固定，表里没有 | **每局不同，必须在 obs 里** |
| 换一张地图 | 表失效 | 仍可用（若训练充分） |
| 可解释性 | 每格画箭头 | 需 rollout 看路径 |

为什么不能继续用表格 `Q[格子]`：同一格 `(2,2)` 在「墙在左边」和「墙在右边」时最优动作不同，状态里必须带上墙的信息。

### 1.1 为什么用 CNN，而不是把 100 个数直接丢进 MLP

环境给的是长度 100 的向量，**扁平 MLP 也能吃**（`Linear(100, …)`）。这里用卷积，不是「MLP 算不了」，而是网格世界里有一种 **MLP 默认不知道、CNN 一上来就假定** 的结构。

**邻接在地图上，不在向量下标里。**  
格子 `(1,2)` 和 `(1,3)` 在世界上是左右邻居。flatten 之后它们在向量里碰巧相邻；但 `(0,4)` 和 `(1,0)` 在向量里也是前后两个数（一行写完接下行），**地图上却隔着整行**。MLP 看到的是 100 个平等的坐标，必须自己从数据里猜「哪些下标才是邻居」。

**走路规则是局部的、到处一样。**  
「右边是墙就别往右」「终点在右下就倾向 → / ↓」这类判断，只看自己周围几格 + 终点相对位置，在 `(0,0)` 和 `(3,2)` 是同一套。CNN 的 `3×3` 核在每个格子上 **共用同一组权重**（本例 `Conv2d(..., kernel_size=3, padding=1)`），等于一次学会、整图复用。MLP 每个输入位置各有一套权重，往往要把「墙在右边」在 25 个格子上各学一遍。

用一张示意：

```text
某一步的局部 3×3（obstacle 通道）     同一模式换个位置
. # .                                 . # .
. A .   →  右边是墙，Q(→) 应该低       . A .
. . G                                 . . G
```

卷积会在两处扫到几乎相同的局部图样，输出同类信号；扁平向量里这两处对应的下标完全不同，MLP 没有「这是同一图案」的先验。

5×5 很小，扁平 MLP **也能训到能走**；随机换墙之后，CNN 通常 **用更少样本** 把「看邻格绕路」泛化开。本例 `QNet` 并不是纯卷积：两层 `Conv2d` 提局部特征，再 flatten 进 MLP 头输出 4 个 Q——先利用空间结构，再做全局决策。

感受野要算一下：一层 `3×3` 看 3 格，两层 stride-1 叠起来是 **5 格**，已经等于整张地图。所以「局部」主要体现在 **第一层邻域 + 整图权值共享**，不是第二层还只看身边 8 格。

**对应代码：**

1. 环境先按 **2D 平面** 填通道，再 `ravel` 拼成 Gym 向量（邻接被压扁，但格子顺序仍是行优先）：

```python
# grid_world_env.py _observe()
agent[self.pos] = 1.0
obstacle[r, c] = 1.0          # 仍是 (5,5) 地图
return np.concatenate([agent.ravel(), obstacle.ravel(), goal.ravel(), visited.ravel()])
```

2. 网络 **第一件事** 是把 `(batch, 100)` 还原成 `(batch, 4, 5, 5)`，否则 `Conv2d` 不知道哪 25 个数是一张图。还原成网格后，邻接才重新是「格子的上下左右」，而不是 flatten 后向量里碰巧挨着的下标；**第 3 步** 就是在这张 `(4, 5, 5)` 上扫卷积，再把整图信息压成四个动作的 Q。

```python
# dqn_train.py QNet.forward
if x.dim() == 2:
    x = x.view(-1, self.n_channels, self.grid_size, self.grid_size)
```

3. **`QNet` 分两段：卷积提局部特征，全连接头出四个 Q。** 接上面的 `view`，`forward` 先走 `conv`，再走 `head`：

   - **前半 `conv`（空间归纳偏置在这里）**：两层 `Conv2d`，`kernel_size=3` = 第一层每个输出格看自己和 8 邻格；`padding=1` = 特征图仍是 5×5。同一套卷积核权重扫遍全图（权值共享）。第二层感受野已是整张 5×5。若改成扁平 MLP，这两层通常换成 `Linear(100, …)`，就再也没有「3×3 邻域 / 换位置复用」。
   - **后半 `head`（全局决策）**：卷积输出形状是 `(batch, 64, 5, 5)`——仍是「每格一份特征」，还不是四个动作的 Q。所以要 `flatten(1)` 压成长向量，再用 **普通全连接 `nn.Linear`**（相对 Conv 而言没有邻域结构）接到 128 维，最后一层 `Linear(128, 4)` 一次给出 **四个标量**：`Q(↑), Q(→), Q(↓), Q(←)`。选动作时对这 4 个数 `argmax` 即可（对应上文「查 Q → `QNet(obs)[action]`」，但 forward 通常一次算齐四个）。

```python
# dqn_train.py QNet
self.conv = nn.Sequential(
    nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
)
self.head = nn.Sequential(
    nn.Linear(64 * grid_size * grid_size, 128),
    nn.ReLU(),
    nn.Linear(128, action_dim),  # action_dim=4 → 四个 Q
)

# forward 里（接第 2 步的 view 之后）
x = self.conv(x)
return self.head(x.flatten(1))
```

**和「纯扁平 MLP」对照：** 也可以省掉 `view` 与 `Conv2d`，用 `Linear(100, 128)` 直接吃 `_observe()` 的 100 维向量，最后同样 `Linear(128, 4)` 出四个 Q——**输出含义一样**，差别只在中间有没有卷积那一层。本例刻意 **先 conv、再 flatten 进 head**，把「看邻格绕路」交给卷积，把「在当前整图状态下选哪个方向」交给全连接头。

---

## 2. 环境怎么建模（`grid_world_env.py`）

### 2.1 随机布局与可达性

每局 `reset(options={"randomize_layout": True, "n_obstacles": k})`：

```text
从除 S、G 外的格子中随机抽 k 个作障碍
        │
        ▼
  BFS 检查 start(0,0) → goal(4,4) 是否连通
        │
   不通 → 重采样（最多 300 次）
        │
   仍不通 → 本局障碍为空集
        │
        ▼
  用 BFS 从 G 反搜，得到所有「能到 G」的格子
        │
        ▼
  random_start=True 时，起点只从上述集合里采样
```

这样不会出现「起点被困死」或「根本到不了终点」的无效局。注意：`_path_exists` 检查的是 **固定 `self.start=(0,0)` → G**，不是稍后抽到的随机起点；随机起点另由 `cells_reaching_goal()` 保证能到 G。

BFS 做法：从种子格出发，把「一步能走到的相邻空格」依次入队扩展；`_path_exists` 从 `(0,0)` 搜，能碰到 `goal` 即连通；`cells_reaching_goal` 从 `G` 反搜，得到随机起点可用的格子集合（实现见 `grid_world_env.py`）。

### 2.2 观察向量

`_observe()` 把四个 `(5,5)` 平面 **按通道顺序 flatten 再拼接**：

```text
obs = [ agent.ravel() | obstacle.ravel() | goal.ravel() | visited.ravel() ]
      └─ 25 ─┘ └─ 25 ──┘ └─ 25 ─┘ └─ 25 ──┘     → shape (100,)
```

`observation_space = Box(0, 1, shape=(100,))`。训练脚本里网络会 `view(-1, 4, 5, 5)` 还原成卷积输入。

### 2.3 转移规则

与固定地图版类似，但多了 **重访惩罚** 和 **更重的撞墙惩罚**：

```text
当前格子 + 动作 a
        │
        ▼
  算出候选格；出界或撞障碍 → bumped=True，位置不变
  否则 → 移动到新格
        │
        ▼
  更新 visit_count[pos]；记录 prev_visits
        │
        ▼
  到 G → reward=+1, terminated=True
  否则 → step_penalty 或 bump_penalty
         若 prev_visits≥1 → 再加 revisit_penalty × prev_visits
  步数≥50 → truncated=True
```

`step()` 返回 `(obs', r, terminated, truncated, info)`，其中 `info` 含 `manhattan`（到 G 的曼哈顿距离）、`bumped`、`revisits`，供塑形和调试使用。

训练和评估都在 `legal_actions()` 里选动作：静态地图上只要上一步是合法移动，来时的格子仍空着，**几乎不可能被围死**。因此 `bump_penalty` 写在环境里，带掩码的 DQN 轨迹里基本见不到；`dqn_test.py` 的表格 Q 对照**没有**掩码，才会撞墙。原地 bump 时 `prev_visits≥1`，还会叠加重访惩罚。

### 2.4 合法动作与 `avoid_revisit`

`legal_actions()`：只返回「能走进自由格」的动作（不选必然撞墙/出界的方向）。  
`legal_actions(avoid_revisit=True)` 分两段：先只留 **从未访问**（`visit < 1`）的合法邻格；若没有，再只留 **访问次数最少** 的那些。评估 / 测试用来打破贪心 A↔B 来回 oscillation。

训练选动作：ε-greedy，但在 **合法动作子集** 里随机 / argmax（**不用** `avoid_revisit`）。  
评估 / 测试：纯贪心 + 合法掩码 + `avoid_revisit=True`。

---

## 3. 算法：CNN Double DQN

### 3.1 Q 值是什么（神经网络版）

`Q(s, a)` 仍表示「在状态 `s`（整张图）选动作 `a`，从这一步起往后大概能攒多少折扣回报」。  
差别是：**不再查表**，而是由 `QNet` 一次输出 4 个动作的 Q 值：

```text
QNet(obs) → [ Q(↑), Q(→), Q(↓), Q(←) ]   shape (4,)
```

网络结构（`dqn_train.py` 里 `QNet`）：

```text
输入 (4, 5, 5)
  → Conv 4→32, ReLU
  → Conv 32→64, ReLU
  → Flatten → Linear → 128, ReLU → Linear → 4
```

### 3.2 Double DQN 更新

与仓库里 CartPole Double DQN **同构**，核心仍是 TD + Bellman，但拆成 **policy 选动作、target 估价值**，减轻 Q 过估计：

```text
当前网络 policy_net 对 (s,a) 输出 Q(s,a)
下一状态 s'：
  a* = argmax_{a'} policy_net(s')[a']     ← 用 policy 选动作
  y  = r + γ · target_net(s')[a*]         ← 用 target 估 Q(s',a*)
  若 done：y = r
损失：MSE( Q(s,a), y )
```

对应代码逻辑：

```python
q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
best_actions = policy_net(next_states).argmax(dim=1)
next_q = target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
target = rewards + GAMMA * next_q * (1.0 - dones)
loss = F.mse_loss(q_values, target)
```

交互时 `choose_action` 只在合法动作上 argmax；**TD 目标里的 `argmax` 没有掩码**，四个 Q 里裸选。合法动作上的行为和对非法动作的 Q 估计不是同一套。

#### 为什么需要 replay（experience replay）

表格 Q 可以反复改同一个 `(s, a)` 的一格；**神经网络** 一次更新会动整网参数，若 **每走一步就立刻用这一步反传**，数据和学习过程都绑在同一条时间线上，往往不稳、也浪费样本。经典 DQN 的做法是 **off-policy + replay**：先把交互存进池子，学的时候 **随机重放** 历史 transition，再用上面的 Double DQN 公式算 batch 损失。

| 若不用 replay、每步立刻学 | 用 replay 之后 |
|---------------------------|----------------|
| 连续几十步在同一张图、同一段轨迹上，梯度 **追着刚走过的分布改**，易震荡或过拟合某条路 | 从池里 **随机抽** 128 条，很多局、很多地图混在一起，接近 **打乱后的 mini-batch** |
| 某条 `(s, a)` 可能 **只出现一次** 就丢了 | 同一条「绕错 / 重访」可被 **反复抽到**，提高样本利用率 |
| `target_net` 与 `policy_net` 的目标都在 **极相关的相邻步** 上算，二者容易 **互相追着跑** | batch 里的 `s、 s'` 来自不同时间/不同局，TD 目标相对 **更稳**（仍配合下文 target 软更新） |

**在本例 grid-world 上尤其明显：** 每局 **随机换墙**（状态是整图 obs），回报又 **很稀疏**（多数步 `-0.01`，到 G 才 `+1`）。没有 replay，网络大部分时间只在「当前这一张图的连续步子」上更新，少数成功局里学到的绕障很难 **混进** 其它地图的梯度里；replay 把「曾经在某张图上走对的一步」和「现在在别张图上的步子」放进同一 batch，才更容易泛化。

表格 Q-learning 里 **一格一格往回传** 的直觉，在这里变成：**随机抽历史 transition，用同一套 TD 目标更新网络权重**；看不到「格子 23 那一行被改了」，但终点附近的好决策会通过 replay 慢慢泛化到相似图样。

（On-policy 算法如 PPO 通常 **不** 用这种大 buffer，而是用当前策略刚采的一批数据；本例是 **Double DQN** 路线，replay 是标配。）

#### 本例怎么存、怎么抽

上式里的 `(states, actions, rewards, next_states, dones)` **不是刚走的那一步**，而是从 **Replay Buffer（经验池）** 里抽出来的。

每走一步 `push` 一条 `(s, a, r, s', done)`（本例 `r` 是塑形后的 `train_r`，见 §3.3）。池子是定长队列（`deque`，满了丢最旧的）。至少 **`2000`** 条才开始 `train_step`；之后 **每个 env 步** 仍调用一次更新逻辑，内部 `sample(128)` 抽一批算损失。实现见 `dqn_train.py` 的 `ReplayBuffer` 与 `_run_episode`。

此外还有 DQN 标配：

| 机制 | 本例取值 | 作用 |
|------|----------|------|
| Replay Buffer | 容量 `50000`，至少 `2000` 条后开始学 | 存 transition，随机重放，打破样本相关性 |
| Batch | `128` | 每次从 buffer 抽这么多条一起更新（Adam） |
| Target 软更新 | `τ = 0.005` | 每次成功的 `train_step`（buffer≥2000 后约每个 env 步一次）软更新 `target ← τ·policy + (1-τ)·target` |
| 梯度裁剪 | `10.0` | 防爆炸 |
| Optimizer | Adam, `lr=5e-4` | |
| 设备 | **CPU only** | 训练脚本没有 `.cuda()`；checkpoint 也 `map_location="cpu"` |

### 3.3 距离塑形（Potential-based shaping）

稀疏奖励（几乎全是 `-0.01`，终点才 `+1`）在随机大图上很难探索。训练时对 **存入 replay 的 reward** 加塑形项（**不改变环境真实回报**，日志里的 `train_return` 仍是 base reward 之和）：

```text
Φ(s) = -manhattan(s, G) / max_dist        max_dist = 2×(size-1) = 8
F(s, s') = γ·Φ(s') - Φ(s)
train_r = base_reward + SHAPE_COEF × F     SHAPE_COEF = 0.1
代码：若 done（terminated 或 truncated），Φ(s') 置 0
```

靠近终点 → `Φ` 变大（负得少）→ 塑形项为正，鼓励缩短路径。到达 G 时曼哈顿已是 0，`Φ(s')=0` 与理论一致。**超时截断**时人还在远处，代码仍把 `Φ(s')` 当成 0（和终点一样），会给一步虚假正塑形；potential-based「不改变最优策略」只对 **到达终点、折扣一致** 成立，不宜直接套到 truncation。

**为什么要塑形：** 随机换图 + 绕墙时，若只靠「到 G 才 +1」，多数 transition 的 TD 信号几乎一样（全是小负步长），网络很难从探索里分辨「哪一步更好」，学习会很慢。**塑形只改 replay 里用于更新的 `train_r`**，给「是否更接近 G」一条稠密反馈，主要 **缩短探索期**；环境 `step()` 与评估时的回报仍是 base reward（见 `shaped_reward()` in `dqn_train.py`）。曼哈顿距离 **看不见墙**，直线近未必能走通，故 `SHAPE_COEF=0.1` 较小，绕路仍靠 obs 里的障碍与终点 `+1` 学（见下文 FAQ）。

### 3.4 课程学习

障碍数由易到难，避免一上来 3 墙就探索不动：

| Episode 范围 | 每局障碍数 `n_obstacles` |
|--------------|--------------------------|
| `1 … 2000` | `1` |
| `2001 … 4000` | `2` |
| `4001 … 6000` | `3` |

评估始终用 **最难设置：3 障碍 + 随机起点**，方便看「真正任务」上的进度。因此前 2000 局课程还在 1 墙时，eval `success` 低是正常的。

### 3.5 ε-greedy 探索

本例 ε 按 **梯度步数**（每从 buffer 抽 batch 更新一次算一步）**指数衰减**，不是按 episode：

```text
ε(t) = 0.02 + (1.0 − 0.02) · exp(−t / 25000)
```

`25000` 是时间常数，不是「走到第 25000 步就变成 0.02」：

| 梯度步 `t` | ε（约） |
|------------|---------|
| `0` | `1.00` |
| `25000` | `0.38` |
| `50000` | `0.15` |
| 更久 | 渐近到 `0.02` |

buffer 未满 `2000` 条时还不更新，ε 一直停在 `1.0`。选动作时始终在 `legal_actions()` 返回的集合里探索 / argmax。

### 3.6 微调阶段（`--finetune` / 训练结束自动跑）

主训练 6000 局结束后，`dqn_train.py` 会 **自动** 再跑一轮 hard-only 微调（也可单独 `python dqn_train.py --finetune`）。**没有命令行开关跳过**，只能改代码。

- 全程 **3 障碍**
- 学习率降到 `1e-4`
- ε **重设** 为 `0.15`，按时间常数 `12000` 降到 `0.01`（不接着主训练末尾已接近 `0.02` 的 ε）
- `SEED=1`（主训练是 `SEED=0`），`step_count` 归零
- 先在固定评估流上打 **baseline**，再用贪心 **探测 300 局**，收集仍失败的 `(obstacles, start)` 对
- 之后每局以概率 `35%` **重放这些 hard case**，否则随机新图
- 每 1000 局刷新 hard pool（同样 300 局探测）
- 最多 `4000` 局
- 只在评估分 **严格高于** 当前 best 时覆盖 `dqn_random_layout.pth`（主训练打平也会存）
- 结束时 **覆盖** `dqn_reward_history.png`（即使权重没变得更好）

目的：把课程末期仍卡住的布局 **针对性补练**。这里的 **hard-replay** 是「再玩一遍失败地图」，和 §3.2 的 **experience replay**（从 buffer 随机抽 transition）不是同一件事：前者决定 **下一局环境怎么 reset**，后者决定 **梯度用哪些历史 `(s,a,r,s')` 来算**。微调阶段里，hard 局与随机新图走出的步子都 `push` 进 **同一个** `agent.memory`（没有为 hard case 单独再建一个池）；但 `finetune()` 会 **新建** `DQNAgent`，经验池从空重新攒，**不会**接着主训练 6000 局里已经存满的那 5 万条（只加载 `dqn_random_layout.pth` 权重）。buffer 再满 `2000` 条之前不更新，这段时间 ε 停在 `0.15`。

---

## 4. 训练循环在干什么（`dqn_train.py`）

```text
初始化 QNet（policy + target）、ReplayBuffer、Adam
for episode = 1 .. 6000:
    n_obs = curriculum_obstacles(episode)   # 1 → 2 → 3
    reset(随机布局, n_obs, random_start=True)
    while 未结束:
        在 legal_actions 上做 ε-greedy 选 a
        env.step(a) → base_reward
        train_r = shaped_reward(base_reward, 曼哈顿距离变化)
        memory.push(s, a, train_r, s', done)
        若 buffer 足够 → sample batch → Double DQN 更新 + 软更新 target + 衰减 ε
    每隔 200 局：
        固定 EVAL_SEED 上跑 300 张 3-障碍图（greedy + avoid_revisit）
        打印 success / mean_return
        分数 = success_rate + 0.001 × mean_return；≥ 当前 best 则保存（打平也覆盖）
保存 dqn_reward_history.png          ← 课程曲线，随即被微调覆盖
自动进入 finetune()（hard-replay；曲线图写成微调阶段；权重仅严格更好才覆盖）
```

### 4.1 单局何时结束

| 条件 | 结果 |
|------|------|
| 踩到 `G` | `terminated=True` |
| 走了 **50** 步仍未到达 | `truncated=True` |

`done = terminated or truncated`；TD 目标里 **两种 done 都不 bootstrap**（与表格 Q 例相同，超时严格说也可 bootstrap，本例地图小、影响有限）。

### 4.2 整个训练何时停止

| 阶段 | 局数 | 说明 |
|------|------|------|
| 课程主训练 | `6000` | 跑满即进入微调；`SEED=0` |
| Hard 微调 | `4000` | 主训练后**自动**执行（无 CLI 跳过）；或 `--finetune` 单独跑；`SEED=1` |

没有 early stopping。全程 **CPU**。看 `success` 和曲线判断是否已够用；完整跑完后曲线图是微调阶段的。

### 4.3 超参数速查

| 符号 / 常量 | 值 | 含义 |
|---------------|-----|------|
| `GAMMA` | `0.99` | 折扣 |
| `LR` | `5e-4` | 主训练学习率 |
| `BATCH_SIZE` | `128` | |
| `MEMORY_SIZE` | `50000` | |
| `MIN_MEMORY_SIZE` | `2000` | 开始更新前最少样本 |
| `TAU` | `0.005` | target 软更新系数 |
| `EPS_START / END` | `1.0 / 0.02` | |
| `EPS_DECAY_STEPS` | `25000` | ε 指数衰减的时间常数（不是线性降到 `0.02` 的步数） |
| `EVAL_EVERY` | `200` | |
| `EVAL_LAYOUTS` | `300` | 训练内评估图数 |
| `EVAL_SEED` | `12345` | **固定**评估流，success% 可跨 checkpoint 对比 |
| `SHAPE_COEF` | `0.1` | 距离塑形强度 |
| `SEED` | `0` | 主训练随机种子；微调用 `SEED+1` |
| `FINETUNE_LR` | `1e-4` | 微调学习率 |
| `FINETUNE_EPS_START / END` | `0.15 / 0.01` | 微调重新设 ε |
| `FINETUNE_EPS_DECAY` | `12000` | 微调 ε 衰减时间常数 |
| `HARD_MIX` | `0.35` | 微调局中重放 hard case 的概率 |
| `GRAD_CLIP` | `10.0` | |

---

## 5. 如何读训练 / 测试结果

### 5.1 训练日志 vs 最终测试

两套评估 **目的不同**：

| | 训练中 `evaluate()` | `dqn_test.py` |
|--|---------------------|---------------|
| 种子 | **固定** `EVAL_SEED=12345` | 每次运行 **随机抽 5 个种子** |
| 图数 | 300 张 / 次 | 5 × 200 = **1000** 局 |
| 用途 | 看收敛、少抖动、存 best 模型 | 证明 **换种子换墙** 仍能到 |
| 动作 | greedy + legal + `avoid_revisit` | 同左 |

训练里 `success=300/300` 只说明「在这 300 张固定考题上全过」，不代表万能；`dqn_test.py` 的 `overall success` + `mean ± std` 才是泛化成绩。

### 5.2 控制台示例

训练一段时间后常见：

```text
episode= 400  n_obs=1  eps=0.842  train_return=-0.320
  eval_mean=0.450  success=85% (255/300)  min=-0.500
  saved dqn_random_layout.pth (success=85%, mean=0.450)
```

- `train_return`：本局 **环境原始回报** 之和（含探索乱走、重访），会抖。带掩码时几乎不含撞墙。
- `eval_mean` / `success`：固定 300 张 **3 障碍** 图上的贪心成绩，更可信。前 2000 局课程还是 1 墙，这条 eval 仍考 3 墙。
- `n_obs`：当前课程阶段的障碍数。
- 存盘看 `success_rate + 0.001 * mean_return`，所以 success 相同、mean 略高也会覆盖。

### 5.3 `dqn_reward_history.png`

`_save_curve` 始终写到这一个文件名。`python dqn_train.py` 会先存课程曲线，**微调结束再覆盖**，所以完整跑完后图上是 **hard-replay 微调的 4000 局**，不是课程 6000 局。

- 蓝线：训练回报滑动平均，窗口 **50**（`MA50`；仍有 ε 探索，会抖）。
- 橙线：每隔 200 局的 **eval mean**（3 障碍固定种子，点+连线），看真实任务进度。

---

## 6. 测试脚本（`dqn_test.py`）

```bash
python dqn_test.py
```

**不再训练、不再探索**。加载 `dqn_random_layout.pth`，全程 greedy + 动作掩码 + `avoid_revisit`。

`main()` 顺序：

```text
1) 加载 QNet 权重
2) 随机抽 N_SEEDS=5 个种子
3) 用 seeds[0] 跑 N_DEMOS=5 局，打印地图（给人看路）
4) 对每个 seed 跑 LAYOUTS_PER_SEED=200 局，汇总 success / mean_return
5) 打印 overall + per-seed mean±std + worst/best
6) 若存在 ../grid-world-qlearning/q_table.npy → 同一批种子上跑表格 Q 对照
```

地图上的 `S` **永远标左上 `(0,0)`**，不一定是本局起点；本局起点看打印的 `start=`。`A` 才是 Agent。

### 6.1 为什么多种子

固定一套考题容易 **过拟合评估集**。随机种子意味着 **布局流也换**，`worst seed` 暴露最差情况；这比「同一张图跑 5 次」更有说服力。

### 6.2 表格 Q 对照

若已训练过 qlearning，测试脚本会用 **相同种子、相同 200 局/种子** 跑表格贪心（只看格子 id，**不看墙通道**，也 **不用** 合法动作掩码 / `avoid_revisit`）。布局一变，表格 Q 成功率通常会 **明显低于 DQN**——直观说明「状态必须包含墙信息才能泛化」。

### 6.3 排查失败（`inspect_fails.py`）

固定 `SEED=42`，跑 `N_LAYOUTS=200`，把 **未到终点** 的局打印出来：起点、障碍、步数、撞墙次数、路径头尾、渲染地图。改文件顶部的 `SEED` / `N_LAYOUTS` 可复现其它失败集。

典型失败模式：

- 步数打满 50（绕圈或走太远）
- `max_visit` 很高（在少数格子间来回）
- `bumps` 多：对 **带掩码的 DQN** 几乎不应出现（静态图上合法走法围不死）；对照表格 Q 才会经常撞墙。若 DQN 失败局里 bumps>0，优先怀疑掩码没生效或打印的是别的策略。

---

## 7. 和表格 Q-learning / CartPole 的关系

```text
grid-world-qlearning     本例 grid-world-dqn          CartPole Double DQN
────────────────────     ───────────────────          ───────────────────
表格 Q(s,a)              CNN 近似 Q(s,a)              MLP 近似 Q(s,a)
状态=格子 id             状态=4 通道网格              状态=4 维连续向量
固定地图                 随机地图 + 课程                固定动力学
无 replay                Replay + target              同左
无塑形、无重访惩罚        曼哈顿塑形（只进 replay）     通常无塑形
                         + 环境重访惩罚（评估也算）
```

建议认知顺序：

1. [`../grid-world-qlearning/`](../grid-world-qlearning/)：看清 Q、TD、ε-greedy、回报回传
2. **本例**：状态变「看图」、Q 变网络、加 replay / Double / 课程 / 塑形
3. [`../../cliff-walking/cliff-walking-q-sarsa/`](../../cliff-walking/cliff-walking-q-sarsa/)：离策略 Q-learning vs 在策略 SARSA
4. [`../../cart-pole/cart-pole-dqn/`](../../cart-pole/cart-pole-dqn/)：连续 4 维状态、MLP Double DQN

---

## 8. 常见问题

**Q: 学习是在学什么？和 qlearning 目录有何本质区别？**  
A: 仍是在学 **策略意义上的 Q 值**（每个动作好坏），但 Q 由 **CNN 权重** 表示，输入必须含 **障碍 + 终点 + 自身位置 + 访问历史**。qlearning 目录背的是 **一张固定地图**；这里要 **换墙仍能找到 G**。

**Q: 训练里的 `success` 和 `dqn_test.py` 的 `success` 为什么可能差很多？**  
A: 训练评估用 **固定** `EVAL_SEED` 的 300 张图，方便对比 checkpoint；测试用 **新随机种子** × 更多图。测试分数通常更严、更代表泛化。

**Q: 为什么需要 `visited` 通道和 `avoid_revisit`？**  
A: 纯贪心在某些图上会在两格间 **无限来回**，直到 50 步截断。访问信息让网络知道「来过了」；测试时再优先走未访问格，打破循环。带动作掩码时几乎不会原地撞墙，循环主要是 A↔B。

**Q: 塑形会不会让 Agent 只追曼哈顿距离、忽略绕墙？**  
A: 塑形系数 `0.1` 较小；主信号仍是到 G 的 `+1`。它主要 **缩短探索期**，不是替代终点奖励。超时截断时代码把 `Φ(s')` 置 0，理论上会给虚假正塑形，本例地图小、影响有限。

**Q: 跑 `dqn_train.py` 为什么要等很久？**  
A: 默认 `6000 + 4000` 局，且 buffer 满后每步做梯度更新；脚本跑在 **CPU** 上（没有用 GPU）。CNN + replay 比表格 Q 慢一个数量级是正常的。可先改小 `MAX_EPISODES` / `FINETUNE_EPISODES` 做 smoke test。没有 CLI 跳过自动微调。

**Q: 完整训练后曲线图怎么不像课程 6000 局？**  
A: 微调结束会 **覆盖** 同一张 `dqn_reward_history.png`。权重只在微调评估严格更好时才覆盖；图总会被写成微调曲线。

**Q: 加载 checkpoint 报错 incompatible？**  
A: 可能是旧版 **3 通道** 权重；需重新 `python dqn_train.py` 训练 **4 通道** 网络。`dqn_test.py` / `inspect_fails.py` 没有微调脚本那么友好的提示。

**Q: 表格 Q 在随机图上完全没用吗？**  
A: 不是完全零分——它仍会在 **碰巧和固定地图相似** 的布局上走几步——但 **没有墙信息**，无法系统性绕新障碍；`dqn_test.py` 的对照会量化这一点。

**Q: 地图上的 `S` 不是起点？**  
A: `S` 固定画在 `(0,0)`。随机起点时看日志里的 `start=`，`A` 才是 Agent。

---

## 9. 下一步可扩展

- 更大 `size` / 更多 `n_obstacles`；加深 CNN 或加 dueling head
- 对比 **无课程 / 无塑形 / 无 visited 通道** 的 ablation
- 连续控制：`MountainCarContinuous` 等 Gymnasium 环境
- 把 **动作掩码 + avoid_revisit** 迁到其它网格决策任务
