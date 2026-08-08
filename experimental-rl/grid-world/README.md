# GridWorld + 表格 Q-learning

本目录用 **表格 Q-learning** 训练自定义的 **5×5 GridWorld**，并可视化贪心策略。

| 文件 | 作用 |
|------|------|
| `grid_world_env.py` | 自定义 Gymnasium 环境 |
| `q_learning_train.py` | 训练：ε-greedy + TD 更新，保存 Q 表与曲线 |
| `q_learning_test.py` | 加载 `q_table.npy`，打印路径与策略图 |
| `q_table.npy` | 训练得到的 Q 表（运行训练后生成） |
| `reward_history.png` | 训练回报曲线 |
| `policy_map.png` | 状态价值热力图 + 贪心箭头 |

```bash
# 在本目录下运行
python q_learning_train.py
python q_learning_test.py
```

依赖：`gymnasium`、`numpy`、`matplotlib`（不需要 torch）。

---

## 1. 问题：GridWorld 在学什么

Agent 从左上角 `S` 走到右下角 `G`，中间有障碍 `#`。目标是尽快到达终点。

```text
S . . # .
. . . # .
. . . . .
. # . . .
. . . . G
```

| 概念 | 本例取值 |
|------|----------|
| 状态 `s` | 格子坐标，编码为 `0..24` 的离散 id |
| 动作 `a` | 4 个：`0`↑ `1`→ `2`↓ `3`← |
| 奖励 `r` | 每步 `-0.01`；到达终点 `+1` |
| `terminated` | 到达 `G` |
| `truncated` | 超过 `100` 步 |
| 撞墙/障碍 | 位置不变，仍扣逐步惩罚 |

和 CartPole 的差别：状态完全离散、可查表；策略可以画成「每格一个箭头」，学习过程更直观。

---

## 2. 算法：表格 Q-learning

更新公式：

```text
Q(s,a) ← Q(s,a) + α [ r + γ max_{a'} Q(s', a') − Q(s,a) ]
```

其中：

- `α = 0.1`：学习率
- `γ = 0.99`：折扣
- 动作选择：ε-greedy，ε 从 `1.0` 线性降到 `0.05`
- 终止态：`max Q(s',·)` 不再 bootstrap

这是**离策略**值迭代：行为策略在探索，但更新目标按 `max` 估计最优动作价值。

---

## 3. 预期现象

训练几千局后，贪心策略应绕开障碍、大致沿最短可行路径走向 `G`。  
评估回报接近 `1 - 0.01 × 最短步数`（例如约 8–12 步时，回报约 `0.88–0.92`）。

若策略箭头乱指或评估回报接近 `0`，常见原因：

- 探索衰减太快 / 太慢
- `α` 过大导致震荡
- 训练局数不够

---

## 4. 下一步可扩展

- 换成 Gymnasium 自带的 `CliffWalking-v0`，对比 **Q-learning vs SARSA**
- 去掉障碍，改成纯最短路，核对是否等于曼哈顿距离
- 用神经网络近似 Q（小规模 DQN），对照表格法
