# GridWorld + Double DQN（随机布局）

本目录用 **Double DQN** 训练 **每局障碍都可能变化** 的 5×5 GridWorld。  
和同级 `grid-world-qlearning` 的差别：那里背熟一张固定地图；这里必须**看见墙**才能换图泛化。

| 文件 | 作用 |
|------|------|
| `grid_world_env.py` | 随机可解布局；观察为 3 通道拼成的向量 |
| `dqn_train.py` | 经验回放 + Double DQN + 软更新 |
| `dqn_test.py` | 新随机地图上测 DQN；可选对照表格 Q |
| `dqn_random_layout.pth` | 最佳策略网络 |
| `dqn_reward_history.png` | 训练曲线 |

```bash
cd experimental-rl/grid-world/grid-world-dqn
python dqn_train.py
python dqn_test.py
```

依赖：`gymnasium`、`numpy`、`matplotlib`、`torch`。

建议先跑通 `../grid-world-qlearning/`。

---

## 1. 在学什么

- 每局采样 `n_obstacles`（默认 3）个障碍，BFS 保证 `S→G` 连通
- 起点也可随机
- Agent 输入整张图，输出上下左右的 Q 值，尽快到 `G`
- 奖励：每步 `-0.01`，到终点 `+1`

观察（`3 × 5 × 5` 拉平为 75 维）：

| 通道 | 含义 |
|------|------|
| agent | 当前位置为 1 |
| obstacle | 墙为 1 |
| goal | 终点为 1 |

为什么不能继续用表格 `Q[格子]`：同一格在不同地图上最优动作不同，状态里必须带上墙的信息。

---

## 2. 算法

与 CartPole Double DQN 同构：

1. Replay Buffer  
2. policy_net 选 `a*`，target_net 估 `Q(s', a*)`  
3. 目标网络软更新 `τ`

选动作仍是 ε-greedy；评估时纯贪心。

---

## 3. 测试在证明什么

`dqn_test.py`：

1. 在一批**全新随机地图**上跑 DQN，看成功率  
2. 若存在 `../grid-world-qlearning/q_table.npy`，用「只看坐标」的表格策略在同样风格的随机图上对照——通常明显更差  

说服力来自「换墙还能到」，不是重复同一固定轨迹。

---

## 4. 下一步

- 把扁平 MLP 换成小型 CNN（输入 `3×H×W`）
- 增大 `size` / `n_obstacles`
- 再去做 CliffWalking 或连续控制（MountainCarContinuous）
