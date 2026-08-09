# GridWorld + CNN Double DQN（随机布局）

本目录用 **CNN + Double DQN** 训练 **每局障碍都可能变化** 的 5×5 GridWorld。  
和同级 `grid-world-qlearning` 的差别：那里背熟一张固定地图；这里必须**看见墙**才能换图泛化。

| 文件 | 作用 |
|------|------|
| `grid_world_env.py` | 随机可解布局；观察为 3 通道拼成的向量 |
| `dqn_train.py` | CNN Q 网络 + 经验回放 + Double DQN + 软更新 |
| `dqn_test.py` | 新随机地图上测 DQN；可选对照表格 Q |
| `dqn_random_layout.pth` | 最佳策略网络 |
| `dqn_reward_history.png` | 训练曲线 |

```bash
cd experimental-rl/grid-world/grid-world-dqn
python dqn_train.py              # 课程学习
python dqn_train.py --finetune   # 从最佳权重做 3 障碍微调
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

观察（`4 × 5 × 5`，训练时会 reshape 回卷积输入）：

| 通道 | 含义 |
|------|------|
| agent | 当前位置为 1 |
| obstacle | 墙为 1 |
| goal | 终点为 1 |
| visited | 本局访问次数（归一化），用来打破 A↔B / 原地撞墙死循环 |

为什么不能继续用表格 `Q[格子]`：同一格在不同地图上最优动作不同，状态里必须带上墙的信息。  
为什么不用扁平 MLP：网格是空间结构；CNN 更容易学「绕墙」「朝终点」这类局部模式。

---

## 2. 算法与提分点

与 CartPole Double DQN 同构，并针对随机布局做了加强：

1. Replay Buffer（更大）
2. policy_net 选 `a*`，target_net 估 `Q(s', a*)`
3. 目标网络软更新 `τ`
4. **CNN** 替代扁平 MLP
5. **课程学习**：障碍数 1 → 2 → 3
6. **距离塑形**：`Φ = -manhattan`，加速靠近终点
7. **撞墙 / 重访惩罚**；随机起点只从「能到 G」的格子采样
8. **visited 通道** + 失败布局回放，专门打掉 A↔B / 原地撞墙死循环
9. **动作掩码**：有可走格时禁止选择撞墙/出界动作（测试与训练选动作都用）

选动作仍是 ε-greedy；评估 / 测试时纯贪心，且固定 3 障碍。

---

## 3. 测试在证明什么

`dqn_test.py`：

1. 在一批**全新随机地图**上跑 DQN，看成功率（默认 100 张）
2. 若存在 `../grid-world-qlearning/q_table.npy`，用「只看坐标」的表格策略对照

说服力来自「换墙还能到」，不是重复同一固定轨迹。

---

## 4. 若还想更高

- 课程学习：先 1 个障碍，再 2，再 3
- 奖励塑形：按到终点的曼哈顿距离给小负分（注意别引入捷径）
- 更大地图 / 更多障碍时，继续加深 CNN
- 再去做 CliffWalking 或连续控制（MountainCarContinuous）
