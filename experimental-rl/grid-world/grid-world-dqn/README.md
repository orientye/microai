# GridWorld + CNN Double DQN（随机布局）

本目录用 **CNN + Double DQN** 训练 **每局障碍都可能变化** 的 5×5 GridWorld。  
和同级 `grid-world-qlearning` 的差别：那里背熟一张固定地图；这里必须**看见墙**才能换图泛化。

| 文件 | 作用 |
|------|------|
| `grid_world_env.py` | 随机可解布局；观察为 **4** 通道拼成的向量 |
| `dqn_train.py` | CNN + 课程学习 + 塑形 + 动作掩码；结束可自动微调 |
| `dqn_test.py` | **随机多种子**评估；可选对照表格 Q |
| `inspect_fails.py` | 打印失败局的路径（排查卡死） |
| `dqn_random_layout.pth` | 最佳策略网络 |
| `dqn_reward_history.png` | 训练曲线 |

```bash
cd experimental-rl/grid-world/grid-world-dqn
python dqn_train.py              # 课程学习（结束会自动微调）
python dqn_train.py --finetune   # 仅微调
python dqn_test.py               # 最终成绩：随机 5 种子 × 每种子 200 张图
python inspect_fails.py          # 排查失败路径（可自改 SEED）
```

- 训练中的 `success=`：固定评估种子 + 300 张图，方便看收敛、少抖动  
- `dqn_test.py`：每次随机抽多个种子，报告 overall / mean±std / worst，用来证明泛化  

依赖：`gymnasium`、`numpy`、`matplotlib`、`torch`。

建议先跑通 `../grid-world-qlearning/`。

---

## 1. 在学什么

- 每局采样 `n_obstacles`（默认 3）个障碍，BFS 保证 `S→G` 连通  
- 随机起点只从「能到 G」的格子采样  
- Agent 输入整张图，输出上下左右的 Q 值，尽快到 `G`  

奖励（环境原始回报）：

| 情况 | 奖励 |
|------|------|
| 到终点 | `+1` |
| 普通一步 | `-0.01` |
| 撞墙 / 出界 | `-0.05` |
| 重访格子 | 额外负分（随访问次数加重） |

观察（`4 × 5 × 5`，训练时 reshape 成卷积输入）：

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
7. **撞墙 / 重访惩罚**；可达起点过滤  
8. **visited 通道** + 失败布局回放  
9. **动作掩码**：有可走格时禁止撞墙/出界；评估/测试还可 `avoid_revisit`（优先未访问格）  

训练选动作：ε-greedy + 合法动作掩码。  
评估 / 测试：纯贪心 + 合法掩码 + 优先未访问。

---

## 3. 测试在证明什么

`dqn_test.py`：

1. 默认 **5 个随机种子 × 每种子 200 张图**（共 1000 局）  
2. 报告 overall 成功率、多种子 mean±std、worst/best  
3. 若存在 `../grid-world-qlearning/q_table.npy`，用同一批种子对照表格 Q  

说服力来自「换种子换墙还能到」，不是背熟某一套固定考题。

---

## 4. 下一步可扩展

- 换成 Gymnasium `CliffWalking-v0`，对比 Q-learning vs SARSA  
- 更大 `size` / 更多障碍；CNN 可再加深  
- 连续控制：`MountainCarContinuous`  
- 把动作掩码 / visited 思路迁到别的网格决策任务  
