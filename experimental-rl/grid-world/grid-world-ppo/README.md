# GridWorld + CNN PPO（随机布局）

本目录用 **CNN Actor-Critic + PPO** 训练 **每局障碍都可能变化** 的 5×5 GridWorld。  
环境和观察与同级 `grid-world-dqn` 相同；差别是学 **策略 π(a|s)**，而不是 Q 值。

| 文件 | 作用 |
|------|------|
| `grid_world_env.py` | 随机可解布局；观察为 **4** 通道拼成的向量（与 DQN 一致） |
| `ppo_train.py` | CNN Actor-Critic + GAE + clip；课程学习 + 塑形 + 动作掩码 |
| `ppo_test.py` | **随机多种子**评估；可选对照表格 Q |
| `inspect_fails.py` | 打印失败局的路径（排查卡死） |
| `ppo_random_layout.pth` | 最佳策略网络 |
| `ppo_reward_history.png` | 训练曲线 |

```bash
cd experimental-rl/grid-world/grid-world-ppo
python ppo_train.py
python ppo_test.py               # 最终成绩：随机 5 种子 × 每种子 200 张图
python inspect_fails.py          # 排查失败路径（可自改 SEED）
```

- 训练中的 `success=`：固定评估种子 + 300 张图，方便看收敛  
- `ppo_test.py`：每次随机抽多个种子，报告 overall / mean±std / worst  

依赖：`gymnasium`、`numpy`、`matplotlib`、`torch`。

建议顺序：`qlearning` → `dqn` → 本目录。

---

## 1. 在学什么

环境、奖励、观察与 DQN 版相同：

| 情况 | 奖励 |
|------|------|
| 到终点 | `+1` |
| 普通一步 | `-0.01` |
| 撞墙 / 出界 | `-0.05` |
| 重访格子 | 额外负分 |

观察：`4 × 5 × 5`（agent / obstacle / goal / visited）。

网络输出两路：

- **Actor**：各动作 logits → 掩码后 Categorical 采样  
- **Critic**：状态价值 `V(s)`，供 GAE 算 advantage  

---

## 2. 算法要点（相对 DQN）

| | DQN | PPO |
|--|-----|-----|
| 学什么 | Q(s,a) | π(a\|s) + V(s) |
| 数据 | off-policy replay | on-policy rollout |
| 探索 | ε-greedy | 策略熵 |
| 更新 | TD + target net | clip 比率 + 多 epoch |

本实现保留随机布局上的提分手段：

1. **CNN** 共享骨干  
2. **GAE**（λ=0.95）  
3. **PPO clip**（ε=0.2）  
4. **课程学习**：障碍 1 → 2 → 3（按 update）  
5. **距离塑形**  
6. **动作掩码**（训练采样 / 评估贪心）  
7. 评估 / 测试：`argmax` + `avoid_revisit`  

训练单位是 **update**（每次约 2048 步 on-policy 数据），不是 DQN 那种逐步 `train_step`。

---

## 3. 测试在证明什么

与 `dqn_test.py` 同构：换种子换墙仍能到 `G`，说明学的是「看见墙再绕」，不是背一张图。

---

## 4. 和 DQN 怎么比

- 同一环境可直接对成功率；PPO 通常更稳但样本效率往往不如 replay DQN  
- 若以后上连续动作 / 多维控制，PPO 更自然；纯离散小格子，DQN 往往够用且更快  
