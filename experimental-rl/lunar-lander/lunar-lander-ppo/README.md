# LunarLander + 离散 PPO

Gymnasium **`LunarLander-v3`**：着陆器点火降落到平地旗标之间。  
相对 CartPole，状态更丰富、奖励有多项加减，失败模式（摔毁 / 飞出）更直观。

| 文件 | 作用 |
|------|------|
| `ppo_train.py` | 离散 Actor-Critic + GAE + PPO clip（minibatch） |
| `ppo_test.py` | 贪心 `argmax` 评估 + `human` 渲染 |

```bash
cd experimental-rl/lunar-lander/lunar-lander-ppo
python ppo_train.py
python ppo_test.py
```

依赖：`gymnasium[box2d]`（需 `Box2D`）、`torch`、`numpy`、`matplotlib`；动画需 `pygame`。

---

## 1. 环境在学什么

https://gymnasium.farama.org/environments/box2d/lunar_lander/

| 概念 | 取值 |
|------|------|
| 观察 | 8 维：坐标、线速度、角度、角速度、左右腿触地 |
| 动作 | **4** 离散：无操作 / 左喷管 / 主引擎 / 右喷管 |
| 奖励 | 靠近平台、减速、腿着地加分；摔毁 / 飞出扣分；主引擎与侧喷有小代价 |
| 截断 | 默认最多约 1000 步 |

验收：确定性评估回报约 **≥ 200**（Gymnasium 常见 solved 标准）。

---

## 2. 和 CartPole PPO 差在哪

| | `cart-pole-ppo` | 本目录 |
|--|-----------------|--------|
| 任务 | 立杆不倒 | **点火降落** |
| 状态 / 动作 | 4 维 / 2 | 8 维 / 4 |
| 网络 | 64×64 | **128×128** |
| Rollout | 128 | **2048** + minibatch |
| 验收 | ≥ 450 | ≥ 200 |

算法骨架对齐：Categorical 策略、GAE(λ)、ratio clip、value MSE、entropy bonus。

---

## 3. 验收

- 日志里 `train_avg20` / `eval_mean` 抬升  
- 达到约 **≥ 200** 会打印 threshold 并提前结束；最佳权重写入 `ppo_lunar_lander.pth`  
- `ppo_test.py` 窗口里着陆器应能大致落到两旗之间（允许轻微弹跳）

---

## 4. 非目标

- 不做 Continuous 版（可作下一课 `LunarLanderContinuous` + SAC）  
- 不做观测归一化、多环境并行、优先经验回放  
