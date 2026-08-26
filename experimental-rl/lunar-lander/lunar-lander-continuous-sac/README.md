# LunarLanderContinuous + Soft Actor-Critic (SAC)

Gymnasium **`LunarLanderContinuous-v3`**：同一着陆任务，动作用 **2 维连续推力**。  
相对同目录离散 PPO，这里是 **off-policy + 熵正则** 的连续控制路线。

| 文件 | 作用 |
|------|------|
| `sac_train.py` | 双 Q + Target 软更新 + Replay + tanh 重参数化 + 自动 α |
| `sac_test.py` | 确定性均值动作评估 + `human` 渲染 |

```bash
cd experimental-rl/lunar-lander/lunar-lander-continuous-sac
python sac_train.py
python sac_test.py
```

依赖：`gymnasium[box2d]`（需 `Box2D`）、`torch`、`numpy`、`matplotlib`；动画需 `pygame`。

---

## 1. 环境在学什么

https://gymnasium.farama.org/environments/box2d/lunar_lander/

| 概念 | 取值 |
|------|------|
| 观察 | 8 维：坐标、线速度、角度、角速度、左右腿触地（与离散版相同） |
| 动作 | **2 维连续** ∈ `[-1, 1]²`：主引擎油门 + 侧喷 |
| 奖励 | 靠近平台、减速、腿着地加分；摔毁 / 飞出扣分；喷火有小代价 |
| 截断 | 默认最多约 1000 步 |

主引擎：`a[0] ≤ 0` 熄火，`> 0` 按幅度点火。  
侧喷：`a[1]` 偏负喷左、偏正喷右，中间一段死区。

验收：确定性评估回报约 **≥ 200**（与离散版同一口径）。

---

## 2. SAC 在学什么

与 [`../../pendulum/pendulum-sac/`](../../pendulum/pendulum-sac/) 同一套：

| 组件 | 作用 |
|------|------|
| Actor | 高斯 → `tanh` 压到 `[-1,1]²`；重参数化采样 |
| 双 Q | \(Q_1, Q_2\)；target 取 **min** |
| Target 网 | 软更新 `τ` |
| Replay | 经验反复用 |
| 自动 α | 学 `log α`，熵目标 `-|A|`（此处 `|A|=2`） |

有 CUDA 时自动用 GPU（本课步数比 Pendulum 多一个数量级）。

Bootstrap：`terminated`（着陆成功 / 摔毁）才当真正结束；`truncated` 不当 terminal。

相对 Pendulum：动作从 1 维变成 2 维；步数预算更长（`MAX_STEPS=300000`）。  
相对 MountainCar：奖励不是欺骗性的「少出力≈0」，**不做**势能 shaping。

---

## 3. 和离散 PPO 差在哪

| | `lunar-lander-ppo` | 本目录 |
|--|--------------------|--------|
| 动作 | 4 离散开关 | 2 维连续油门 |
| 数据 | on-policy rollout | off-policy replay |
| Critic | V(s) + GAE | 双 Q(s,a) |
| 探索 | 采样 + 熵系数 | **熵进目标** + 自动温度 |
| 验收 | ≥ 200 | ≥ 200（同一任务） |

---

## 4. 验收

- 日志里 `train_avg20` / `eval_mean` 抬升  
- 达到约 **≥ 200** 会打印 threshold 并提前结束；最佳 Actor 写入 `sac_lunar_lander.pth`  
- `sac_test.py` 窗口里着陆器应能大致落到两旗之间  

---

## 5. 非目标

- 不做观测归一化、多环境并行、优先经验回放  
- 不做与离散 PPO 的严格同 seed 公平对比脚本  
- 不做 TD3 / DDPG  
