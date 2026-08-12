# MountainCarContinuous + Soft Actor-Critic (SAC)

Gymnasium **`MountainCarContinuous-v0`**：连续推力把小车推上右侧山顶。  
相对 Pendulum，这里成功信号更稀：多数步只有动作代价，**到顶才有大奖励**。

| 文件 | 作用 |
|------|------|
| `sac_train.py` | 双 Q + Target 软更新 + Replay + tanh 重参数化 + 自动 α |
| `sac_test.py` | 确定性均值动作评估 + `human` 渲染 |

```bash
cd experimental-rl/mountain-car-continuous/mountain-car-continuous-sac
python sac_train.py
python sac_test.py
```

依赖：`gymnasium`、`torch`、`numpy`、`matplotlib`；动画需 `pygame`。

---

## 1. 环境在学什么

| 概念 | 取值 |
|------|------|
| 观察 | `[position, velocity]`，2 维连续 |
| 动作 | 力 ∈ `[-1, 1]`，**1 维连续** |
| 奖励 | 每步约 `-0.1·a²`；到达目标位置（≥ 0.45）时 **+100** |
| 截断 | 默认最多 999 步 |

目标不是「稳住倒立」，而是学会**左右蓄力**攒动量再冲顶。  
验收：确定性评估（**原始**环境回报）约 **≥ 90**。

---

## 2. 欺骗性奖励（为什么纯 SAC 会趴窝）

环境原始奖励会诱导局部最优：**少出力 → 回报≈0**。若训练早期碰不到 +100，策略会收敛到 `a≈0`，永远停在谷底。

本课因此加了「教学向」帮助（**不改验收口径**）：

| 手段 | 作用 |
|------|------|
| 机械能势能 shaping | replay 奖励 = 原始 + `γΦ(s')-Φ(s)`，`Φ=sin(3x)+½v²`，系数 100 |
| 探索噪声 | 策略动作再加 `N(0, 0.5)`（仅采集） |
| 固定温度 α=0.2 | 此环境上自动 α 会塌到 ~0，探索随之消失 |

日志里的 `train_avg20` / `eval_mean` / `SAVE_THRESHOLD` 一律按**原始**回报计算；达到阈值可提前结束训练。

---

## 3. SAC 在学什么

与 [`../../pendulum/pendulum-sac/`](../../pendulum/pendulum-sac/) 同一套：

| 组件 | 作用 |
|------|------|
| Actor | 高斯 → `tanh` 压到 `[-1,1]`；重参数化采样 |
| 双 Q | \(Q_1, Q_2\)；target 取 **min** |
| Target 网 | 软更新 `τ` |
| Replay | 冲顶样本可反复学 |
| 自动 α | 学 `log α`，熵目标 `-|A|` |

Bootstrap：`terminated`（到顶）才当真正结束；`truncated`（到步数上限）不当 terminal。

---

## 4. 和 Pendulum SAC 差在哪

| | `pendulum-sac` | 本目录 |
|--|----------------|--------|
| 奖励 | 稠密、几乎每步有梯度 | 稀疏成功 + 动作惩罚（易学「不动」） |
| 策略直觉 | 拧起并稳住 | 先荡后冲 |
| 额外技巧 | 无 | 能量 shaping + 采集噪声 + 固定 α |
| 验收阈值 | ≥ -200 | ≥ 90（原始回报） |

---

## 5. 验收

- `eval_mean` 抬升到约 **≥ 90** 会打印 threshold；最佳 Actor 写入 `sac_mountain_car.pth`  
- `sac_test.py` 窗口里车应能左右蓄力后冲上右侧旗杆  

---

## 6. 非目标

- 不做观测归一化、优先经验回放、好奇心 / HER  
- 不做 PPO / TD3 对照目录（需要时可再加）  
