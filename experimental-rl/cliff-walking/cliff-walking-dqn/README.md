# CliffWalking + Double DQN

同一环境 `CliffWalking-v1`（`is_slippery=False`），把表格 `Q[s,a]` 换成 **one-hot → MLP** 的 Double DQN。  
表格对照：[`../cliff-walking-q-sarsa/`](../cliff-walking-q-sarsa/)。父目录：[`../`](../)。

| 文件 | 作用 |
|------|------|
| `dqn_train.py` | 经验回放 + Double DQN + 软更新；存权重与策略图 |
| `dqn_test.py` | 贪心评估 + Gymnasium **human** 像素动画 |

```bash
cd experimental-rl/cliff-walking/cliff-walking-dqn
python dqn_train.py
python dqn_test.py
```

依赖：`gymnasium`、`torch`、`numpy`、`matplotlib`；动画需 `pygame`。

---

## 1. 和表格版差在哪

| | [`../cliff-walking-q-sarsa/`](../cliff-walking-q-sarsa/) | 本目录 |
|--|----------------------------------------------------------|--------|
| Q 的表示 | `numpy` 表 `(48, 4)` | `QNet`：48 维 one-hot → 128 → 128 → 4 |
| 更新 | 逐步 TD（Q / SARSA） | 回放采样 + Double TD + 目标网 |
| 探索 | ε 按 episode 线性衰减 | ε 按环境步指数衰减 |
| γ | `1.0`（对齐教材） | `0.99`（深度 TD 更稳） |
| 损失 | 无（直接改表） | Smooth L1 + 梯度裁剪（应对 `-100` 悬崖） |

观察用 **one-hot**，故意不做 CNN：状态本就是离散格子 id，网络学的是「可微的软 Q 表」，方便和表格课对齐。

---

## 2. 算法要点（对齐 CartPole DQN）

- **估计网** `policy_net`：选动作、算当前 `Q(s,a)`；Double 里还在 `s'` 上选 `a*`
- **目标网** `target_net`：估 `Q(s', a*)`；每步软更新 `τ=0.005`
- **掉崖**：`reward=-100`，送回起点，**`done=False`** → 仍对起点 bootstrap（与表格版一致）

验收：贪心成功率高（训练日志里 `success` 近 100%）；是否贴崖不强制复现教科书图。

---

## 3. 非目标

- 不做深度 SARSA / PPO  
- 不做 `is_slippery=True`（可自行消融）  
- 不抽与 CartPole 的公共 DQN 库  

---

## 4. 阅读顺序

1. [`../cliff-walking-q-sarsa/`](../cliff-walking-q-sarsa/) — 离策略 vs 在策略  
2. **本目录** — 同一悬崖，值函数用网络近似  
3. [`../../cart-pole/cart-pole-dqn/`](../../cart-pole/cart-pole-dqn/) — 连续状态上的同一套 Double DQN  
