# CliffWalking：表格 Q-learning vs SARSA

本目录在 Gymnasium **`CliffWalking-v1`**（`is_slippery=False`）上对照两种表格 TD 算法，回答：

> 同样 ε-greedy 探索，为什么 **Q-learning** 会贴着悬崖走，而 **SARSA** 更愿意绕远路？

父目录总览：[`../`](../)。网络版：[`../cliff-walking-dqn/`](../cliff-walking-dqn/)。

| 文件 | 作用 |
|------|------|
| `td_updates.py` | 纯函数：Q-learning / SARSA 的一步 TD 更新 |
| `cliff_env.py` | 建环境、ε、选动作、策略可视化 |
| `q_sarsa_train.py` | 同一超参并排训练，存两张 Q 表与图 |
| `q_sarsa_test.py` | 贪心评估 + **Gymnasium human 像素动画**（对齐官方 GIF / CartPole） |
| `test_td_updates.py` | 更新式单元测试（`python -m unittest`） |

```bash
cd experimental-rl/cliff-walking/cliff-walking-q-sarsa
python -m unittest test_td_updates -v
python q_sarsa_train.py
python q_sarsa_test.py
```

依赖：`gymnasium`、`numpy`、`matplotlib`；看动态画面还需 **`pygame`**（`pip install pygame`，与 CartPole `render_mode="human"` 相同）。  
需 Gymnasium 版本支持 `CliffWalking-v1`（较新版本；官方文档当前入口也是 v1）。

`q_sarsa_test.py` 会先打印策略与统计，再弹出官方同款像素窗口，依次演示 Q-learning / SARSA 各 3 局（步间约 0.25s，方便看清贴崖 vs 绕远）。

---

## 1. 环境在学什么

4×12 网格（第 0 行在上）：

```text
. . . . . . . . . . . .
. . . . . . . . . . . .
. . . . . . . . . . . .
S C C C C C C C C C C G
```

| 概念 | 取值 |
|------|------|
| 状态 | 格子 id `0..47`（`row * 12 + col`）；起点 `36`，终点 `47` |
| 动作 | `0`↑ `1`→ `2`↓ `3`← |
| 奖励 | 普通一步 `-1`；踩悬崖 `-100` 并**送回起点**（局不结束） |
| `terminated` | 到达 `G` |
| `truncated` | 本例 `max_episode_steps=100` |
| 滑动 | **关闭**（`is_slippery=False`），保持确定性，突出算法差异 |

和自定义 GridWorld 的差别：悬崖惩罚极大，且掉崖后继续玩——探索时「贴边」的策略在训练回报上会很惨，但贪心评估时可能又最优。

---

## 2. 两种更新（唯一公式差）

行为策略都是 **ε-greedy**。差别只在 bootstrap 用哪一项：

| | Q-learning（离策略） | SARSA（在策略） |
|--|----------------------|-----------------|
| 目标 | \(r + \gamma \max_{a'} Q(s',a')\) | \(r + \gamma Q(s',a')\)（\(a'\) 真选的） |
| 在学什么 | 假设以后永远贪心 | 假设以后仍按当前 ε-greedy 行事 |
| 悬崖边 | 贪心最优路径常贴崖 | 探索会偶发下坠 → 学到「离崖远一点更安全」 |

对应代码：`td_updates.py`。单元测试刻意造了「行为 `a'` ≠ `argmax`」的下一状态，证明两者目标不同。

---

## 3. 训练在看什么

`q_sarsa_train.py` 用**同一套** `α / γ / ε` 日程分别训两种算法，输出：

- `q_qlearning.npy` / `q_sarsa.npy`
- `policy_qlearning.png` / `policy_sarsa.png`（箭头 + `V(s)`）
- `reward_history.png`（训练滑动平均 + 周期性贪心均值）

预期现象（允许种子波动）：

1. **训练回报**：Q-learning 往往更抖、更低（贴崖 + 探索 = 常摔）  
2. **贪心策略图**：Q-learning 沿悬崖上方最近一排右行；SARSA 更靠上排  
3. **贪心评估**：两者都能到 `G`；最优路径更短，但 SARSA「安全路径」在仍有探索时训练更稳

---

## 4. 和仓库其它例子的位置

```text
grid-world-qlearning          固定小图，只看 Q-learning
       │
       ▼
cliff-walking-q-sarsa（本目录） 同一环境对比 Q vs SARSA
       │
       ▼
../cliff-walking-dqn          同一悬崖，Q 换成 one-hot Double DQN
       │
       ▼
cart-pole / grid-world-dqn    连续状态或视觉上的值函数近似
```

上一课（`../../grid-world/grid-world-qlearning/`）只学了 `max` 更新；本课补上「若用真实下一步动作更新，策略会怎么变」。

---

## 5. 常见问题

**Q: 掉崖为什么 `done=False`？**  
A: Gymnasium 按 Sutton 设定：`-100` 并送回起点，回合继续。TD 仍对下一状态（起点）bootstrap。

**Q: 为何 `γ=1`、`α=0.5`？**  
A: 对齐教科书 Cliff Walking 常用设定；episode 短，`γ=1` 更直观。

**Q: 策略图和教科书不完全一样？**  
A: ε 日程、种子、episode 数都会影响。调高 `MAX_EPISODES` 或放慢 ε 衰减，分叉通常更清晰。
