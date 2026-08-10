# CliffWalking 示例

https://gymnasium.farama.org/environments/toy_text/cliff_walking/

本目录对齐 `grid-world/`：同一悬崖环境、两种表示。

| 子目录 | 内容 |
|--------|------|
| [`cliff-walking-q-sarsa`](cliff-walking-q-sarsa/) | 表格 **Q-learning vs SARSA**（先看这个） |
| [`cliff-walking-dqn`](cliff-walking-dqn/) | 同一环境 + one-hot **Double DQN** |

```bash
# 1) 表格：离策略 vs 在策略
cd cliff-walking-q-sarsa
python q_sarsa_train.py
python q_sarsa_test.py          # human 像素动画

# 2) 网络：值函数近似
cd ../cliff-walking-dqn
python dqn_train.py
python dqn_test.py
```

- 表格：看清「`max Q` vs 真实下一步 `Q`」在悬崖上如何分叉  
- DQN：同一地图，`Q` 换成 MLP；可与 CartPole DQN 对照实现细节  
