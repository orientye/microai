# GridWorld 示例

本目录包含两个递进例子：

| 子目录 | 内容 |
|--------|------|
| [`grid-world-qlearning`](grid-world-qlearning/) | 固定地图 + 表格 Q-learning（先看这个） |
| [`grid-world-dqn`](grid-world-dqn/) | 随机地图 + CNN Double DQN（看见整张图才能换墙） |

```bash
# 1) 固定地图表格法
cd grid-world-qlearning
python q_learning_train.py
python q_learning_test.py

# 2) 随机布局 DQN
cd ../grid-world-dqn
python dqn_train.py
python dqn_test.py          # 随机多种子评估（默认 5×200）
```

- Q-learning：离散状态查表，策略可画成箭头图  
- DQN：每局换障碍；最终用**随机多种子**测泛化，训练过程用固定评估种子看收敛  
