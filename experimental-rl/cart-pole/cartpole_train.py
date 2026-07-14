import random
import collections
import math
import sys
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Windows 控制台/重定向默认用 GBK，遇到 emoji 会崩，强制 stdout/stderr 用 UTF-8
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ----------------- 1. 超参数设置 -----------------
LR = 1e-3  # 学习率
GAMMA = 0.99  # 折扣因子
BATCH_SIZE = 64  # 每次批量训练的样本数
MEMORY_SIZE = 10000  # 经验回放池容量
MIN_MEMORY_SIZE = 1000  # 回放池最少样本数（达到后才开始训练）
TARGET_UPDATE = 10  # 目标网络更新频率（回合）
EPS_START = 1.0  # 初始随机探索率
EPS_END = 0.01  # 最小随机探索率
EPS_DECAY_STEPS = 5000  # 探索率按"训练步数"指数衰减的步数常数
MAX_EPISODES = 500  # 总训练回合数
SAVE_BEST = "dqn_cartpole.pth"  # 保存"历史最佳"模型的路径


# ----------------- 2. 神经网络结构 -----------------
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        # 2层全连接网络，输入4维状态，输出2维动作对应的Q值
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


# ----------------- 3. 经验回放池 -----------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (torch.FloatTensor(np.array(state)),
                torch.LongTensor(action),
                torch.FloatTensor(reward),
                torch.FloatTensor(np.array(next_state)),
                torch.FloatTensor(done))

    def __len__(self):
        return len(self.buffer)


# ----------------- 4. DQN 核心算法 -----------------
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.epsilon = EPS_START
        self.step_count = 0  # 记录训练步数，用于按步数衰减探索率

        # 估计网络与目标网络
        self.policy_net = QNet(state_dim, action_dim)
        self.target_net = QNet(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)

    def choose_action(self, state):
        # Epsilon-Greedy 探索策略
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                return self.policy_net(state).argmax().item()

    def train_step(self):
        if len(self.memory) < MIN_MEMORY_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # 计算当前 Q 值
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标 Q 值
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            expected_q_values = rewards + GAMMA * max_next_q_values * (1 - dones)

        # 计算损失并更新
        loss = F.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 按训练步数指数衰减探索率（比按回合衰减更平滑、更早进入利用阶段）
        self.step_count += 1
        self.epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * self.step_count / EPS_DECAY_STEPS)


# ----------------- 5. 训练主循环 -----------------
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n  # 2

    agent = DQNAgent(state_dim, action_dim)
    reward_history = []
    best_avg_reward = -float("inf")  # 记录历史最佳近10局均分，用于保存最佳模型

    print("开始训练 DQN Agent 玩 CartPole...")

    for episode in range(MAX_EPISODES):
        state, info = env.reset()
        episode_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            episode_reward += reward

            if done:
                break

        # 探索率已在 train_step 中按步数衰减，这里无需再处理
        reward_history.append(episode_reward)

        if episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            print(
                f"Episode: {episode + 1}/{MAX_EPISODES} | 近10局均分: {avg_reward:.1f} | Epsilon: {agent.epsilon:.2f}")

            # 近10局均分达到 450 分以上视为通关
            if avg_reward >= 450:
                print(f"🎉 训练成功！在第 {episode + 1} 回合完美通关。")

            # 保存"历史最佳"模型：仅当近10局均分刷新历史最高时才覆盖保存
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(agent.policy_net.state_dict(), SAVE_BEST)
                print(f"💾 保存新最佳模型（近10局均分 {best_avg_reward:.1f}）")

    env.close()

    # ====== 收尾：若始终未触发"保存最佳"则兜底保存最后模型 ======
    import os
    if not os.path.exists(SAVE_BEST):
        torch.save(agent.policy_net.state_dict(), SAVE_BEST)
        print("💾 保存最后模型为 dqn_cartpole.pth（未刷新历史最佳）")

    # 绘制训练曲线
    plt.plot(reward_history)
    plt.title("DQN CartPole Reward History")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()
