import random
import collections
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

# ----------------- 1. 超参数与网络设置 -----------------
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 10000
MIN_MEMORY_SIZE = 1000
TARGET_UPDATE = 10
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
MAX_EPISODES = 300


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


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


# ----------------- 2. 自动化训练阶段 (不弹窗，全速运行) -----------------
def train_agent():
    # 后台快速训练环境
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]  # 获取整数4
    action_dim = env.action_space.n

    policy_net = QNet(state_dim, action_dim)
    target_net = QNet(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPS_START
    reward_history = []

    print("🤖 正在后台全速训练 AI... 马上就好，请稍候...")

    for episode in range(MAX_EPISODES):
        state, info = env.reset()
        episode_reward = 0

        while True:
            # Epsilon-Greedy 探索
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                state_t = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = policy_net(state_t).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.push(state, action, reward, next_state, done)

            # 训练网络
            if len(memory) >= MIN_MEMORY_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q_values = target_net(next_states).max(1)[0]
                    expected_q_values = rewards + GAMMA * max_next_q_values * (1 - dones)
                loss = F.mse_loss(q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            episode_reward += reward
            if done:
                break

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        reward_history.append(episode_reward)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            print(f"回合: {episode + 1} | 近10局均分: {avg_reward:.1f} | 探索率: {epsilon:.2f}")

            # 如果近10局均分 > 460，说明彻底学会了，终止训练直接看表现
            if avg_reward >= 460:
                print("🎉 AI 已经毕业！正在开启无限滑动炫技模式...")
                env.close()
                return policy_net

    env.close()
    return policy_net


# ----------------- 3. 无限滑动渲染阶段 (真·无限循环) -----------------
def watch_ai_play(trained_model):
    # 创建可视化环境
    env = gym.make("CartPole-v1", render_mode="human")
    trained_model.eval()

    episode_count = 0
    while True:
        episode_count += 1
        state, info = env.reset()  # 严格解包，确保 state 是 numpy 数组
        episode_reward = 0

        while True:
            # 严格的数据流包装，避免数据走样
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = trained_model(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward

            time.sleep(0.015)  # 控制刷新率，让画面丝滑流畅

            if done:
                # 满 500 步或者由于物理惯性微调时，在此处无缝重置，继续下一局
                break


if __name__ == "__main__":
    # 第一步：直接在内存中训练出合格的权重，绝不产生路径Bug
    smart_model = train_agent()

    # 第二步：直接把训练完的模型送给渲染引擎，开始无尽的优雅滑动
    try:
        watch_ai_play(smart_model)
    except KeyboardInterrupt:
        print("\n👋 播放已手动结束。")
