import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


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


def test_and_render():
    # 开启人类视觉渲染模式
    env = gym.make("CartPole-v1", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = QNet(state_dim, action_dim)
    try:
        model.load_state_dict(torch.load("dqn_cartpole.pth"))
        model.eval()
        print("🚀 成功加载模型，准备看 AI 表演！")
    except FileNotFoundError:
        print("❌ 未找到 dqn_cartpole.pth 文件，请先运行训练脚本。")
        return

    # 让 AI 纯测试玩 5 局
    for test_episode in range(5):
        state, info = env.reset()
        episode_reward = 0
        print(f"\n--- 第 {test_episode + 1} 局开始 ---")

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()  # 完全交由AI决策

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward

            time.sleep(0.02)  # 控制画面速度

            if done:
                print(f"游戏结束！本局 AI 坚持了 {int(episode_reward)} 步。")
                time.sleep(1)
                break

    env.close()


if __name__ == "__main__":
    test_and_render()
