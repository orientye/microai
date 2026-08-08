"""Double DQN on GridWorld with randomized obstacle layouts.

Tabular Q(s=cell) cannot handle changing walls: the same cell needs different
actions on different maps. This script feeds a 3-channel grid observation
(agent / obstacle / goal) into a small MLP Q-network.
"""

from __future__ import annotations

import collections
import math
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from grid_world_env import GridWorldEnv

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 20000
MIN_MEMORY_SIZE = 1000
TAU = 0.005
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 8000
MAX_EPISODES = 3000
EVAL_EVERY = 100
EVAL_LAYOUTS = 40
SAVE_BEST = "dqn_random_layout.pth"
SAVE_CURVE = "dqn_reward_history.png"


class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: collections.deque = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.as_tensor(np.array(state), dtype=torch.float32),
            torch.as_tensor(action, dtype=torch.int64),
            torch.as_tensor(reward, dtype=torch.float32),
            torch.as_tensor(np.array(next_state), dtype=torch.float32),
            torch.as_tensor(done, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.action_dim = action_dim
        self.epsilon = EPS_START
        self.step_count = 0
        self.policy_net = QNet(state_dim, action_dim)
        self.target_net = QNet(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)

    def choose_action(self, state: np.ndarray, *, greedy: bool = False) -> int:
        if (not greedy) and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            return int(self.policy_net(x).argmax(dim=1).item())

    def train_step(self) -> None:
        if len(self.memory) < MIN_MEMORY_SIZE:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target = rewards + GAMMA * next_q * (1.0 - dones)
        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        self.epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(
            -self.step_count / EPS_DECAY_STEPS
        )
        for tp, p in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(TAU * p.data + (1.0 - TAU) * tp.data)


def evaluate(agent: DQNAgent, env: GridWorldEnv, n_layouts: int) -> dict[str, float]:
    returns = []
    successes = 0
    for _ in range(n_layouts):
        state, _ = env.reset(options={"randomize_layout": True, "random_start": True})
        total = 0.0
        done = False
        reached = False
        while not done:
            action = agent.choose_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            reached = terminated
            done = terminated or truncated
        returns.append(total)
        successes += int(reached)
    return {
        "mean_return": float(np.mean(returns)),
        "success_rate": successes / n_layouts,
        "min_return": float(np.min(returns)),
    }


def main() -> None:
    env = GridWorldEnv(randomize_layout=True, n_obstacles=3)
    state_dim = int(np.prod(env.observation_space.shape))
    agent = DQNAgent(state_dim, env.action_space.n)

    episode_returns: list[float] = []
    eval_curve: list[tuple[int, float]] = []
    best_score = -1.0

    print("Training Double DQN on randomized GridWorld layouts...")
    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset(options={"randomize_layout": True, "random_start": True})
        ep_return = 0.0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.push(state, action, reward, next_state, float(done))
            agent.train_step()
            state = next_state
            ep_return += reward
        episode_returns.append(ep_return)

        if episode % EVAL_EVERY == 0 or episode == MAX_EPISODES:
            stats = evaluate(agent, env, EVAL_LAYOUTS)
            eval_curve.append((episode, stats["mean_return"]))
            print(
                f"episode={episode:4d}  eps={agent.epsilon:.3f}  "
                f"train_return={ep_return:.3f}  "
                f"eval_mean={stats['mean_return']:.3f}  "
                f"success={stats['success_rate']:.0%}  "
                f"min={stats['min_return']:.3f}"
            )
            # Prefer higher success; tie-break on mean return.
            score = stats["success_rate"] + 0.001 * stats["mean_return"]
            if score >= best_score:
                best_score = score
                torch.save(agent.policy_net.state_dict(), SAVE_BEST)
                print(
                    f"  saved {SAVE_BEST} "
                    f"(success={stats['success_rate']:.0%}, mean={stats['mean_return']:.3f})"
                )

    if not os.path.exists(SAVE_BEST):
        torch.save(agent.policy_net.state_dict(), SAVE_BEST)

    fig, ax = plt.subplots(figsize=(8, 4))
    window = 50
    if len(episode_returns) >= window:
        smooth = np.convolve(episode_returns, np.ones(window) / window, mode="valid")
        ax.plot(range(window, len(episode_returns) + 1), smooth, label=f"train return (MA{window})")
    else:
        ax.plot(episode_returns, label="train return")
    if eval_curve:
        xs, ys = zip(*eval_curve)
        ax.plot(xs, ys, "o-", label="random-layout greedy mean")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title("GridWorld DQN (random layouts)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SAVE_CURVE, dpi=140)
    plt.close(fig)
    print(f"saved curve -> {SAVE_CURVE}")


if __name__ == "__main__":
    main()
