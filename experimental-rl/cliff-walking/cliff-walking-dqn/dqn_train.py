"""CliffWalking-v1 + Double DQN (one-hot state → MLP Q).

Phase B of the cliff-walking lesson: same env as tabular Q/SARSA,
but Q is a neural net instead of a table.
"""

from __future__ import annotations

import collections
import math
import random
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent

# ----------------- Hyperparameters -----------------
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 20_000
MIN_MEMORY_SIZE = 1_000
TAU = 0.005
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 8_000
MAX_EPISODES = 800
MAX_EPISODE_STEPS = 100
EVAL_EVERY = 50
EVAL_EPISODES = 20
SEED = 0

N_STATES = 48
N_ACTIONS = 4
NROWS, NCOLS = 4, 12
GOAL = (3, 11)
CLIFF_COLS = range(1, 11)
ACTION_ARROWS = ("↑", "→", "↓", "←")

SAVE_BEST = "dqn_cliff.pth"
SAVE_CURVE = "reward_history.png"
SAVE_POLICY = "policy_dqn.png"


def make_env(*, render_mode: str | None = None) -> gym.Env:
    return gym.make(
        "CliffWalking-v1",
        is_slippery=False,
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=render_mode,
    )


def encode(state: int) -> np.ndarray:
    """One-hot over 48 cells — network approximates a soft Q-table."""
    x = np.zeros(N_STATES, dtype=np.float32)
    x[int(state)] = 1.0
    return x


def is_cliff(row: int, col: int) -> bool:
    return row == NROWS - 1 and col in CLIFF_COLS


def rc_to_state(row: int, col: int) -> int:
    return row * NCOLS + col


class QNet(nn.Module):
    def __init__(self, state_dim: int = N_STATES, action_dim: int = N_ACTIONS):
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
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(done),
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

    def choose_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q = self.policy_net(torch.FloatTensor(state).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def train_step(self) -> None:
        if len(self.memory) < MIN_MEMORY_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            # Cliff fall: done=False (sent to start); only goal/truncated cuts bootstrap.
            expected = rewards + GAMMA * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, expected)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.step_count += 1
        self.epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.step_count / EPS_DECAY_STEPS
        )
        for tp, p in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(TAU * p.data + (1.0 - TAU) * tp.data)


def net_to_q_table(net: QNet) -> np.ndarray:
    q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    net.eval()
    with torch.no_grad():
        for s in range(N_STATES):
            q[s] = net(torch.FloatTensor(encode(s)).unsqueeze(0)).numpy()[0]
    return q


def print_policy(q: np.ndarray, title: str) -> None:
    print(f"{title} (↑→↓←; C=cliff, G=goal):")
    for r in range(NROWS):
        cells: list[str] = []
        for c in range(NCOLS):
            if is_cliff(r, c):
                cells.append("C")
            elif (r, c) == GOAL:
                cells.append("G")
            else:
                cells.append(ACTION_ARROWS[int(np.argmax(q[rc_to_state(r, c)]))])
        print(" ".join(cells))


def save_policy_figure(q: np.ndarray, path: Path, title: str) -> None:
    values = np.full((NROWS, NCOLS), np.nan, dtype=np.float64)
    arrows = np.full((NROWS, NCOLS), "", dtype=object)
    for r in range(NROWS):
        for c in range(NCOLS):
            if is_cliff(r, c):
                continue
            s = rc_to_state(r, c)
            values[r, c] = float(np.max(q[s]))
            if (r, c) != GOAL:
                arrows[r, c] = ACTION_ARROWS[int(np.argmax(q[s]))]

    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(values, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="V(s)=max_a Q(s,a)")
    for r in range(NROWS):
        for c in range(NCOLS):
            if is_cliff(r, c):
                ax.text(c, r, "C", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
            elif (r, c) == GOAL:
                ax.text(c, r, "G", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
            else:
                ax.text(c, r, arrows[r, c], ha="center", va="center", color="white", fontsize=12)
    ax.set_xticks(range(NCOLS))
    ax.set_yticks(range(NROWS))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def evaluate_greedy(env: gym.Env, net: QNet, *, n_episodes: int) -> dict[str, float]:
    net.eval()
    returns: list[float] = []
    successes = 0
    for _ in range(n_episodes):
        state, _ = env.reset()
        total = 0.0
        done = False
        reached = False
        while not done:
            with torch.no_grad():
                action = int(net(torch.FloatTensor(encode(state)).unsqueeze(0)).argmax().item())
            state, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            reached = bool(terminated)
            done = terminated or truncated
        returns.append(total)
        successes += int(reached)
    return {
        "success_rate": successes / n_episodes,
        "mean_return": float(np.mean(returns)),
        "min_return": float(np.min(returns)),
    }


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = make_env()
    agent = DQNAgent(N_STATES, N_ACTIONS)
    reward_history: list[float] = []
    best_greedy_mean = -float("inf")

    print("Training Double DQN on CliffWalking-v1 (one-hot)...")
    for episode in range(1, MAX_EPISODES + 1):
        state_i, _ = env.reset(seed=SEED + episode)
        state = encode(state_i)
        ep_return = 0.0

        while True:
            action = agent.choose_action(state)
            next_i, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            next_state = encode(next_i)

            agent.memory.push(state, action, float(reward), next_state, float(done))
            agent.train_step()

            state = next_state
            ep_return += float(reward)
            if done:
                break

        reward_history.append(ep_return)

        if episode % EVAL_EVERY == 0 or episode == MAX_EPISODES:
            stats = evaluate_greedy(env, agent.policy_net, n_episodes=EVAL_EPISODES)
            print(
                f"episode={episode:4d}  eps={agent.epsilon:.3f}  "
                f"train_return={ep_return:8.1f}  "
                f"greedy_mean={stats['mean_return']:7.1f}  "
                f"success={stats['success_rate']:.0%}"
            )
            if stats["mean_return"] > best_greedy_mean and stats["success_rate"] >= 0.9:
                best_greedy_mean = stats["mean_return"]
                torch.save(agent.policy_net.state_dict(), HERE / SAVE_BEST)
                print(f"  saved best -> {SAVE_BEST} (greedy_mean={best_greedy_mean:.1f})")

    env.close()

    if not (HERE / SAVE_BEST).exists():
        torch.save(agent.policy_net.state_dict(), HERE / SAVE_BEST)
        print(f"saved fallback -> {SAVE_BEST}")

    # Reload best for policy figure
    agent.policy_net.load_state_dict(
        torch.load(HERE / SAVE_BEST, map_location="cpu", weights_only=True)
    )
    q = net_to_q_table(agent.policy_net)
    print()
    print_policy(q, "Double DQN greedy policy")
    save_policy_figure(q, HERE / SAVE_POLICY, "CliffWalking Double DQN")
    print(f"saved policy map -> {SAVE_POLICY}")

    fig, ax = plt.subplots(figsize=(9, 4))
    window = 20
    if len(reward_history) >= window:
        smooth = np.convolve(reward_history, np.ones(window) / window, mode="valid")
        ax.plot(range(window, len(reward_history) + 1), smooth, label=f"train return (MA{window})")
    else:
        ax.plot(reward_history, label="train return")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title("CliffWalking-v1 Double DQN")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / SAVE_CURVE, dpi=140)
    plt.close(fig)
    print(f"saved reward curve -> {SAVE_CURVE}")


if __name__ == "__main__":
    main()
