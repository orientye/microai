"""CNN Double DQN on randomized GridWorld with curriculum + distance shaping."""

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

LR = 5e-4
GAMMA = 0.99
BATCH_SIZE = 128
MEMORY_SIZE = 50000
MIN_MEMORY_SIZE = 2000
TAU = 0.005
GRAD_CLIP = 10.0
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY_STEPS = 25000
MAX_EPISODES = 6000
EVAL_EVERY = 200
EVAL_LAYOUTS = 150
# Hard-only fine-tune after curriculum (or via --finetune).
FINETUNE_EPISODES = 4000
FINETUNE_LR = 1e-4
FINETUNE_EPS_START = 0.15
FINETUNE_EPS_END = 0.01
FINETUNE_EPS_DECAY = 12000
FINETUNE_EVAL_LAYOUTS = 200
HARD_MIX = 0.35  # fraction of finetune episodes that replay failing layouts
# Potential-based shaping: Φ = -manhattan / max_manhattan
SHAPE_COEF = 0.1
SAVE_BEST = "dqn_random_layout.pth"
SAVE_CURVE = "dqn_reward_history.png"
SEED = 0


class QNet(nn.Module):
    def __init__(self, grid_size: int, action_dim: int, n_channels: int = 4):
        super().__init__()
        self.grid_size = grid_size
        self.n_channels = n_channels
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        flat = 64 * grid_size * grid_size
        self.head = nn.Sequential(
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, self.n_channels, self.grid_size, self.grid_size)
        x = self.conv(x)
        return self.head(x.flatten(1))


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
    def __init__(self, grid_size: int, action_dim: int):
        self.action_dim = action_dim
        self.epsilon = EPS_START
        self.step_count = 0
        self.policy_net = QNet(grid_size, action_dim)
        self.target_net = QNet(grid_size, action_dim)
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
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        self.step_count += 1
        self.epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(
            -self.step_count / EPS_DECAY_STEPS
        )
        for tp, p in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(TAU * p.data + (1.0 - TAU) * tp.data)


def curriculum_obstacles(episode: int) -> int:
    # Easy → hard: 1 obstacle, then 2, then full 3.
    if episode <= 2000:
        return 1
    if episode <= 4000:
        return 2
    return 3


def shaped_reward(
    env: GridWorldEnv,
    base_reward: float,
    prev_dist: int,
    next_dist: int,
    done: bool,
) -> float:
    """Potential-based shaping: F = γΦ(s') - Φ(s), Φ = -dist / max_dist."""
    max_dist = max(1, 2 * (env.size - 1))
    phi = -prev_dist / max_dist
    phi_next = 0.0 if done else (-next_dist / max_dist)
    return base_reward + SHAPE_COEF * (GAMMA * phi_next - phi)


def evaluate(agent: DQNAgent, env: GridWorldEnv, n_layouts: int) -> dict[str, float]:
    """Evaluate on hardest setting: 3 obstacles, random reachable starts."""
    returns = []
    successes = 0
    for _ in range(n_layouts):
        state, info = env.reset(
            options={"randomize_layout": True, "random_start": True, "n_obstacles": 3}
        )
        total = 0.0
        done = False
        reached = False
        while not done:
            action = agent.choose_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            # Report raw env return (no shaping) for apples-to-apples scores.
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_episode(
    agent: DQNAgent,
    env: GridWorldEnv,
    n_obs: int,
    *,
    obstacles: set[tuple[int, int]] | frozenset[tuple[int, int]] | None = None,
    start: tuple[int, int] | None = None,
) -> float:
    if obstacles is not None and start is not None:
        options = {
            "randomize_layout": False,
            "obstacles": obstacles,
            "start": start,
        }
    else:
        options = {
            "randomize_layout": True,
            "random_start": True,
            "n_obstacles": n_obs,
        }
    state, info = env.reset(options=options)
    prev_dist = int(info["manhattan"])
    ep_return = 0.0
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        next_dist = int(step_info["manhattan"])
        train_r = shaped_reward(env, reward, prev_dist, next_dist, done)
        agent.memory.push(state, action, train_r, next_state, float(done))
        agent.train_step()
        state = next_state
        prev_dist = next_dist
        ep_return += reward
    return ep_return


def collect_failures(
    agent: DQNAgent,
    env: GridWorldEnv,
    n_probe: int = 250,
) -> list[tuple[frozenset[tuple[int, int]], tuple[int, int]]]:
    """Gather (obstacles, start) pairs where greedy policy currently fails."""
    hard: list[tuple[frozenset[tuple[int, int]], tuple[int, int]]] = []
    for _ in range(n_probe):
        state, info = env.reset(
            options={"randomize_layout": True, "random_start": True, "n_obstacles": 3}
        )
        done = False
        reached = False
        while not done:
            action = agent.choose_action(state, greedy=True)
            state, _, terminated, truncated, _ = env.step(action)
            reached = terminated
            done = terminated or truncated
        if not reached:
            hard.append((frozenset(info["obstacles"]), info["start"]))
    return hard


def _maybe_save(agent: DQNAgent, env: GridWorldEnv, best_score: float, n_eval: int) -> float:
    stats = evaluate(agent, env, n_eval)
    score = stats["success_rate"] + 0.001 * stats["mean_return"]
    print(
        f"  eval_mean={stats['mean_return']:.3f}  "
        f"success={stats['success_rate']:.0%}  "
        f"min={stats['min_return']:.3f}",
        flush=True,
    )
    if score > best_score:
        torch.save(agent.policy_net.state_dict(), SAVE_BEST)
        print(
            f"  saved {SAVE_BEST} "
            f"(success={stats['success_rate']:.0%}, mean={stats['mean_return']:.3f})",
            flush=True,
        )
        return score
    return best_score


def _save_curve(
    episode_returns: list[float],
    eval_curve: list[tuple[int, float]],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    window = 50
    if len(episode_returns) >= window:
        smooth = np.convolve(episode_returns, np.ones(window) / window, mode="valid")
        ax.plot(range(window, len(episode_returns) + 1), smooth, label=f"train return (MA{window})")
    else:
        ax.plot(episode_returns, label="train return")
    if eval_curve:
        xs, ys = zip(*eval_curve)
        ax.plot(xs, ys, "o-", label="eval mean (3 obstacles)")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SAVE_CURVE, dpi=140)
    plt.close(fig)
    print(f"saved curve -> {SAVE_CURVE}", flush=True)


def main() -> None:
    set_seed(SEED)
    env = GridWorldEnv(randomize_layout=True, n_obstacles=3)
    agent = DQNAgent(env.size, env.action_space.n)

    episode_returns: list[float] = []
    eval_curve: list[tuple[int, float]] = []
    best_score = -1.0

    print("Training CNN-DQN with curriculum + distance shaping...", flush=True)
    for episode in range(1, MAX_EPISODES + 1):
        n_obs = curriculum_obstacles(episode)
        ep_return = _run_episode(agent, env, n_obs)
        episode_returns.append(ep_return)

        if episode % EVAL_EVERY == 0 or episode == MAX_EPISODES:
            print(
                f"episode={episode:4d}  n_obs={n_obs}  eps={agent.epsilon:.3f}  "
                f"train_return={ep_return:.3f}",
                flush=True,
            )
            stats = evaluate(agent, env, EVAL_LAYOUTS)
            eval_curve.append((episode, stats["mean_return"]))
            score = stats["success_rate"] + 0.001 * stats["mean_return"]
            print(
                f"  eval_mean={stats['mean_return']:.3f}  "
                f"success={stats['success_rate']:.0%}  "
                f"min={stats['min_return']:.3f}",
                flush=True,
            )
            if score >= best_score:
                best_score = score
                torch.save(agent.policy_net.state_dict(), SAVE_BEST)
                print(
                    f"  saved {SAVE_BEST} "
                    f"(success={stats['success_rate']:.0%}, mean={stats['mean_return']:.3f})",
                    flush=True,
                )

    if not os.path.exists(SAVE_BEST):
        torch.save(agent.policy_net.state_dict(), SAVE_BEST)

    _save_curve(episode_returns, eval_curve, "GridWorld CNN-DQN (curriculum + shaping)")


def finetune() -> None:
    """Hard-only fine-tune from an existing checkpoint + failure replay."""
    if not os.path.exists(SAVE_BEST):
        raise SystemExit(f"missing {SAVE_BEST}; run curriculum training first")

    set_seed(SEED + 1)
    env = GridWorldEnv(randomize_layout=True, n_obstacles=3)
    agent = DQNAgent(env.size, env.action_space.n)
    try:
        state_dict = torch.load(SAVE_BEST, map_location="cpu", weights_only=True)
        agent.policy_net.load_state_dict(state_dict)
        agent.target_net.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise SystemExit(
            f"checkpoint incompatible with current QNet (likely old 3-channel weights): {exc}\n"
            "Run: python dqn_train.py"
        ) from exc
    agent.optimizer = optim.Adam(agent.policy_net.parameters(), lr=FINETUNE_LR)
    agent.epsilon = FINETUNE_EPS_START
    agent.step_count = 0

    global EPS_START, EPS_END, EPS_DECAY_STEPS
    EPS_START = FINETUNE_EPS_START
    EPS_END = FINETUNE_EPS_END
    EPS_DECAY_STEPS = FINETUNE_EPS_DECAY

    print("Fine-tuning on 3-obstacle layouts + hard-failure replay...", flush=True)
    base = evaluate(agent, env, FINETUNE_EVAL_LAYOUTS)
    best_score = base["success_rate"] + 0.001 * base["mean_return"]
    print(
        f"  baseline success={base['success_rate']:.0%}  "
        f"mean={base['mean_return']:.3f}",
        flush=True,
    )
    hard_cases = collect_failures(agent, env, n_probe=300)
    print(f"  mined hard cases: {len(hard_cases)}", flush=True)

    episode_returns: list[float] = []
    eval_curve: list[tuple[int, float]] = []
    for episode in range(1, FINETUNE_EPISODES + 1):
        use_hard = hard_cases and random.random() < HARD_MIX
        if use_hard:
            obstacles, start = random.choice(hard_cases)
            ep_return = _run_episode(agent, env, n_obs=3, obstacles=obstacles, start=start)
        else:
            ep_return = _run_episode(agent, env, n_obs=3)
        episode_returns.append(ep_return)

        if episode % EVAL_EVERY == 0 or episode == FINETUNE_EPISODES:
            print(
                f"finetune={episode:4d}  eps={agent.epsilon:.3f}  "
                f"train_return={ep_return:.3f}  hard_pool={len(hard_cases)}",
                flush=True,
            )
            stats = evaluate(agent, env, FINETUNE_EVAL_LAYOUTS)
            eval_curve.append((episode, stats["mean_return"]))
            score = stats["success_rate"] + 0.001 * stats["mean_return"]
            print(
                f"  eval_mean={stats['mean_return']:.3f}  "
                f"success={stats['success_rate']:.0%}  "
                f"min={stats['min_return']:.3f}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                torch.save(agent.policy_net.state_dict(), SAVE_BEST)
                print(
                    f"  saved {SAVE_BEST} "
                    f"(success={stats['success_rate']:.0%}, mean={stats['mean_return']:.3f})",
                    flush=True,
                )
            # Refresh hard pool periodically as the policy improves.
            if episode % 1000 == 0:
                hard_cases = collect_failures(agent, env, n_probe=300)
                print(f"  refreshed hard cases: {len(hard_cases)}", flush=True)

    _save_curve(episode_returns, eval_curve, "GridWorld CNN-DQN (hard-replay fine-tune)")


if __name__ == "__main__":
    if "--finetune" in sys.argv:
        finetune()
    else:
        main()
        # Auto hard-finetune with the new 4-channel checkpoint.
        finetune()
