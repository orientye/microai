"""Soft Actor-Critic on MountainCarContinuous-v0 (twin Q + auto temperature)."""

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
from torch.distributions import Normal

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

HERE = Path(__file__).resolve().parent

# ----------------- Hyperparameters -----------------
LR = 3e-4
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 256
MEMORY_SIZE = 100_000
START_STEPS = 10_000
UPDATE_AFTER = 1_000
UPDATES_PER_STEP = 1
MAX_STEPS = 80_000
LOG_EVERY = 5_000
EVAL_EVERY = 5_000
EVAL_EPISODES = 5
SEED = 0
HID = 256
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
# Deceptive env reward: without help SAC collapses to a≈0.
# Train on energy potential shaping; eval / logs use raw reward.
SHAPE_COEF = 100.0
EXPL_NOISE = 0.5
# Auto-α collapses to ~0 here and kills exploration; keep fixed temperature.
FIXED_ALPHA = 0.2

SAVE_THRESHOLD = 90.0
SAVE_BEST = "sac_mountain_car.pth"
SAVE_CURVE = "reward_history.png"


def mech_energy(position: float, velocity: float) -> float:
    """Potential (track height) + kinetic energy proxy."""
    return math.sin(3.0 * float(position)) + 0.5 * float(velocity) ** 2


def shaped_reward(
    raw: float,
    position: float,
    velocity: float,
    next_position: float,
    next_velocity: float,
) -> float:
    """Potential-based shaping on mechanical energy."""
    phi = mech_energy(position, velocity)
    phi_n = mech_energy(next_position, next_velocity)
    return float(raw) + SHAPE_COEF * (GAMMA * phi_n - phi)


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
            torch.FloatTensor(np.array(action)),
            torch.FloatTensor(reward).unsqueeze(-1),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(done).unsqueeze(-1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class GaussianActor(nn.Module):
    """Squashed Gaussian: a = high * tanh(u), u ~ N(μ(s), σ(s))."""

    def __init__(self, state_dim: int, action_dim: int, action_high: float):
        super().__init__()
        self.action_high = float(action_high)
        self.net = nn.Sequential(
            nn.Linear(state_dim, HID),
            nn.ReLU(),
            nn.Linear(HID, HID),
            nn.ReLU(),
        )
        self.mean = nn.Linear(HID, action_dim)
        self.log_std = nn.Linear(HID, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(state)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reparameterized sample + tanh correction; also returns pre-tanh mean action."""
        mean, log_std = self(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        u = dist.rsample()
        action = torch.tanh(u) * self.action_high
        # change of variables: a = high * tanh(u)
        log_prob = dist.log_prob(u) - torch.log(1.0 - torch.tanh(u).pow(2) + 1e-6)
        log_prob = log_prob - math.log(self.action_high)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_high
        return action, log_prob, mean_action

    @torch.no_grad()
    def act(self, state: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        if deterministic:
            mean, _ = self(x)
            action = torch.tanh(mean) * self.action_high
        else:
            action, _, _ = self.sample(x)
        return action.squeeze(0).cpu().numpy().astype(np.float32)


class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, HID),
            nn.ReLU(),
            nn.Linear(HID, HID),
            nn.ReLU(),
            nn.Linear(HID, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, action_high: float):
        self.action_high = float(action_high)
        self.actor = GaussianActor(state_dim, action_dim, action_high)
        self.q1 = QNet(state_dim, action_dim)
        self.q2 = QNet(state_dim, action_dim)
        self.q1_targ = QNet(state_dim, action_dim)
        self.q2_targ = QNet(state_dim, action_dim)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=LR)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=LR)
        self._alpha = float(FIXED_ALPHA)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.tensor(self._alpha)

    def soft_update(self) -> None:
        for src, dst in (
            (self.q1, self.q1_targ),
            (self.q2, self.q2_targ),
        ):
            for p, tp in zip(src.parameters(), dst.parameters()):
                tp.data.mul_(1.0 - TAU).add_(TAU * p.data)

    def update(self, memory: ReplayBuffer) -> dict[str, float]:
        states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            q1_n = self.q1_targ(next_states, next_actions)
            q2_n = self.q2_targ(next_states, next_actions)
            q_n = torch.min(q1_n, q2_n) - self.alpha * next_log_probs
            target = rewards + (1.0 - dones) * GAMMA * q_n

        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        pi_actions, log_probs, _ = self.actor.sample(states)
        q1_pi = self.q1(states, pi_actions)
        q2_pi = self.q2(states, pi_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_probs - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.soft_update()
        return {
            "q1_loss": float(q1_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self._alpha),
        }


def evaluate(env: gym.Env, agent: SACAgent, n_episodes: int) -> float:
    totals: list[float] = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        ep = 0.0
        done = False
        while not done:
            action = agent.actor.act(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            ep += float(reward)
            done = terminated or truncated
        totals.append(ep)
    return float(np.mean(totals))


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make("MountainCarContinuous-v0")
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    action_high = float(env.action_space.high[0])

    agent = SACAgent(state_dim, action_dim, action_high)
    memory = ReplayBuffer(MEMORY_SIZE)

    state, _ = env.reset(seed=SEED)
    episode_reward = 0.0
    reward_history: list[float] = []
    best_eval = float("-inf")
    reported = False
    last_stats: dict[str, float] = {"q1_loss": 0.0, "actor_loss": 0.0, "alpha": 1.0}

    print(
        "Training SAC on MountainCarContinuous-v0 "
        f"(energy shaping coef={SHAPE_COEF}, explor noise={EXPL_NOISE}, "
        f"fixed alpha={FIXED_ALPHA})..."
    )
    for step in range(1, MAX_STEPS + 1):
        if step <= START_STEPS:
            action = env.action_space.sample().astype(np.float32)
        else:
            action = agent.actor.act(state, deterministic=False)
            if EXPL_NOISE > 0.0:
                action = action + np.random.randn(*action.shape).astype(np.float32) * EXPL_NOISE
                action = np.clip(action, -action_high, action_high).astype(np.float32)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        # Train on shaped reward; logging / eval keep raw env return.
        train_r = shaped_reward(
            float(reward),
            float(state[0]),
            float(state[1]),
            float(next_state[0]),
            float(next_state[1]),
        )
        # Gymnasium truncates at horizon; treat as non-terminal for bootstrap.
        memory.push(state, action, train_r, next_state, float(terminated))

        episode_reward += float(reward)
        state = next_state
        if done:
            reward_history.append(episode_reward)
            episode_reward = 0.0
            state, _ = env.reset()

        if step >= UPDATE_AFTER and len(memory) >= BATCH_SIZE:
            for _ in range(UPDATES_PER_STEP):
                last_stats = agent.update(memory)

        if step % LOG_EVERY == 0 and len(reward_history) >= 5:
            avg_train = float(np.mean(reward_history[-20:]))
            print(
                f"step={step:6d}/{MAX_STEPS}  "
                f"train_avg20={avg_train:7.1f}  "
                f"alpha={last_stats['alpha']:.3f}  "
                f"q1_loss={last_stats['q1_loss']:.3f}  "
                f"actor_loss={last_stats['actor_loss']:.3f}"
            )

        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            eval_mean = evaluate(env, agent, EVAL_EPISODES)
            print(f"  eval_mean({EVAL_EPISODES})={eval_mean:7.1f}")
            if eval_mean > best_eval:
                best_eval = eval_mean
                torch.save(agent.actor.state_dict(), HERE / SAVE_BEST)
                print(f"  saved best -> {SAVE_BEST} (eval={best_eval:.1f})")
            if eval_mean >= SAVE_THRESHOLD and not reported:
                print(f"  reached threshold {SAVE_THRESHOLD} at step {step}")
                reported = True
                break

    env.close()
    if not (HERE / SAVE_BEST).exists():
        torch.save(agent.actor.state_dict(), HERE / SAVE_BEST)
        print(f"saved fallback -> {SAVE_BEST}")

    fig, ax = plt.subplots(figsize=(9, 4))
    if reward_history:
        window = 20
        if len(reward_history) >= window:
            smooth = np.convolve(reward_history, np.ones(window) / window, mode="valid")
            ax.plot(range(window, len(reward_history) + 1), smooth, label=f"return MA{window}")
        else:
            ax.plot(reward_history, label="return")
    ax.axhline(
        SAVE_THRESHOLD, color="gray", linestyle="--", linewidth=1, label=f"threshold {SAVE_THRESHOLD}"
    )
    ax.set_xlabel("episode")
    ax.set_ylabel("return (env scale)")
    ax.set_title("MountainCarContinuous-v0 Soft Actor-Critic")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / SAVE_CURVE, dpi=140)
    plt.close(fig)
    print(f"saved curve -> {SAVE_CURVE}; best_eval={best_eval:.1f}")


if __name__ == "__main__":
    main()
