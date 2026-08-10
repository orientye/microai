"""Continuous PPO on Pendulum-v1 (Gaussian actor + GAE)."""

from __future__ import annotations

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

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent

# ----------------- Hyperparameters -----------------
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.0  # continuous Pendulum often trains better without entropy push
ROLLOUT_STEPS = 2048
UPDATE_EPOCHS = 10
MINIBATCH_SIZE = 64
MAX_UPDATES = 400
LOG_EVERY = 10
EVAL_EVERY = 10
EVAL_EPISODES = 10
SEED = 0
# Shrink reward magnitude so value MSE does not drown the policy loss.
REWARD_SCALE = 0.1

SAVE_THRESHOLD = -200.0
SAVE_BEST = "ppo_pendulum.pth"
SAVE_CURVE = "reward_history.png"


class ActorCritic(nn.Module):
    """Gaussian policy: tanh-scaled mean + learnable log_std."""

    def __init__(self, state_dim: int, action_dim: int, action_high: float):
        super().__init__()
        self.action_high = float(action_high)
        hid = 128
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hid),
            nn.Tanh(),
            nn.Linear(hid, hid),
            nn.Tanh(),
            nn.Linear(hid, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hid),
            nn.Tanh(),
            nn.Linear(hid, hid),
            nn.Tanh(),
            nn.Linear(hid, 1),
        )
        # std ≈ 0.6 at start — enough explore, not a flat random walk.
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tanh(self.actor(x)) * self.action_high
        value = self.critic(x).squeeze(-1)
        return mean, value

    def _dist(self, mean: torch.Tensor) -> Normal:
        std = self.log_std.exp().clamp(min=1e-3, max=2.0)
        return Normal(mean, std.expand_as(mean))

    def act(self, state: np.ndarray) -> tuple[np.ndarray, float, float]:
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            mean, value = self(x)
            dist = self._dist(mean)
            action = dist.sample()
            action = action.clamp(-self.action_high, self.action_high)
            log_prob = dist.log_prob(action).sum(dim=-1)
        return (
            action.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(value.item()),
        )

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, values = self(states)
        dist = self._dist(mean)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        return log_probs, values, entropy


def normalize_adv(adv: torch.Tensor) -> torch.Tensor:
    return (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    next_value = last_value
    for t in reversed(range(len(rewards))):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[t] = gae
        next_value = values[t]
    return advantages, advantages + values


def ppo_update(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
) -> float:
    n = states.size(0)
    loss_value = 0.0
    for _ in range(UPDATE_EPOCHS):
        idx = torch.randperm(n)
        for start in range(0, n, MINIBATCH_SIZE):
            mb = idx[start : start + MINIBATCH_SIZE]
            new_log_probs, values, entropy = model.evaluate_actions(states[mb], actions[mb])
            ratio = torch.exp(new_log_probs - old_log_probs[mb])
            adv = advantages[mb]
            surrogate = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv
            policy_loss = -torch.min(surrogate, clipped).mean()
            value_loss = F.mse_loss(values, returns[mb])
            loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            loss_value = float(loss.item())
    return loss_value


def evaluate(env: gym.Env, model: ActorCritic, n_episodes: int) -> float:
    """Deterministic mean action; returns unscaled env return."""
    totals: list[float] = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        ep = 0.0
        done = False
        while not done:
            with torch.no_grad():
                x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                mean, _ = model(x)
                action = mean.squeeze(0).cpu().numpy().astype(np.float32)
            state, reward, terminated, truncated, _ = env.step(action)
            ep += float(reward)
            done = terminated or truncated
        totals.append(ep)
    return float(np.mean(totals))


def collect_rollout(
    env: gym.Env,
    model: ActorCritic,
    state: np.ndarray,
    episode_reward: float,
    reward_history: list[float],
):
    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
    for _ in range(ROLLOUT_STEPS):
        action, log_prob, value = model.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        # Scale only for learning; episode_reward keeps raw env scale for logs.
        rewards.append(float(reward) * REWARD_SCALE)
        dones.append(float(done))

        episode_reward += float(reward)
        state = next_state
        if done:
            reward_history.append(episode_reward)
            episode_reward = 0.0
            state, _ = env.reset()

    batch = (
        torch.FloatTensor(np.array(states)),
        torch.FloatTensor(np.array(actions)),
        torch.FloatTensor(log_probs),
        torch.FloatTensor(rewards),
        torch.FloatTensor(dones),
        torch.FloatTensor(values),
    )
    return batch, state, episode_reward


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make("Pendulum-v1")
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    action_high = float(env.action_space.high[0])

    model = ActorCritic(state_dim, action_dim, action_high)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    state, _ = env.reset(seed=SEED)
    episode_reward = 0.0
    reward_history: list[float] = []
    best_eval = float("-inf")
    reported = False

    print("Training continuous PPO on Pendulum-v1...")
    for update in range(1, MAX_UPDATES + 1):
        batch, state, episode_reward = collect_rollout(
            env, model, state, episode_reward, reward_history
        )
        states, actions, old_log_probs, rewards, dones, values = batch

        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            _, last_value = model(x)
            last_value = last_value.squeeze(0)

        advantages, returns = compute_gae(rewards, dones, values, last_value)
        advantages = normalize_adv(advantages)
        loss = ppo_update(
            model, optimizer, states, actions, old_log_probs, returns, advantages
        )

        if update % LOG_EVERY == 0 and len(reward_history) >= 5:
            avg_train = float(np.mean(reward_history[-20:]))
            print(
                f"update={update:4d}/{MAX_UPDATES}  loss={loss:.3f}  "
                f"train_avg20={avg_train:7.1f}  "
                f"std={model.log_std.exp().mean().item():.3f}"
            )

        if update % EVAL_EVERY == 0 or update == MAX_UPDATES:
            eval_mean = evaluate(env, model, EVAL_EPISODES)
            print(f"  eval_mean({EVAL_EPISODES})={eval_mean:7.1f}")
            if eval_mean > best_eval:
                best_eval = eval_mean
                torch.save(model.state_dict(), HERE / SAVE_BEST)
                print(f"  saved best -> {SAVE_BEST} (eval={best_eval:.1f})")
            if eval_mean >= SAVE_THRESHOLD and not reported:
                print(f"  reached threshold {SAVE_THRESHOLD} at update {update}")
                reported = True

    env.close()
    if not (HERE / SAVE_BEST).exists():
        torch.save(model.state_dict(), HERE / SAVE_BEST)
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
    ax.set_title("Pendulum-v1 continuous PPO")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / SAVE_CURVE, dpi=140)
    plt.close(fig)
    print(f"saved curve -> {SAVE_CURVE}; best_eval={best_eval:.1f}")


if __name__ == "__main__":
    main()
