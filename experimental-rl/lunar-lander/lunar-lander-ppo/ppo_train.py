"""Discrete PPO on LunarLander-v3 (Categorical policy + GAE)."""

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
from torch.distributions import Categorical

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

HERE = Path(__file__).resolve().parent

# ----------------- Hyperparameters -----------------
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
ROLLOUT_STEPS = 2048
UPDATE_EPOCHS = 10
MINIBATCH_SIZE = 64
MAX_UPDATES = 800
LOG_EVERY = 10
EVAL_EVERY = 20
EVAL_EPISODES = 20
SEED = 0
HID = 128

SAVE_THRESHOLD = 250.0
STREAK_NEEDED = 2  # consecutive evals at/above threshold before early stop
SAVE_BEST = "ppo_lunar_lander.pth"
SAVE_CURVE = "reward_history.png"


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, HID),
            nn.Tanh(),
            nn.Linear(HID, HID),
            nn.Tanh(),
            nn.Linear(HID, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, HID),
            nn.Tanh(),
            nn.Linear(HID, HID),
            nn.Tanh(),
            nn.Linear(HID, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(x), self.critic(x).squeeze(-1)

    def act(self, state: np.ndarray) -> tuple[int, float, float]:
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, value = self(x)
            dist = Categorical(logits=logits)
            action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())


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
            logits, values = model(states[mb])
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs[mb])
            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns[mb])
            loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            loss_value = float(loss.item())
    return loss_value


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
        rewards.append(float(reward))
        dones.append(float(done))

        episode_reward += float(reward)
        state = next_state
        if done:
            reward_history.append(episode_reward)
            episode_reward = 0.0
            state, _ = env.reset()

    batch = (
        torch.as_tensor(np.array(states), dtype=torch.float32),
        torch.as_tensor(actions, dtype=torch.long),
        torch.as_tensor(log_probs, dtype=torch.float32),
        torch.as_tensor(rewards, dtype=torch.float32),
        torch.as_tensor(dones, dtype=torch.float32),
        torch.as_tensor(values, dtype=torch.float32),
    )
    return batch, state, episode_reward


@torch.no_grad()
def evaluate(env: gym.Env, model: ActorCritic, n_episodes: int) -> float:
    totals: list[float] = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        ep = 0.0
        done = False
        while not done:
            x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, _ = model(x)
            action = int(torch.argmax(logits, dim=-1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            ep += float(reward)
            done = bool(terminated or truncated)
        totals.append(ep)
    return float(np.mean(totals))


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make("LunarLander-v3")
    model = ActorCritic(
        state_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.n),
    )
    optimizer = optim.Adam(model.parameters(), lr=LR)

    state, _ = env.reset(seed=SEED)
    episode_reward = 0.0
    reward_history: list[float] = []
    best_eval = float("-inf")
    reported = False
    streak = 0

    print("Training discrete PPO on LunarLander-v3...")
    for update in range(1, MAX_UPDATES + 1):
        batch, state, episode_reward = collect_rollout(
            env, model, state, episode_reward, reward_history
        )
        states, actions, old_log_probs, rewards, dones, values = batch

        with torch.no_grad():
            _, last_value = model(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
            last_value = last_value.squeeze(0)

        advantages, returns = compute_gae(rewards, dones, values, last_value)
        advantages = normalize_adv(advantages)
        loss = ppo_update(
            model, optimizer, states, actions, old_log_probs, returns, advantages
        )

        if update % LOG_EVERY == 0 and len(reward_history) >= 10:
            avg_train = float(np.mean(reward_history[-20:]))
            print(
                f"update={update:4d}/{MAX_UPDATES}  "
                f"train_avg20={avg_train:7.1f}  loss={loss:.3f}"
            )

        if update % EVAL_EVERY == 0 or update == MAX_UPDATES:
            eval_mean = evaluate(env, model, EVAL_EPISODES)
            print(f"  eval_mean({EVAL_EPISODES})={eval_mean:7.1f}")
            if eval_mean > best_eval:
                best_eval = eval_mean
                torch.save(model.state_dict(), HERE / SAVE_BEST)
                print(f"  saved best -> {SAVE_BEST} (eval={best_eval:.1f})")
            if eval_mean >= SAVE_THRESHOLD:
                streak += 1
                print(f"  threshold streak {streak}/{STREAK_NEEDED}")
                if streak >= STREAK_NEEDED and not reported:
                    print(f"  reached threshold {SAVE_THRESHOLD} at update {update}")
                    reported = True
                    break
            else:
                streak = 0

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
        SAVE_THRESHOLD,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"threshold {SAVE_THRESHOLD}",
    )
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title("LunarLander-v3 discrete PPO")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / SAVE_CURVE, dpi=140)
    plt.close(fig)
    print(f"saved curve -> {SAVE_CURVE}; best_eval={best_eval:.1f}")


if __name__ == "__main__":
    main()
