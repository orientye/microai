import os
import sys

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

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

LR = 3e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
ROLLOUT_STEPS = 128
UPDATE_EPOCHS = 4
MAX_UPDATES = 500
SAVE_BEST = "ppo_cartpole.pth"


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.network(x)


class Critic(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def act(self, state):
        """Return a sampled action, its log probability, and the state value."""
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, value = self(state_tensor)
            distribution = Categorical(logits=logits)
            action = distribution.sample()
        return action.item(), distribution.log_prob(action).item(), value.item()


def compute_returns(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """Discounted returns; no bootstrap through done steps: R_t = r_t + gamma * R_{t+1} * (1-done_t)."""
    returns = torch.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running * (0.0 if dones[t] > 0.5 else 1.0)
        returns[t] = running
    return returns


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
    """Compute generalized advantage estimates and bootstrapped returns."""
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


def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages):
    loss_value = 0.0
    for _ in range(UPDATE_EPOCHS):
        logits, values = model(states)
        distribution = Categorical(logits=logits)
        new_log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate = ratio * advantages
        clipped_surrogate = (
            torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
        )
        policy_loss = -torch.min(surrogate, clipped_surrogate).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        loss_value = loss.item()

    return loss_value


def collect_rollout(env, model, state, episode_reward, reward_history):
    """Fill ROLLOUT_STEPS; return tensors + updated (state, episode_reward)."""
    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

    for _ in range(ROLLOUT_STEPS):
        action, log_prob, value = model.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        dones.append(float(done))

        episode_reward += reward
        state = next_state
        if done:
            reward_history.append(episode_reward)
            episode_reward = 0.0
            state, _ = env.reset()

    batch = (
        torch.FloatTensor(np.array(states)),
        torch.LongTensor(actions),
        torch.FloatTensor(log_probs),
        torch.FloatTensor(rewards),
        torch.FloatTensor(dones),
        torch.FloatTensor(values),
    )
    return batch, state, episode_reward


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    model = ActorCritic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )
    optimizer = optim.Adam(model.parameters(), lr=LR)
    state, _ = env.reset()
    episode_reward = 0.0
    reward_history = []
    best_avg_reward = float("-inf")
    reported_threshold = False

    for update in range(1, MAX_UPDATES + 1):
        batch, state, episode_reward = collect_rollout(
            env, model, state, episode_reward, reward_history
        )
        states, actions, old_log_probs, rewards, dones, values = batch

        with torch.no_grad():
            _, last_value = model(torch.FloatTensor(state).unsqueeze(0))
            last_value = last_value.squeeze(0)

        advantages, returns = compute_gae(
            rewards, dones, values, last_value, GAMMA, GAE_LAMBDA
        )
        advantages = normalize_adv(advantages)
        loss = ppo_update(
            model,
            optimizer,
            states,
            actions,
            old_log_probs,
            returns,
            advantages,
        )

        if len(reward_history) >= 10:
            avg_reward = float(np.mean(reward_history[-10:]))
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(model.state_dict(), SAVE_BEST)
            if avg_reward >= 450 and not reported_threshold:
                print(
                    f"Reached avg10 >= 450 at update {update}: "
                    f"avg reward={avg_reward:.1f}"
                )
                reported_threshold = True
            if update % 25 == 0:
                print(
                    f"Update {update}/{MAX_UPDATES}, loss={loss:.3f}, "
                    f"avg reward={avg_reward:.1f}, best={best_avg_reward:.1f}"
                )

    env.close()
    if not os.path.exists(SAVE_BEST):
        torch.save(model.state_dict(), SAVE_BEST)

    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("CartPole PPO Reward History")
    plt.tight_layout()
    plt.savefig("reward_history.png")
    plt.close()
