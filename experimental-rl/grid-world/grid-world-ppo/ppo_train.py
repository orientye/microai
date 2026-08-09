"""CNN Actor-Critic PPO on randomized GridWorld with curriculum + distance shaping."""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from grid_world_env import GridWorldEnv

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
GRAD_CLIP = 0.5
ROLLOUT_STEPS = 2048
MINI_BATCH = 256
PPO_EPOCHS = 4
MAX_UPDATES = 300
EVAL_EVERY = 10
EVAL_LAYOUTS = 300
EVAL_SEED = 12345
SHAPE_COEF = 0.1
SAVE_BEST = "ppo_random_layout.pth"
SAVE_CURVE = "ppo_reward_history.png"
SEED = 0


class ActorCritic(nn.Module):
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
        self.shared = nn.Sequential(nn.Linear(flat, 128), nn.ReLU())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.view(-1, self.n_channels, self.grid_size, self.grid_size)
        h = self.shared(self.conv(x).flatten(1))
        return self.actor(h), self.critic(h).squeeze(-1)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[1]


@dataclass
class Transition:
    state: np.ndarray
    action: int
    log_prob: float
    reward: float
    done: float
    value: float
    legal: list[int]


def curriculum_obstacles(update: int) -> int:
    # Easy → hard over PPO updates (not env steps).
    if update <= 80:
        return 1
    if update <= 180:
        return 2
    return 3


def shaped_reward(
    env: GridWorldEnv,
    base_reward: float,
    prev_dist: int,
    next_dist: int,
    done: bool,
) -> float:
    max_dist = max(1, 2 * (env.size - 1))
    phi = -prev_dist / max_dist
    phi_next = 0.0 if done else (-next_dist / max_dist)
    return base_reward + SHAPE_COEF * (GAMMA * phi_next - phi)


def mask_logits(logits: torch.Tensor, legal: list[int]) -> torch.Tensor:
    masked = torch.full_like(logits, -1e9)
    idx = torch.as_tensor(legal, dtype=torch.int64, device=logits.device)
    masked[idx] = logits[idx]
    return masked


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_action(
    net: ActorCritic,
    state: np.ndarray,
    legal: list[int],
    *,
    greedy: bool = False,
) -> tuple[int, float, float]:
    with torch.no_grad():
        x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, value = net(x)
        logits = mask_logits(logits.squeeze(0), legal)
        if greedy:
            action = int(logits.argmax().item())
            dist = Categorical(logits=logits)
            log_prob = float(dist.log_prob(torch.tensor(action)).item())
        else:
            dist = Categorical(logits=logits)
            action_t = dist.sample()
            action = int(action_t.item())
            log_prob = float(dist.log_prob(action_t).item())
        return action, log_prob, float(value.item())


def evaluate(
    net: ActorCritic,
    env: GridWorldEnv,
    n_layouts: int,
    *,
    seed: int | None = EVAL_SEED,
) -> dict[str, float]:
    returns = []
    successes = 0
    for i in range(n_layouts):
        reset_seed = None if seed is None else seed + i
        state, info = env.reset(
            seed=reset_seed,
            options={"randomize_layout": True, "random_start": True, "n_obstacles": 3},
        )
        total = 0.0
        done = False
        reached = False
        while not done:
            action, _, _ = select_action(
                net,
                state,
                env.legal_actions(avoid_revisit=True),
                greedy=True,
            )
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            reached = terminated
            done = terminated or truncated
        returns.append(total)
        successes += int(reached)
    return {
        "mean_return": float(np.mean(returns)),
        "success_rate": successes / n_layouts,
        "successes": float(successes),
        "n_layouts": float(n_layouts),
        "min_return": float(np.min(returns)),
    }


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[float],
    last_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages: list[float] = []
    gae = 0.0
    next_value = last_value
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * next_value * nonterminal - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * nonterminal * gae
        advantages.insert(0, gae)
        next_value = values[t]
    adv = torch.as_tensor(advantages, dtype=torch.float32)
    ret = adv + torch.as_tensor(values, dtype=torch.float32)
    return adv, ret


def ppo_update(
    net: ActorCritic,
    optimizer: optim.Optimizer,
    batch: list[Transition],
    last_value: float,
) -> dict[str, float]:
    states = torch.as_tensor(np.array([t.state for t in batch]), dtype=torch.float32)
    actions = torch.as_tensor([t.action for t in batch], dtype=torch.int64)
    old_log_probs = torch.as_tensor([t.log_prob for t in batch], dtype=torch.float32)
    rewards = [t.reward for t in batch]
    values = [t.value for t in batch]
    dones = [t.done for t in batch]
    legals = [t.legal for t in batch]

    advantages, returns = compute_gae(rewards, values, dones, last_value)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(batch)
    idx = np.arange(n)
    total_policy = 0.0
    total_value = 0.0
    total_entropy = 0.0
    n_minibatches = 0

    for _ in range(PPO_EPOCHS):
        np.random.shuffle(idx)
        for start in range(0, n, MINI_BATCH):
            mb = idx[start : start + MINI_BATCH]
            mb_states = states[mb]
            mb_actions = actions[mb]
            mb_old_lp = old_log_probs[mb]
            mb_adv = advantages[mb]
            mb_ret = returns[mb]
            mb_legals = [legals[i] for i in mb]

            logits, values_pred = net(mb_states)
            # Per-sample action mask (legal set can differ).
            masked = []
            for i, legal in enumerate(mb_legals):
                masked.append(mask_logits(logits[i], legal))
            masked_logits = torch.stack(masked, dim=0)
            dist = Categorical(logits=masked_logits)
            new_lp = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_lp - mb_old_lp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_pred, mb_ret)
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            optimizer.step()

            total_policy += float(policy_loss.item())
            total_value += float(value_loss.item())
            total_entropy += float(entropy.item())
            n_minibatches += 1

    scale = max(1, n_minibatches)
    return {
        "policy_loss": total_policy / scale,
        "value_loss": total_value / scale,
        "entropy": total_entropy / scale,
    }


def collect_rollout(
    net: ActorCritic,
    env: GridWorldEnv,
    n_obs: int,
    n_steps: int,
) -> tuple[list[Transition], float, list[float]]:
    """Collect on-policy transitions; restart episodes as needed."""
    batch: list[Transition] = []
    episode_returns: list[float] = []
    state, info = env.reset(
        options={"randomize_layout": True, "random_start": True, "n_obstacles": n_obs}
    )
    prev_dist = int(info["manhattan"])
    ep_return = 0.0

    for _ in range(n_steps):
        legal = env.legal_actions()
        action, log_prob, value = select_action(net, state, legal, greedy=False)
        next_state, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        next_dist = int(step_info["manhattan"])
        train_r = shaped_reward(env, reward, prev_dist, next_dist, done)
        batch.append(
            Transition(
                state=state,
                action=action,
                log_prob=log_prob,
                reward=train_r,
                done=float(done),
                value=value,
                legal=legal,
            )
        )
        ep_return += reward
        state = next_state
        prev_dist = next_dist

        if done:
            episode_returns.append(ep_return)
            state, info = env.reset(
                options={
                    "randomize_layout": True,
                    "random_start": True,
                    "n_obstacles": n_obs,
                }
            )
            prev_dist = int(info["manhattan"])
            ep_return = 0.0

    with torch.no_grad():
        x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        last_value = float(net.value(x).item())
        # If the last transition already ended an episode, bootstrap with 0.
        if batch and batch[-1].done > 0.5:
            last_value = 0.0

    return batch, last_value, episode_returns


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
    ax.set_xlabel("episode (approx)")
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
    net = ActorCritic(env.size, env.action_space.n)
    optimizer = optim.Adam(net.parameters(), lr=LR)

    episode_returns: list[float] = []
    eval_curve: list[tuple[int, float]] = []
    best_score = -1.0
    ep_counter = 0

    print("Training CNN-PPO with curriculum + distance shaping...", flush=True)
    for update in range(1, MAX_UPDATES + 1):
        n_obs = curriculum_obstacles(update)
        batch, last_value, eps = collect_rollout(net, env, n_obs, ROLLOUT_STEPS)
        losses = ppo_update(net, optimizer, batch, last_value)
        episode_returns.extend(eps)
        ep_counter += len(eps)

        if update % EVAL_EVERY == 0 or update == MAX_UPDATES:
            recent = float(np.mean(eps)) if eps else float("nan")
            print(
                f"update={update:3d}  n_obs={n_obs}  episodes≈{ep_counter}  "
                f"rollout_return={recent:.3f}  "
                f"pi={losses['policy_loss']:.3f}  v={losses['value_loss']:.3f}  "
                f"H={losses['entropy']:.3f}",
                flush=True,
            )
            stats = evaluate(net, env, EVAL_LAYOUTS)
            eval_curve.append((ep_counter, stats["mean_return"]))
            score = stats["success_rate"] + 0.001 * stats["mean_return"]
            print(
                f"  eval_mean={stats['mean_return']:.3f}  "
                f"success={stats['success_rate']:.0%} "
                f"({int(stats['successes'])}/{int(stats['n_layouts'])})  "
                f"min={stats['min_return']:.3f}",
                flush=True,
            )
            if score >= best_score:
                best_score = score
                torch.save(net.state_dict(), SAVE_BEST)
                print(
                    f"  saved {SAVE_BEST} "
                    f"(success={stats['success_rate']:.0%}, mean={stats['mean_return']:.3f})",
                    flush=True,
                )

    if not os.path.exists(SAVE_BEST):
        torch.save(net.state_dict(), SAVE_BEST)

    _save_curve(episode_returns, eval_curve, "GridWorld CNN-PPO (curriculum + shaping)")


if __name__ == "__main__":
    main()
