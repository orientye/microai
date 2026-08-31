"""Legal-set PPO: landlord only, random farmers, WP reward."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

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
ENV_DIR = HERE.parent / "doudizhu-env"
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

LR = 3e-4
GAMMA = 1.0
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
HID = 256
LSTM_HID = 128
UPDATE_EPOCHS = 4
MIN_LANDLORD_STEPS = 256
MAX_UPDATES = 80
EVAL_EVERY = 10
EVAL_EPISODES = 40
SAVE_THRESHOLD = 0.42
SAVE_BEST = "ppo_landlord.pth"
SAVE_CURVE = "reward_history.png"
X_ACTION = 373
X_STATE = 319
Z_STEPS = 5
Z_DIM = 162


class LstmScorer(nn.Module):
    """LSTM over move history, then MLP on (h, x) -> scalar."""

    def __init__(self, x_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(Z_DIM, LSTM_HID, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(x_dim + LSTM_HID, HID),
            nn.ReLU(),
            nn.Linear(HID, HID),
            nn.ReLU(),
            nn.Linear(HID, 1),
        )

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(z)
        h = out[:, -1]
        return self.mlp(torch.cat([h, x], dim=-1)).squeeze(-1)


class LegalActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_head = LstmScorer(X_ACTION)
        self.critic_head = LstmScorer(X_STATE)

    @torch.no_grad()
    def act(self, obs: dict, *, deterministic: bool = False) -> tuple[int, float, float]:
        z = torch.as_tensor(obs["z"], dtype=torch.float32).unsqueeze(0)
        x = torch.as_tensor(obs["x_batch"], dtype=torch.float32).unsqueeze(0)
        mask = torch.ones(1, x.size(1), dtype=torch.bool)
        logits = legal_logits(self, z, x, mask)[0]
        dist = Categorical(logits=logits)
        idx = logits.argmax() if deterministic else dist.sample()
        x_no = torch.as_tensor(obs["x_no_action"], dtype=torch.float32).unsqueeze(0)
        value = self.critic_head(z, x_no)
        return int(idx.item()), float(dist.log_prob(idx).item()), float(value.item())


def legal_logits(
    model: LegalActorCritic,
    z: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """z [B,5,162], x [B,K,373], mask [B,K] -> logits [B,K] with illegal = -1e9."""
    bsz, n_legal, x_dim = x.shape
    z_rep = z.unsqueeze(1).expand(bsz, n_legal, Z_STEPS, Z_DIM).reshape(
        bsz * n_legal, Z_STEPS, Z_DIM
    )
    logits = model.actor_head(z_rep, x.reshape(bsz * n_legal, x_dim)).view(bsz, n_legal)
    return logits.masked_fill(~mask, -1e9)


def pad_legal_batch(
    z_list: list[torch.Tensor],
    x_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(x_list)
    max_k = max(x.shape[0] for x in x_list)
    z_b = torch.stack(z_list, dim=0)
    x_b = x_list[0].new_zeros(n, max_k, x_list[0].shape[-1])
    mask = torch.zeros(n, max_k, dtype=torch.bool)
    for i, x in enumerate(x_list):
        k = x.shape[0]
        x_b[i, :k] = x
        mask[i, :k] = True
    return z_b, x_b, mask


def apply_opponent_terminal(steps: list[dict], reward: float) -> None:
    if not steps:
        return
    steps[-1]["reward"] = float(reward)
    steps[-1]["done"] = True


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


def collect_episodes(
    env,
    model: LegalActorCritic,
    min_landlord_steps: int,
    farmer: str = "random",
) -> list[dict]:
    if farmer != "random":
        raise ValueError("only random farmers in this lesson")
    collected: list[dict] = []
    while len(collected) < min_landlord_steps:
        obs = env.reset()
        ep: list[dict] = []
        done = False
        while not done:
            if env.position == "landlord":
                idx, log_prob, value = model.act(obs)
                action = env.legal_actions[idx]
                next_obs, reward, done, _info = env.step(action)
                ep.append(
                    {
                        "z": torch.as_tensor(obs["z"], dtype=torch.float32),
                        "x_batch": torch.as_tensor(obs["x_batch"], dtype=torch.float32),
                        "x_no_action": torch.as_tensor(
                            obs["x_no_action"], dtype=torch.float32
                        ),
                        "action_idx": idx,
                        "log_prob": log_prob,
                        "value": value,
                        "reward": float(reward),
                        "done": bool(done),
                    }
                )
                obs = next_obs
            else:
                action = random.choice(env.legal_actions)
                obs, reward, done, _info = env.step(action)
                if done:
                    apply_opponent_terminal(ep, float(reward))
        collected.extend(ep)
    return collected


def ppo_update(
    model: LegalActorCritic,
    optimizer: optim.Optimizer,
    batch: list[dict],
) -> float:
    z_b, x_b, mask = pad_legal_batch(
        [s["z"] for s in batch],
        [s["x_batch"] for s in batch],
    )
    x_no = torch.stack([s["x_no_action"] for s in batch])
    idx = torch.tensor([s["action_idx"] for s in batch], dtype=torch.long)
    old_lp = torch.tensor([s["log_prob"] for s in batch], dtype=torch.float32)
    rewards = torch.tensor([s["reward"] for s in batch], dtype=torch.float32)
    dones = torch.tensor([float(s["done"]) for s in batch], dtype=torch.float32)
    values = torch.tensor([s["value"] for s in batch], dtype=torch.float32)

    last_value = torch.tensor(0.0)
    if not batch[-1]["done"]:
        with torch.no_grad():
            last_value = model.critic_head(
                batch[-1]["z"].unsqueeze(0),
                batch[-1]["x_no_action"].unsqueeze(0),
            ).squeeze(0)

    advantages, returns = compute_gae(rewards, dones, values, last_value)
    advantages = normalize_adv(advantages)

    loss_value = 0.0
    for _ in range(UPDATE_EPOCHS):
        logits = legal_logits(model, z_b, x_b, mask)
        dist = Categorical(logits=logits)
        new_lp = dist.log_prob(idx)
        entropy = dist.entropy().mean()
        v_pred = model.critic_head(z_b, x_no)
        ratio = torch.exp(new_lp - old_lp)
        surrogate = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
        policy_loss = -torch.min(surrogate, clipped).mean()
        value_loss = F.mse_loss(v_pred, returns)
        loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        loss_value = float(loss.item())
    return loss_value


def evaluate_wp(env, model: LegalActorCritic, n_episodes: int) -> float:
    wins = 0
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        reward = 0.0
        while not done:
            if env.position == "landlord":
                idx, _lp, _v = model.act(obs, deterministic=True)
                obs, reward, done, _info = env.step(env.legal_actions[idx])
            else:
                obs, reward, done, _info = env.step(random.choice(env.legal_actions))
        if reward > 0:
            wins += 1
    return wins / n_episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_updates", type=int, default=MAX_UPDATES)
    parser.add_argument("--min_steps", type=int, default=MIN_LANDLORD_STEPS)
    args = parser.parse_args()

    from doudizhu_env import DoudizhuEnv

    env = DoudizhuEnv(objective="wp")
    model = LegalActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    wp_history: list[float] = []
    best_wp = -1.0

    for update in range(1, args.max_updates + 1):
        batch = collect_episodes(env, model, args.min_steps)
        loss = ppo_update(model, optimizer, batch)
        if update == 1 or update % EVAL_EVERY == 0 or update == args.max_updates:
            wp = evaluate_wp(env, model, EVAL_EPISODES)
            wp_history.append(wp)
            if wp > best_wp:
                best_wp = wp
                torch.save(model.state_dict(), HERE / SAVE_BEST)
            print(
                f"update {update}/{args.max_updates} loss={loss:.3f} "
                f"eval_wp={wp:.3f} best={best_wp:.3f} steps={len(batch)}"
            )
            if wp >= SAVE_THRESHOLD:
                print(f"reached threshold WP>={SAVE_THRESHOLD}")
                break

    if not (HERE / SAVE_BEST).exists():
        torch.save(model.state_dict(), HERE / SAVE_BEST)
    if wp_history:
        plt.plot(wp_history)
        plt.xlabel("eval")
        plt.ylabel("landlord WP vs random")
        plt.title("Doudizhu legal-set PPO")
        plt.tight_layout()
        plt.savefig(HERE / SAVE_CURVE)
        plt.close()


if __name__ == "__main__":
    main()
