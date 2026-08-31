"""Legal-set PPO with a perfect-information critic (training only)."""

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

import importlib.util

HERE = Path(__file__).resolve().parent
ENV_DIR = HERE.parent / "doudizhu-env"
PPO_DIR = HERE.parent / "doudizhu-ppo"
DOUZERO_ROOT = HERE.parent / "DouZero"
for _p in (ENV_DIR, DOUZERO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_spec = importlib.util.spec_from_file_location(
    "doudizhu_legal_ppo", PPO_DIR / "ppo_train.py"
)
_base = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_base)

CLIP_EPS = _base.CLIP_EPS
ENT_COEF = _base.ENT_COEF
EVAL_EPISODES = _base.EVAL_EPISODES
EVAL_EVERY = _base.EVAL_EVERY
GAE_LAMBDA = _base.GAE_LAMBDA
GAMMA = _base.GAMMA
LR = _base.LR
MAX_UPDATES = _base.MAX_UPDATES
MIN_LANDLORD_STEPS = _base.MIN_LANDLORD_STEPS
SAVE_THRESHOLD = _base.SAVE_THRESHOLD
UPDATE_EPOCHS = _base.UPDATE_EPOCHS
VF_COEF = _base.VF_COEF
X_ACTION = _base.X_ACTION
X_STATE = _base.X_STATE
apply_opponent_terminal = _base.apply_opponent_terminal
compute_gae = _base.compute_gae
legal_logits = _base.legal_logits
LstmScorer = _base.LstmScorer
normalize_adv = _base.normalize_adv
pad_legal_batch = _base.pad_legal_batch

PERFECT_DIM = 162  # 3 positions × 54 card encoding
SAVE_BEST = "ppo_landlord_critic.pth"
SAVE_CURVE = "reward_history.png"


def encode_perfect_hands(all_handcards: dict) -> np.ndarray:
    """54-dim unary encoding per seat, concatenated. Actor never sees this."""
    from douzero.env.env import _cards2array

    chunks = [
        _cards2array(all_handcards["landlord"]),
        _cards2array(all_handcards["landlord_up"]),
        _cards2array(all_handcards["landlord_down"]),
    ]
    return np.concatenate(chunks).astype(np.float32)


class PerfectLegalAC(nn.Module):
    """Actor: imperfect (s,a) scores. Critic: imperfect public + three hands."""

    def __init__(self):
        super().__init__()
        self.actor_head = LstmScorer(X_ACTION)
        self.critic_head = LstmScorer(X_STATE + PERFECT_DIM)

    def value(self, obs: dict, perfect: torch.Tensor | np.ndarray) -> torch.Tensor:
        z = torch.as_tensor(obs["z"], dtype=torch.float32).unsqueeze(0)
        x_no = torch.as_tensor(obs["x_no_action"], dtype=torch.float32).unsqueeze(0)
        if isinstance(perfect, np.ndarray):
            perfect = torch.as_tensor(perfect, dtype=torch.float32)
        if perfect.dim() == 1:
            perfect = perfect.unsqueeze(0)
        return self.critic_head(z, torch.cat([x_no, perfect], dim=-1))

    @torch.no_grad()
    def act(
        self,
        obs: dict,
        perfect: np.ndarray | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        z = torch.as_tensor(obs["z"], dtype=torch.float32).unsqueeze(0)
        x = torch.as_tensor(obs["x_batch"], dtype=torch.float32).unsqueeze(0)
        mask = torch.ones(1, x.size(1), dtype=torch.bool)
        logits = legal_logits(self, z, x, mask)[0]
        dist = Categorical(logits=logits)
        idx = logits.argmax() if deterministic else dist.sample()
        value = 0.0
        if perfect is not None:
            value = float(self.value(obs, perfect).item())
        return int(idx.item()), float(dist.log_prob(idx).item()), value


def collect_episodes(
    env,
    model: PerfectLegalAC,
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
                perfect = encode_perfect_hands(env.all_handcards)
                idx, log_prob, value = model.act(obs, perfect)
                action = env.legal_actions[idx]
                next_obs, reward, done, _info = env.step(action)
                ep.append(
                    {
                        "z": torch.as_tensor(obs["z"], dtype=torch.float32),
                        "x_batch": torch.as_tensor(obs["x_batch"], dtype=torch.float32),
                        "x_no_action": torch.as_tensor(
                            obs["x_no_action"], dtype=torch.float32
                        ),
                        "perfect": torch.as_tensor(perfect, dtype=torch.float32),
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
    model: PerfectLegalAC,
    optimizer: optim.Optimizer,
    batch: list[dict],
) -> float:
    z_b, x_b, mask = pad_legal_batch(
        [s["z"] for s in batch],
        [s["x_batch"] for s in batch],
    )
    x_crit = torch.cat(
        [
            torch.stack([s["x_no_action"] for s in batch]),
            torch.stack([s["perfect"] for s in batch]),
        ],
        dim=-1,
    )
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
                torch.cat(
                    [
                        batch[-1]["x_no_action"].unsqueeze(0),
                        batch[-1]["perfect"].unsqueeze(0),
                    ],
                    dim=-1,
                ),
            ).squeeze(0)

    advantages, returns = compute_gae(rewards, dones, values, last_value)
    advantages = normalize_adv(advantages)

    loss_value = 0.0
    for _ in range(UPDATE_EPOCHS):
        logits = legal_logits(model, z_b, x_b, mask)
        dist = Categorical(logits=logits)
        new_lp = dist.log_prob(idx)
        entropy = dist.entropy().mean()
        v_pred = model.critic_head(z_b, x_crit)
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


def evaluate_wp(env, model: PerfectLegalAC, n_episodes: int) -> float:
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
    model = PerfectLegalAC()
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
        plt.title("Doudizhu PPO + perfect critic")
        plt.tight_layout()
        plt.savefig(HERE / SAVE_CURVE)
        plt.close()


if __name__ == "__main__":
    main()
