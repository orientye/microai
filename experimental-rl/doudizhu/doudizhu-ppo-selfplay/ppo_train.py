"""Three-seat self-play PPO (WP). Farmers learn; target is -landlord G."""

from __future__ import annotations

import argparse
import importlib.util
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
PPO_DIR = HERE.parent / "doudizhu-ppo"
CRITIC_DIR = HERE.parent / "doudizhu-ppo-critic"
DOUZERO_ROOT = HERE.parent / "DouZero"
for _p in (ENV_DIR, DOUZERO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_base = _load("doudizhu_legal_ppo", PPO_DIR / "ppo_train.py")
_critic = _load("doudizhu_perfect_ppo", CRITIC_DIR / "ppo_train.py")

CLIP_EPS = _base.CLIP_EPS
ENT_COEF = _base.ENT_COEF
EVAL_EPISODES = _base.EVAL_EPISODES
EVAL_EVERY = _base.EVAL_EVERY
LR = _base.LR
MAX_UPDATES = _base.MAX_UPDATES
SAVE_THRESHOLD = _base.SAVE_THRESHOLD
UPDATE_EPOCHS = _base.UPDATE_EPOCHS
VF_COEF = _base.VF_COEF
X_ACTION = _base.X_ACTION
X_STATE = _base.X_STATE
compute_gae = _base.compute_gae
legal_logits = _base.legal_logits
LstmScorer = _base.LstmScorer
normalize_adv = _base.normalize_adv
pad_legal_batch = _base.pad_legal_batch
encode_perfect_hands = _critic.encode_perfect_hands
PERFECT_DIM = _critic.PERFECT_DIM

FARMER_X_ACTION = 484
FARMER_X_STATE = 430
POSITIONS = ("landlord", "landlord_up", "landlord_down")
MIN_GAMES = 8
SAVE_BEST = "ppo_selfplay.pth"
SAVE_CURVE = "reward_history.png"


class SeatAC(nn.Module):
    def __init__(self, x_action: int, x_state: int):
        super().__init__()
        self.actor_head = LstmScorer(x_action)
        self.critic_head = LstmScorer(x_state + PERFECT_DIM)

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


class TripleModels:
    def __init__(self):
        self.models = {
            "landlord": SeatAC(X_ACTION, X_STATE),
            "landlord_up": SeatAC(FARMER_X_ACTION, FARMER_X_STATE),
            "landlord_down": SeatAC(FARMER_X_ACTION, FARMER_X_STATE),
        }

    def __getitem__(self, position: str) -> SeatAC:
        return self.models[position]

    def optimizers(self) -> dict:
        return {
            p: optim.Adam(m.parameters(), lr=LR) for p, m in self.models.items()
        }

    def state_dict(self) -> dict:
        return {p: m.state_dict() for p, m in self.models.items()}

    def load_state_dict(self, state: dict) -> None:
        for p, sd in state.items():
            self.models[p].load_state_dict(sd)


def assign_episode_returns(bufs: dict, landlord_g: float) -> None:
    """Zero-sum WP: farmers share -G. Intermediate steps stay 0."""
    for pos, steps in bufs.items():
        g = float(landlord_g) if pos == "landlord" else -float(landlord_g)
        for step in steps:
            step["reward"] = 0.0
            step["done"] = False
        if steps:
            steps[-1]["reward"] = g
            steps[-1]["done"] = True


def collect_games(env, models: TripleModels, min_games: int) -> dict:
    pooled = {p: [] for p in POSITIONS}
    for _ in range(min_games):
        obs = env.reset()
        ep = {p: [] for p in POSITIONS}
        done = False
        reward = 0.0
        while not done:
            pos = env.position
            perfect = encode_perfect_hands(env.all_handcards)
            idx, log_prob, value = models[pos].act(obs, perfect)
            action = env.legal_actions[idx]
            next_obs, reward, done, _info = env.step(action)
            ep[pos].append(
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
                    "reward": 0.0,
                    "done": False,
                }
            )
            obs = next_obs
        assign_episode_returns(ep, landlord_g=float(reward))
        for p in POSITIONS:
            pooled[p].extend(ep[p])
    return pooled


def ppo_update_seat(
    model: SeatAC,
    optimizer: optim.Optimizer,
    batch: list[dict],
) -> float:
    if len(batch) < 2:
        return 0.0
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


def evaluate_landlord_vs_random(env, models: TripleModels, n_episodes: int) -> float:
    wins = 0
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        reward = 0.0
        while not done:
            if env.position == "landlord":
                idx, _lp, _v = models["landlord"].act(obs, deterministic=True)
                obs, reward, done, _info = env.step(env.legal_actions[idx])
            else:
                obs, reward, done, _info = env.step(random.choice(env.legal_actions))
        if reward > 0:
            wins += 1
    return wins / n_episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_updates", type=int, default=MAX_UPDATES)
    parser.add_argument("--min_games", type=int, default=MIN_GAMES)
    args = parser.parse_args()

    from doudizhu_env import DoudizhuEnv

    env = DoudizhuEnv(objective="wp")
    models = TripleModels()
    opts = models.optimizers()
    wp_history: list[float] = []
    best_wp = -1.0

    for update in range(1, args.max_updates + 1):
        batch = collect_games(env, models, args.min_games)
        losses = []
        for pos in POSITIONS:
            losses.append(ppo_update_seat(models[pos], opts[pos], batch[pos]))
        if update == 1 or update % EVAL_EVERY == 0 or update == args.max_updates:
            wp = evaluate_landlord_vs_random(env, models, EVAL_EPISODES)
            wp_history.append(wp)
            if wp > best_wp:
                best_wp = wp
                torch.save(models.state_dict(), HERE / SAVE_BEST)
            print(
                f"update {update}/{args.max_updates} "
                f"loss_ll={losses[0]:.3f} eval_wp={wp:.3f} best={best_wp:.3f} "
                f"n_ll={len(batch['landlord'])}"
            )
            if wp >= SAVE_THRESHOLD:
                print(f"reached threshold WP>={SAVE_THRESHOLD}")
                break

    if not (HERE / SAVE_BEST).exists():
        torch.save(models.state_dict(), HERE / SAVE_BEST)
    if wp_history:
        plt.plot(wp_history)
        plt.xlabel("eval")
        plt.ylabel("landlord WP vs random")
        plt.title("Doudizhu self-play PPO")
        plt.tight_layout()
        plt.savefig(HERE / SAVE_CURVE)
        plt.close()


if __name__ == "__main__":
    main()
