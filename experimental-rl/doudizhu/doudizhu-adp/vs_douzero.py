"""Play frozen official DouZero; Monte-Carlo return is the whole-episode G."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

HERE = Path(__file__).resolve().parent
SELFPLAY = HERE.parent / "doudizhu-ppo-selfplay"
RULER = HERE.parent / "eval-ruler"
DOUZERO = HERE.parent / "DouZero"
for _p in (RULER, SELFPLAY, DOUZERO, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_spec = importlib.util.spec_from_file_location("doudizhu_selfplay", SELFPLAY / "ppo_train.py")
_sp = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_sp)

CLIP_EPS = _sp.CLIP_EPS
ENT_COEF = _sp.ENT_COEF
UPDATE_EPOCHS = _sp.UPDATE_EPOCHS
VF_COEF = _sp.VF_COEF
POSITIONS = _sp.POSITIONS
encode_perfect_hands = _sp.encode_perfect_hands
legal_logits = _sp.legal_logits
normalize_adv = _sp.normalize_adv
pad_legal_batch = _sp.pad_legal_batch

FARMER_SEATS = ("landlord_up", "landlord_down")
DEFAULT_DZ_DIR = DOUZERO / "baselines" / "douzero_ADP"


def assign_mc_returns(bufs: dict, landlord_g: float) -> None:
    """Every step of a seat carries the same terminal G (farmers get -G)."""
    for pos, steps in bufs.items():
        g = float(landlord_g) if pos == "landlord" else -float(landlord_g)
        for step in steps:
            step["mc_return"] = g
            step["reward"] = 0.0
            step["done"] = False
        if steps:
            steps[-1]["reward"] = g
            steps[-1]["done"] = True


def we_control(pos: str, ours: str) -> bool:
    if ours == "landlord":
        return pos == "landlord"
    if ours == "farmers":
        return pos in FARMER_SEATS
    raise ValueError(f"ours must be landlord or farmers, got {ours!r}")


def load_douzero_players(dz_dir: str | Path) -> dict:
    from eval_ruler import _load_players

    d = Path(dz_dir)
    return _load_players(
        str(d / "landlord.ckpt"),
        str(d / "landlord_up.ckpt"),
        str(d / "landlord_down.ckpt"),
    )


def collect_vs_douzero(
    env,
    models,
    dz: dict,
    min_games: int,
    ours: str = "mix",
) -> dict:
    pooled = {p: [] for p in POSITIONS}
    for i in range(min_games):
        role = ours
        if ours == "mix":
            role = "landlord" if i % 2 == 0 else "farmers"
        _play_one(env, models, dz, role, pooled)
    return pooled


def _play_one(env, models, dz: dict, ours: str, pooled: dict) -> None:
    obs = env.reset()
    ep = {p: [] for p in POSITIONS}
    done = False
    reward = 0.0
    while not done:
        pos = env.position
        if we_control(pos, ours):
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
                    "mc_return": 0.0,
                }
            )
            obs = next_obs
        else:
            action = dz[pos].act(env._env.infoset)
            obs, reward, done, _info = env.step(action)
    assign_mc_returns(ep, landlord_g=float(reward))
    for p in POSITIONS:
        pooled[p].extend(ep[p])


def ppo_update_seat_mc(
    model,
    optimizer,
    batch: list[dict],
    max_grad_norm: float = 40.0,
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
    values = torch.tensor([s["value"] for s in batch], dtype=torch.float32)
    returns = torch.tensor([s["mc_return"] for s in batch], dtype=torch.float32)
    advantages = normalize_adv(returns - values)

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
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        loss_value = float(loss.item())
    return loss_value
