"""Behavior-clone PPO actors from frozen DouZero-ADP expert play."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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

POSITIONS = _sp.POSITIONS
UPDATE_EPOCHS = _sp.UPDATE_EPOCHS
legal_logits = _sp.legal_logits
pad_legal_batch = _sp.pad_legal_batch


def expert_action_index(legal: list, action) -> int:
    for i, a in enumerate(legal):
        if a == action:
            return i
    raise ValueError("expert action is not in the legal set")


def collect_expert_games(env, dz: dict, min_games: int) -> dict:
    """DouZero plays all seats; store (obs, expert legal index) per position."""
    pooled = {p: [] for p in POSITIONS}
    for _ in range(min_games):
        obs = env.reset()
        done = False
        while not done:
            pos = env.position
            legal = env.legal_actions
            action = dz[pos].act(env._env.infoset)
            idx = expert_action_index(legal, action)
            pooled[pos].append(
                {
                    "z": torch.as_tensor(obs["z"], dtype=torch.float32),
                    "x_batch": torch.as_tensor(obs["x_batch"], dtype=torch.float32),
                    "action_idx": idx,
                }
            )
            obs, _reward, done, _info = env.step(action)
    return pooled


def clone_update_seat(model, optimizer, batch: list[dict]) -> float:
    if len(batch) < 1:
        return 0.0
    z_b, x_b, mask = pad_legal_batch(
        [s["z"] for s in batch],
        [s["x_batch"] for s in batch],
    )
    idx = torch.tensor([s["action_idx"] for s in batch], dtype=torch.long)
    loss_value = 0.0
    for _ in range(UPDATE_EPOCHS):
        logits = legal_logits(model, z_b, x_b, mask)
        loss = F.cross_entropy(logits, idx)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        loss_value = float(loss.item())
    return loss_value
