"""DAgger: student (PPO) walks, frozen DouZero labels the current legal row."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

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

from clone import expert_action_index


def collect_dagger_games(env, models, dz: dict, min_games: int) -> dict:
    """PPO steps; store (obs, DouZero legal index) on the student's states."""
    pooled = {p: [] for p in POSITIONS}
    for _ in range(min_games):
        obs = env.reset()
        done = False
        while not done:
            pos = env.position
            legal = env.legal_actions
            teacher_action = dz[pos].act(env._env.infoset)
            idx = expert_action_index(legal, teacher_action)
            student_idx, _lp, _v = models[pos].act(obs, deterministic=False)
            pooled[pos].append(
                {
                    "z": torch.as_tensor(obs["z"], dtype=torch.float32),
                    "x_batch": torch.as_tensor(obs["x_batch"], dtype=torch.float32),
                    "action_idx": idx,
                }
            )
            obs, _reward, done, _info = env.step(legal[student_idx])
    return pooled
