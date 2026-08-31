"""GameEnv-compatible greedy agents wrapping TripleModels."""

from __future__ import annotations

from pathlib import Path

import torch

from ppo_train import POSITIONS, TripleModels


class PpoSeatAgent:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def act(self, infoset):
        from douzero.env.env import get_obs

        legal = infoset.legal_actions
        if len(legal) == 1:
            return legal[0]
        obs = get_obs(infoset)
        idx, _lp, _v = self.model.act(obs, deterministic=True)
        return legal[idx]


def load_ppo_players(ckpt_path: str | Path) -> dict:
    path = Path(ckpt_path)
    models = TripleModels()
    state = torch.load(path, map_location="cpu", weights_only=True)
    models.load_state_dict(state)
    return {pos: PpoSeatAgent(models[pos]) for pos in POSITIONS}
