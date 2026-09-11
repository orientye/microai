"""GameEnv agents: get_obs + public feat. No perfect hands."""

from __future__ import annotations

from pathlib import Path

import torch

from feat import feat_from_infoset
from ppo_feat import POSITIONS, TripleModels


class FeatSeatAgent:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def act(self, infoset):
        from douzero.env.env import get_obs

        legal = infoset.legal_actions
        if len(legal) == 1:
            return legal[0]
        obs = get_obs(infoset)
        feat = feat_from_infoset(infoset)
        idx, _lp, _v = self.model.act(obs, feat, deterministic=True)
        return legal[idx]


def load_feat_players(ckpt_path: str | Path) -> dict:
    path = Path(ckpt_path)
    models = TripleModels()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["models"] if isinstance(payload, dict) and "models" in payload else payload
    models.load_state_dict(state)
    return {pos: FeatSeatAgent(models[pos]) for pos in POSITIONS}


def eval_trio_vs_opponent_deals(models, deals: list, players_b: dict, label_b: str) -> dict:
    import sys
    from pathlib import Path

    ruler = Path(__file__).resolve().parent.parent / "eval-ruler"
    if str(ruler) not in sys.path:
        sys.path.insert(0, str(ruler))
    from eval_ruler import evaluate_seat_swap_players

    players_a = {pos: FeatSeatAgent(models[pos]) for pos in POSITIONS}
    out = evaluate_seat_swap_players(
        players_a, players_b, deals, label_a="ppo-feat", label_b=label_b
    )
    for pos in POSITIONS:
        models[pos].train()
    return out
