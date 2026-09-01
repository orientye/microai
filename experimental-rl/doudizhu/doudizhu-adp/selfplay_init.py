"""Init self-play from the landlord curriculum checkpoint; trio ruler eval."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RULER = HERE.parent / "eval-ruler"
SELFPLAY = HERE.parent / "doudizhu-ppo-selfplay"
for _p in (RULER, SELFPLAY, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_ruler import _load_players, evaluate_seat_swap_players
from ppo_agent import PpoSeatAgent

POSITIONS = ("landlord", "landlord_up", "landlord_down")


def load_landlord_curriculum(models, ckpt_path: str | Path) -> None:
    state = torch.load(Path(ckpt_path), map_location="cpu", weights_only=True)
    models["landlord"].load_state_dict(state)


def set_landlord_trainable(models, trainable: bool) -> None:
    for p in models["landlord"].parameters():
        p.requires_grad = trainable


def eval_trio_vs_random_deals(models, deals: list) -> dict:
    players_a = {pos: PpoSeatAgent(models[pos]) for pos in POSITIONS}
    rnd = _load_players("random", "random", "random")
    out = evaluate_seat_swap_players(
        players_a,
        rnd,
        deals,
        label_a="ppo",
        label_b="random",
    )
    for pos in POSITIONS:
        models[pos].train()
    return out
