"""Landlord-vs-random metrics on the fixed-deal eval ruler."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RULER = HERE.parent / "eval-ruler"
if str(RULER) not in sys.path:
    sys.path.insert(0, str(RULER))

from eval_ruler import _load_players, paper_adp, play_deals_with_players


def metrics_from_landlord_results(results: list[tuple[bool, int]]) -> dict:
    n = len(results)
    if n == 0:
        return {"games": 0, "wp": 0.0, "adp": 0.0}
    wins = sum(1 for won, _k in results if won)
    adp_sum = sum(paper_adp(won, k) for won, k in results)
    return {"games": n, "wp": wins / n, "adp": adp_sum / n}


def is_better(wp: float, adp: float) -> bool:
    return wp > 0.5 and adp > 0.0


def is_strong_vs_random(wp: float, adp: float) -> bool:
    return wp >= 0.90 and adp > 0.0


def make_landlord_vs_random_players(landlord_model) -> dict:
    from ppo_agent import PpoSeatAgent

    rnd = _load_players("random", "random", "random")
    return {
        "landlord": PpoSeatAgent(landlord_model),
        "landlord_up": rnd["landlord_up"],
        "landlord_down": rnd["landlord_down"],
    }


def eval_landlord_vs_random_deals(landlord_model, deals: list) -> dict:
    players = make_landlord_vs_random_players(landlord_model)
    results = play_deals_with_players(deals, players)
    out = metrics_from_landlord_results(results)
    out["num_deals"] = len(deals)
    return out
