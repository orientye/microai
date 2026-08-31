"""Thin Dou Dizhu rule wrapper over DouZero GameEnv. No training loop."""

from __future__ import annotations

import copy
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DOUZERO_ROOT = HERE.parent / "DouZero"
if str(DOUZERO_ROOT) not in sys.path:
    sys.path.insert(0, str(DOUZERO_ROOT))

FULL_DECK: list[int] = []
for _rank in range(3, 15):
    FULL_DECK.extend([_rank] * 4)
FULL_DECK.extend([17] * 4)
FULL_DECK.extend([20, 30])


def make_deal(landlord_cards: list[int]) -> dict:
    """Build a full deal from a 20-card landlord hand. Farmers get the rest."""
    if len(landlord_cards) != 20:
        raise ValueError(f"landlord hand must have 20 cards, got {len(landlord_cards)}")
    pool = Counter(FULL_DECK)
    pool.subtract(Counter(landlord_cards))
    if any(n < 0 for n in pool.values()):
        raise ValueError("landlord hand is not a subset of one deck")
    rest: list[int] = []
    for rank in sorted(pool):
        rest.extend([rank] * pool[rank])
    if len(rest) != 34:
        raise ValueError("deal does not use exactly 54 cards")
    return {
        "landlord": sorted(landlord_cards),
        "landlord_up": sorted(rest[:17]),
        "landlord_down": sorted(rest[17:]),
        "three_landlord_cards": sorted(landlord_cards[:3]),
    }


class DoudizhuEnv:
    """Gym-like step API. legal_actions / get_obs / all_handcards for PPO later."""

    def __init__(self, objective: str = "wp"):
        if objective not in ("wp", "adp", "logadp"):
            raise ValueError(f"unknown objective {objective!r}")
        from douzero.env.env import Env

        self.objective = objective
        self._env = Env(objective)

    def reset(self, deal: dict | None = None, seed: int | None = None) -> dict:
        from douzero.env.env import get_obs

        if deal is None:
            if seed is not None:
                np.random.seed(seed)
            return self._env.reset()

        data = copy.deepcopy(deal)
        for key in data:
            data[key] = list(data[key])
        self._env._env.reset()
        self._env._env.card_play_init(data)
        self._env.infoset = self._env._game_infoset
        return get_obs(self._env.infoset)

    @property
    def legal_actions(self) -> list:
        return list(self._env.infoset.legal_actions)

    @property
    def position(self) -> str:
        return self._env.infoset.player_position

    def get_obs(self) -> dict:
        from douzero.env.env import get_obs

        return get_obs(self._env.infoset)

    @property
    def all_handcards(self) -> dict:
        """Perfect information: three hands. Actor must not use this at play time."""
        return copy.deepcopy(self._env.infoset.all_handcards)

    def step(self, action) -> tuple:
        return self._env.step(action)
