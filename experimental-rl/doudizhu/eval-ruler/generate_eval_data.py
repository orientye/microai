"""Reproducible Dou Dizhu eval decks (same layout as DouZero generate_eval_data.py)."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

FULL_DECK = []
for i in range(3, 15):
    FULL_DECK.extend([i for _ in range(4)])
FULL_DECK.extend([17 for _ in range(4)])
FULL_DECK.extend([20, 30])


def generate_one_deal(rng: np.random.Generator) -> dict:
    deck = FULL_DECK.copy()
    rng.shuffle(deck)
    card_play_data = {
        "landlord": deck[:20],
        "landlord_up": deck[20:37],
        "landlord_down": deck[37:54],
        "three_landlord_cards": deck[17:20],
    }
    for key in card_play_data:
        card_play_data[key] = sorted(card_play_data[key])
    return card_play_data


def generate_deals(num_games: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [generate_one_deal(rng) for _ in range(num_games)]


def save_deals(path: Path, deals: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(deals, f, pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed eval decks")
    parser.add_argument("--output", default=str(HERE / "eval_data.pkl"))
    parser.add_argument("--num_games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    deals = generate_deals(args.num_games, args.seed)
    out = Path(args.output)
    if out.suffix != ".pkl":
        out = out.with_suffix(".pkl")
    save_deals(out, deals)
    print(f"wrote {len(deals)} deals seed={args.seed} -> {out}")


if __name__ == "__main__":
    main()
