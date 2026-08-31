"""Seat-swap eval ruler: same decks, both roles, paper WP / ADP."""

from __future__ import annotations

import argparse
import copy
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOUZERO_ROOT = HERE.parent / "DouZero"
if str(DOUZERO_ROOT) not in sys.path:
    sys.path.insert(0, str(DOUZERO_ROOT))


def paper_adp(landlord_won: bool, bomb_num: int) -> int:
    """DouZero paper ADP from the landlord's view: ±2^k."""
    delta = 2 ** int(bomb_num)
    return delta if landlord_won else -delta


POSITIONS = ("landlord", "landlord_up", "landlord_down")


def resolve_roles(side: str) -> dict[str, str]:
    """Map a side spec to per-position agents.

    ``random`` / ``rlcard`` apply to all seats. A directory must contain
    ``landlord.ckpt``, ``landlord_up.ckpt``, ``landlord_down.ckpt``.
    """
    if side in ("random", "rlcard"):
        return {p: side for p in POSITIONS}
    path = Path(side)
    if path.is_dir():
        return {
            "landlord": str(path / "landlord.ckpt"),
            "landlord_up": str(path / "landlord_up.ckpt"),
            "landlord_down": str(path / "landlord_down.ckpt"),
        }
    raise ValueError(
        f"side must be 'random', 'rlcard', or a checkpoint directory, got {side!r}"
    )


def aggregate_seat_swap(
    seating_a_landlord: list[tuple[bool, int]],
    seating_b_landlord: list[tuple[bool, int]],
) -> dict:
    """Combine two seatings. A is landlord in the first list, farmer in the second.

    Each item is (landlord_won, bomb_num) for that deal.
    """
    if len(seating_a_landlord) != len(seating_b_landlord):
        raise ValueError("seatings must cover the same number of deals")
    n = len(seating_a_landlord)
    a_wins = 0
    a_adp = 0.0
    for won, k in seating_a_landlord:
        if won:
            a_wins += 1
        a_adp += paper_adp(won, k)
    for won, k in seating_b_landlord:
        if not won:
            a_wins += 1
        a_adp += -paper_adp(won, k)
    games = 2 * n
    return {
        "games": games,
        "wp_a": a_wins / games if games else 0.0,
        "adp_a": a_adp / games if games else 0.0,
        "wp_b": 1.0 - (a_wins / games) if games else 0.0,
        "adp_b": -a_adp / games if games else 0.0,
    }


def _load_players(landlord: str, landlord_up: str, landlord_down: str):
    from douzero.evaluation.simulation import load_card_play_models

    return load_card_play_models(
        {
            "landlord": landlord,
            "landlord_up": landlord_up,
            "landlord_down": landlord_down,
        }
    )


def play_deals(
    deals: list[dict],
    landlord: str,
    landlord_up: str,
    landlord_down: str,
) -> list[tuple[bool, int]]:
    """Play each deal once. Returns (landlord_won, bomb_num) per deal."""
    from douzero.env.game import GameEnv

    env = GameEnv(_load_players(landlord, landlord_up, landlord_down))
    return play_deals_with_players(deals, env.players)


def play_deals_with_players(
    deals: list[dict],
    players: dict,
) -> list[tuple[bool, int]]:
    """Same as play_deals, but agents are already constructed."""
    from douzero.env.game import GameEnv

    env = GameEnv(players)
    results: list[tuple[bool, int]] = []
    for deal in deals:
        env.card_play_init(copy.deepcopy(deal))
        while not env.game_over:
            env.step()
        landlord_won = env.winner == "landlord"
        results.append((landlord_won, int(env.bomb_num)))
        env.reset()
    return results


def evaluate_seat_swap_players(
    players_a: dict,
    players_b: dict,
    deals: list[dict],
    *,
    label_a: str,
    label_b: str,
) -> dict:
    seating_a = play_deals_with_players(
        deals,
        {
            "landlord": players_a["landlord"],
            "landlord_up": players_b["landlord_up"],
            "landlord_down": players_b["landlord_down"],
        },
    )
    seating_b = play_deals_with_players(
        deals,
        {
            "landlord": players_b["landlord"],
            "landlord_up": players_a["landlord_up"],
            "landlord_down": players_a["landlord_down"],
        },
    )
    out = aggregate_seat_swap(seating_a, seating_b)
    out["side_a"] = label_a
    out["side_b"] = label_b
    out["num_deals"] = len(deals)
    return out


def evaluate_seat_swap(
    side_a: str,
    side_b: str,
    deals: list[dict],
) -> dict:
    a = resolve_roles(side_a)
    b = resolve_roles(side_b)
    seating_a = play_deals(
        deals, a["landlord"], b["landlord_up"], b["landlord_down"]
    )
    seating_b = play_deals(
        deals, b["landlord"], a["landlord_up"], a["landlord_down"]
    )
    out = aggregate_seat_swap(seating_a, seating_b)
    out["side_a"] = side_a
    out["side_b"] = side_b
    out["num_deals"] = len(deals)
    return out


def format_report(out: dict) -> str:
    return (
        f"deals={out['num_deals']} games={out['games']} "
        f"(each deal played twice, roles swapped)\n"
        f"side A ({out['side_a']})  WP={out['wp_a']:.4f}  ADP={out['adp_a']:.4f}\n"
        f"side B ({out['side_b']})  WP={out['wp_b']:.4f}  ADP={out['adp_b']:.4f}\n"
        f"A is 'better' iff WP>0.5 and ADP>0 on this ruler."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Seat-swap Dou Dizhu eval")
    parser.add_argument("--side_a", default="random")
    parser.add_argument("--side_b", default="random")
    parser.add_argument(
        "--eval_data",
        default=str(HERE / "eval_data.pkl"),
        help="pickle from generate_eval_data.py",
    )
    parser.add_argument("--max_deals", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    with open(args.eval_data, "rb") as f:
        deals = pickle.load(f)
    if args.max_deals > 0:
        deals = deals[: args.max_deals]

    out = evaluate_seat_swap(args.side_a, args.side_b, deals)
    print(format_report(out))


if __name__ == "__main__":
    main()
