"""Seat-swap eval: our DMC trio vs random / DouZero ckpt dir."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RULER = HERE.parent / "eval-ruler"
ADP = HERE.parent / "doudizhu-adp"
for _p in (HERE, RULER, ADP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dmc_agent import load_dmc_players
from eval_ruler import (
    _load_players,
    evaluate_seat_swap_players,
    format_report,
    resolve_roles,
)


def load_side(spec: str, dmc_ckpt: Path | None) -> dict:
    if spec == "dmc":
        if dmc_ckpt is None or not dmc_ckpt.exists():
            raise SystemExit(f"need --dmc checkpoint for side dmc, got {dmc_ckpt}")
        return load_dmc_players(dmc_ckpt)
    roles = resolve_roles(spec)
    return _load_players(roles["landlord"], roles["landlord_up"], roles["landlord_down"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side_a", default="dmc", help="dmc | random | rlcard | ckpt dir")
    parser.add_argument("--side_b", default="random")
    parser.add_argument("--dmc", default=str(HERE / "dmc_adp.pth"))
    parser.add_argument("--eval_data", default=str(RULER / "eval_data.pkl"))
    parser.add_argument("--max_deals", type=int, default=50)
    args = parser.parse_args()

    eval_path = Path(args.eval_data)
    if not eval_path.exists():
        raise SystemExit(f"missing {eval_path}")

    if args.side_b not in ("random", "rlcard", "dmc"):
        bdir = Path(args.side_b)
        if not bdir.is_dir() or not (bdir / "landlord.ckpt").exists():
            raise SystemExit(f"DouZero weights not found at {bdir}")

    with eval_path.open("rb") as f:
        deals = pickle.load(f)
    if args.max_deals > 0:
        deals = deals[: args.max_deals]

    players_a = load_side(args.side_a, Path(args.dmc))
    players_b = load_side(args.side_b, Path(args.dmc))
    out = evaluate_seat_swap_players(
        players_a,
        players_b,
        deals,
        label_a=args.side_a,
        label_b=str(args.side_b),
    )
    print(format_report(out))
    better = out["wp_a"] > 0.5 and out["adp_a"] > 0
    print("A better on this ruler:", better)


if __name__ == "__main__":
    main()
