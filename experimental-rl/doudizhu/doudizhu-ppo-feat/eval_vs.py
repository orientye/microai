"""Seat-swap eval: feat PPO vs random / DouZero ckpt dir."""

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

from eval_ruler import (
    _load_players,
    evaluate_seat_swap_players,
    format_report,
    resolve_roles,
)
from feat_agent import load_feat_players


def load_side(spec: str, feat_ckpt: Path | None) -> dict:
    if spec == "feat":
        if feat_ckpt is None or not feat_ckpt.exists():
            raise SystemExit(f"need --feat checkpoint, got {feat_ckpt}")
        return load_feat_players(feat_ckpt)
    roles = resolve_roles(spec)
    return _load_players(roles["landlord"], roles["landlord_up"], roles["landlord_down"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side_a", default="feat")
    parser.add_argument("--side_b", default="random")
    parser.add_argument("--feat", default=str(HERE / "ppo_feat.pth"))
    parser.add_argument("--eval_data", default=str(RULER / "eval_data.pkl"))
    parser.add_argument("--max_deals", type=int, default=50)
    parser.add_argument("--eval_start", type=int, default=800)
    args = parser.parse_args()

    eval_path = Path(args.eval_data)
    if not eval_path.exists():
        raise SystemExit(f"missing {eval_path}; run generate_eval_data.py in eval-ruler")

    with eval_path.open("rb") as f:
        deals = pickle.load(f)
    deals = deals[args.eval_start :]
    if args.max_deals > 0:
        deals = deals[: args.max_deals]

    players_a = load_side(args.side_a, Path(args.feat))
    players_b = load_side(args.side_b, Path(args.feat))
    out = evaluate_seat_swap_players(
        players_a, players_b, deals, label_a=args.side_a, label_b=str(args.side_b)
    )
    print(format_report(out))
    print("A better on this ruler:", out["wp_a"] > 0.5 and out["adp_a"] > 0)


if __name__ == "__main__":
    main()
