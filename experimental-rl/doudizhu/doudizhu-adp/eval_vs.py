"""Seat-swap eval: our PPO trio vs random / rlcard / DouZero ckpt dir."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RULER = HERE.parent / "eval-ruler"
SELFPLAY = HERE.parent / "doudizhu-ppo-selfplay"
for _p in (RULER, SELFPLAY, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_ruler import (
    _load_players,
    evaluate_seat_swap_players,
    format_report,
    resolve_roles,
)
from ppo_agent import load_ppo_players


def load_side(spec: str, ppo_ckpt: Path | None) -> dict:
    if spec == "ppo":
        if ppo_ckpt is None or not ppo_ckpt.exists():
            raise SystemExit(f"need --ppo checkpoint for side ppo, got {ppo_ckpt}")
        return load_ppo_players(ppo_ckpt)
    roles = resolve_roles(spec)
    return _load_players(roles["landlord"], roles["landlord_up"], roles["landlord_down"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side_a", default="ppo", help="ppo | random | rlcard | ckpt dir")
    parser.add_argument("--side_b", default="random")
    parser.add_argument("--ppo", default=str(HERE / "ppo_adp.pth"))
    parser.add_argument(
        "--eval_data",
        default=str(RULER / "eval_data.pkl"),
    )
    parser.add_argument("--max_deals", type=int, default=50)
    args = parser.parse_args()

    ppo_ckpt = Path(args.ppo)
    eval_path = Path(args.eval_data)
    if not eval_path.exists():
        raise SystemExit(f"missing {eval_path}; run generate_eval_data.py in eval-ruler")

    if args.side_b not in ("random", "rlcard", "ppo"):
        bdir = Path(args.side_b)
        if not bdir.is_dir() or not (bdir / "landlord.ckpt").exists():
            raise SystemExit(
                f"DouZero weights not found at {bdir}. "
                "Download ADP ckpts into that folder (landlord/up/down.ckpt)."
            )

    with eval_path.open("rb") as f:
        deals = pickle.load(f)
    if args.max_deals > 0:
        deals = deals[: args.max_deals]

    players_a = load_side(args.side_a, ppo_ckpt)
    players_b = load_side(args.side_b, ppo_ckpt)
    out = evaluate_seat_swap_players(
        players_a,
        players_b,
        deals,
        label_a=args.side_a,
        label_b=str(args.side_b),
    )
    print(format_report(out))
    better = out["wp_a"] > 0.5 and out["adp_a"] > 0
    print("A better on this ruler:" , better)


if __name__ == "__main__":
    main()
