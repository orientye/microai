"""ADP self-play first cut for public-feat PPO."""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

HERE = Path(__file__).resolve().parent
ENV_DIR = HERE.parent / "doudizhu-env"
RULER = HERE.parent / "eval-ruler"
ADP = HERE.parent / "doudizhu-adp"
DOUZERO = HERE.parent / "DouZero"
for _p in (HERE, ENV_DIR, RULER, ADP, DOUZERO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from doudizhu_env import DoudizhuEnv
from eval_ruler import _load_players
from feat_agent import eval_trio_vs_opponent_deals
from ppo_feat import POSITIONS, TripleModels, collect_games, ppo_update_seat

EVAL_DATA = RULER / "eval_data.pkl"
DZ_DIR = DOUZERO / "baselines" / "douzero_ADP"
SAVE_BEST = HERE / "ppo_feat.pth"
SAVE_LAST = HERE / "ppo_feat_last.pth"


def _load_eval_deals(n: int, start: int) -> list:
    if not EVAL_DATA.exists():
        raise SystemExit(f"missing {EVAL_DATA}; run generate_eval_data.py in eval-ruler")
    with EVAL_DATA.open("rb") as f:
        deals = pickle.load(f)
    return deals[start : start + n]


def save_last(models: TripleModels, opts: dict, update: int, best_wp: float, best_adp: float) -> None:
    torch.save(
        {
            "models": models.state_dict(),
            "optimizers": {p: opts[p].state_dict() for p in POSITIONS},
            "update": update,
            "best_wp": best_wp,
            "best_adp": best_adp,
        },
        SAVE_LAST,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_updates", type=int, default=200)
    parser.add_argument("--min_games", type=int, default=8)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--eval_start", type=int, default=800)
    parser.add_argument("--vs_douzero", action="store_true")
    args = parser.parse_args()

    env = DoudizhuEnv(objective="adp")
    models = TripleModels()
    opts = models.optimizers()
    best_wp, best_adp = -1.0, float("-inf")
    dz_ok = (DZ_DIR / "landlord.ckpt").exists()
    use_dz = args.vs_douzero and dz_ok
    if args.vs_douzero and not dz_ok:
        print("DouZero ADP missing; eval vs random only")

    mid_n = 20 if use_dz else 10
    mid_deals = _load_eval_deals(mid_n, args.eval_start)
    final_deals = _load_eval_deals(50, args.eval_start)

    for update in range(1, args.max_updates + 1):
        t0 = time.time()
        batch = collect_games(env, models, args.min_games)
        losses = [
            ppo_update_seat(models[pos], opts[pos], batch[pos]) for pos in POSITIONS
        ]
        save_last(models, opts, update, best_wp, best_adp)
        do_eval = update == 1 or update % args.eval_every == 0 or update == args.max_updates
        if not do_eval:
            print(
                f"update {update}/{args.max_updates} "
                f"loss_ll={losses[0]:.3f} n_ll={len(batch['landlord'])} "
                f"sec={time.time() - t0:.1f}"
            )
            continue
        rnd = _load_players("random", "random", "random")
        rnd_stats = eval_trio_vs_opponent_deals(
            models, mid_deals[:10], rnd, "random"
        )
        line = (
            f"update {update}/{args.max_updates} loss_ll={losses[0]:.3f} "
            f"eval_random_wp={rnd_stats['wp_a']:.3f}"
        )
        if use_dz:
            from vs_douzero import load_douzero_players

            dz = load_douzero_players(DZ_DIR)
            n = 50 if update == args.max_updates else 20
            stats = eval_trio_vs_opponent_deals(
                models, final_deals[:n], dz, "douzero"
            )
            wp, adp = stats["wp_a"], stats["adp_a"]
            if wp > best_wp or (wp == best_wp and adp > best_adp):
                best_wp, best_adp = wp, adp
                torch.save(models.state_dict(), SAVE_BEST)
            line += f" eval_douzero_wp={wp:.3f} eval_adp={adp:.3f} best_wp={best_wp:.3f}"
        else:
            if rnd_stats["wp_a"] > best_wp:
                best_wp = rnd_stats["wp_a"]
                torch.save(models.state_dict(), SAVE_BEST)
        print(line + f" sec={time.time() - t0:.1f}")

    if not SAVE_BEST.exists():
        torch.save(models.state_dict(), SAVE_BEST)


if __name__ == "__main__":
    main()
