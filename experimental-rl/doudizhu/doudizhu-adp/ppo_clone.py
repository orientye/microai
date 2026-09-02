"""Clone PPO actors from DouZero-ADP expert trajectories. Saves ppo_adp_bc.pth."""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

HERE = Path(__file__).resolve().parent
SELFPLAY = HERE.parent / "doudizhu-ppo-selfplay"
ENV_DIR = HERE.parent / "doudizhu-env"
RULER = HERE.parent / "eval-ruler"
for _p in (HERE, SELFPLAY, ENV_DIR, RULER):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

spec = importlib.util.spec_from_file_location("doudizhu_selfplay", SELFPLAY / "ppo_train.py")
_sp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(_sp)

EVAL_EVERY = _sp.EVAL_EVERY
POSITIONS = _sp.POSITIONS
TripleModels = _sp.TripleModels

from clone import clone_update_seat, collect_expert_games
from selfplay_init import eval_trio_vs_douzero_deals
from vs_douzero import DEFAULT_DZ_DIR, load_douzero_players

MAX_UPDATES = 40
DEFAULT_MIN_GAMES = 16
EVAL_DEALS = 20
EVAL_SLICE_START = 800
SAVE_BEST = "ppo_adp_bc.pth"
SAVE_CURVE = "reward_history_bc.png"
EVAL_DATA = RULER / "eval_data.pkl"
DEFAULT_INIT = HERE / "ppo_adp.pth"


def _load_eval_deals(n: int, start: int) -> list:
    if not EVAL_DATA.exists():
        raise SystemExit(f"missing {EVAL_DATA}")
    with EVAL_DATA.open("rb") as f:
        deals = pickle.load(f)
    return deals[start : start + n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_updates", type=int, default=MAX_UPDATES)
    parser.add_argument("--min_games", type=int, default=DEFAULT_MIN_GAMES)
    parser.add_argument("--init", type=str, default=str(DEFAULT_INIT))
    parser.add_argument("--douzero", type=str, default=str(DEFAULT_DZ_DIR))
    parser.add_argument("--eval_deals", type=int, default=EVAL_DEALS)
    args = parser.parse_args()

    from doudizhu_env import DoudizhuEnv

    init_ckpt = Path(args.init)
    dz_dir = Path(args.douzero)
    if not init_ckpt.exists():
        raise SystemExit(f"missing init {init_ckpt}")
    if not (dz_dir / "landlord.ckpt").exists():
        raise SystemExit(f"missing DouZero ADP at {dz_dir}")

    env = DoudizhuEnv(objective="adp")
    models = TripleModels()
    models.load_state_dict(torch.load(init_ckpt, map_location="cpu", weights_only=True))
    opts = models.optimizers()
    dz = load_douzero_players(dz_dir)
    deals = _load_eval_deals(args.eval_deals, EVAL_SLICE_START)
    wp_history: list[float] = []
    best_wp = -1.0
    best_adp = float("-inf")

    for update in range(1, args.max_updates + 1):
        batch = collect_expert_games(env, dz, args.min_games)
        losses = [
            clone_update_seat(models[pos], opts[pos], batch[pos]) for pos in POSITIONS
        ]
        if update == 1 or update % EVAL_EVERY == 0 or update == args.max_updates:
            stats = eval_trio_vs_douzero_deals(models, deals, dz_dir)
            wp, adp = stats["wp_a"], stats["adp_a"]
            wp_history.append(wp)
            if wp > best_wp or (wp == best_wp and adp > best_adp):
                best_wp, best_adp = wp, adp
                torch.save(models.state_dict(), HERE / SAVE_BEST)
            print(
                f"update {update}/{args.max_updates} "
                f"loss_ll={losses[0]:.3f} n_ll={len(batch['landlord'])} "
                f"eval_wp={wp:.3f} eval_adp={adp:.3f} "
                f"best_wp={best_wp:.3f} best_adp={best_adp:.3f}"
            )

    if not (HERE / SAVE_BEST).exists():
        torch.save(models.state_dict(), HERE / SAVE_BEST)
    if wp_history:
        plt.plot(wp_history)
        plt.xlabel("eval")
        plt.ylabel("seat-swap WP vs DouZero-ADP")
        plt.title("Behavior clone from DouZero-ADP")
        plt.tight_layout()
        plt.savefig(HERE / SAVE_CURVE)
        plt.close()


if __name__ == "__main__":
    main()
