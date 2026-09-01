"""Self-play PPO with ADP. Landlord starts from curriculum; best on seat-swap ruler."""

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
MAX_UPDATES = _sp.MAX_UPDATES
POSITIONS = _sp.POSITIONS
TripleModels = _sp.TripleModels
collect_games = _sp.collect_games
ppo_update_seat = _sp.ppo_update_seat

from ruler_metrics import is_better
from selfplay_init import (
    eval_trio_vs_random_deals,
    load_landlord_curriculum,
    set_landlord_trainable,
)

ADP_GRAD_CLIP = 40.0
DEFAULT_MIN_GAMES = 32
DEFAULT_FREEZE_LANDLORD = 20
EVAL_DEALS = 40
EVAL_SLICE = (40, 80)
SAVE_BEST = "ppo_adp.pth"
SAVE_CURVE = "reward_history.png"
EVAL_DATA = RULER / "eval_data.pkl"
DEFAULT_LANDLORD = HERE / "ppo_adp_landlord.pth"


def _load_eval_deals(n: int, start: int) -> list:
    if not EVAL_DATA.exists():
        raise SystemExit(f"missing {EVAL_DATA}; run generate_eval_data.py in eval-ruler")
    with EVAL_DATA.open("rb") as f:
        deals = pickle.load(f)
    return deals[start : start + n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_updates", type=int, default=MAX_UPDATES)
    parser.add_argument("--min_games", type=int, default=DEFAULT_MIN_GAMES)
    parser.add_argument("--freeze_landlord", type=int, default=DEFAULT_FREEZE_LANDLORD)
    parser.add_argument("--landlord", type=str, default=str(DEFAULT_LANDLORD))
    parser.add_argument("--eval_deals", type=int, default=EVAL_DEALS)
    args = parser.parse_args()

    from doudizhu_env import DoudizhuEnv

    landlord_ckpt = Path(args.landlord)
    if not landlord_ckpt.exists():
        raise SystemExit(f"missing landlord curriculum {landlord_ckpt}")

    env = DoudizhuEnv(objective="adp")
    models = TripleModels()
    load_landlord_curriculum(models, landlord_ckpt)
    opts = models.optimizers()
    deals = _load_eval_deals(args.eval_deals, EVAL_SLICE[0])
    wp_history: list[float] = []
    adp_history: list[float] = []
    best_wp = -1.0
    best_adp = float("-inf")

    for update in range(1, args.max_updates + 1):
        freeze = update <= args.freeze_landlord
        set_landlord_trainable(models, trainable=not freeze)
        batch = collect_games(env, models, args.min_games)
        losses = []
        for pos in POSITIONS:
            if pos == "landlord" and freeze:
                losses.append(0.0)
                continue
            losses.append(
                ppo_update_seat(
                    models[pos], opts[pos], batch[pos], max_grad_norm=ADP_GRAD_CLIP
                )
            )
        if update == 1 or update % EVAL_EVERY == 0 or update == args.max_updates:
            stats = eval_trio_vs_random_deals(models, deals)
            wp, adp = stats["wp_a"], stats["adp_a"]
            wp_history.append(wp)
            adp_history.append(adp)
            if wp > best_wp or (wp == best_wp and adp > best_adp):
                best_wp, best_adp = wp, adp
                torch.save(models.state_dict(), HERE / SAVE_BEST)
            print(
                f"update {update}/{args.max_updates} "
                f"loss_ll={losses[0]:.3f} freeze_ll={int(freeze)} "
                f"eval_wp={wp:.3f} eval_adp={adp:.3f} "
                f"best_wp={best_wp:.3f} best_adp={best_adp:.3f}"
            )
            if (not freeze) and is_better(wp, adp):
                print("reached ruler: WP>0.5 and ADP>0 vs random (seat-swap)")
                break

    if not (HERE / SAVE_BEST).exists():
        torch.save(models.state_dict(), HERE / SAVE_BEST)
    if wp_history:
        fig, ax1 = plt.subplots()
        ax1.plot(wp_history, label="WP")
        ax1.set_xlabel("eval")
        ax1.set_ylabel("seat-swap WP vs random")
        ax2 = ax1.twinx()
        ax2.plot(adp_history, color="tab:orange", label="ADP")
        ax2.set_ylabel("seat-swap ADP vs random")
        fig.suptitle("ADP self-play from curriculum landlord")
        fig.tight_layout()
        fig.savefig(HERE / SAVE_CURVE)
        plt.close(fig)


if __name__ == "__main__":
    main()
