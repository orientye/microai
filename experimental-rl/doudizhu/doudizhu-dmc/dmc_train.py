"""Three-seat DMC self-play (ADP). Init landlord from curriculum if present."""

from __future__ import annotations

import argparse
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
ENV_DIR = HERE.parent / "doudizhu-env"
RULER = HERE.parent / "eval-ruler"
ADP = HERE.parent / "doudizhu-adp"
for _p in (HERE, ENV_DIR, RULER, ADP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dmc import (
    ADP_GRAD_CLIP,
    POSITIONS,
    Replay,
    TripleQ,
    collect_dmc_games,
    dmc_update,
    resolve_device,
)
from dmc_agent import eval_trio_vs_random_deals
from ruler_metrics import is_better

MAX_UPDATES = 80
MIN_GAMES = 16
EVAL_EVERY = 10
EVAL_DEALS = 40
EVAL_SLICE_START = 80
REPLAY_BATCH = 256
UPDATE_EPOCHS = 4
SAVE_BEST = "dmc_adp.pth"
SAVE_CURVE = "reward_history.png"
EVAL_DATA = RULER / "eval_data.pkl"
DEFAULT_INIT = HERE / "dmc_landlord.pth"


def _load_eval_deals(n: int, start: int) -> list:
    if not EVAL_DATA.exists():
        raise SystemExit(f"missing {EVAL_DATA}")
    with EVAL_DATA.open("rb") as f:
        deals = pickle.load(f)
    return deals[start : start + n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_updates", type=int, default=MAX_UPDATES)
    parser.add_argument("--min_games", type=int, default=MIN_GAMES)
    parser.add_argument("--eval_deals", type=int, default=EVAL_DEALS)
    parser.add_argument("--init", type=str, default=str(DEFAULT_INIT))
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--no_early_stop", action="store_true")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    from doudizhu_env import DoudizhuEnv

    device = resolve_device(args.device or None)
    print(f"device={device}")
    env = DoudizhuEnv(objective="adp")
    models = TripleQ()
    models.to(device)
    init_ckpt = Path(args.init)
    if init_ckpt.exists():
        state = torch.load(init_ckpt, map_location=device, weights_only=True)
        models["landlord"].load_state_dict(state)
    opts = models.optimizers()
    replays = {p: Replay() for p in POSITIONS}
    deals = _load_eval_deals(args.eval_deals, EVAL_SLICE_START)
    wp_history: list[float] = []
    adp_history: list[float] = []
    best_wp = -1.0
    best_adp = float("-inf")

    for update in range(1, args.max_updates + 1):
        fresh = collect_dmc_games(env, models, args.min_games, args.epsilon)
        losses = []
        for pos in POSITIONS:
            replays[pos].add_many(fresh[pos])
            loss = 0.0
            for _ in range(UPDATE_EPOCHS):
                loss = dmc_update(
                    models[pos],
                    opts[pos],
                    replays[pos].sample(REPLAY_BATCH),
                    max_grad_norm=ADP_GRAD_CLIP,
                )
            losses.append(loss)
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
                f"loss_ll={losses[0]:.3f} n_ll={len(fresh['landlord'])} "
                f"eval_wp={wp:.3f} eval_adp={adp:.3f} "
                f"best_wp={best_wp:.3f} best_adp={best_adp:.3f}"
            )
            if (not args.no_early_stop) and is_better(wp, adp):
                print("reached ruler: WP>0.5 and ADP>0 vs random")
                break

    if not (HERE / SAVE_BEST).exists():
        torch.save(models.state_dict(), HERE / SAVE_BEST)
    if wp_history:
        fig, ax1 = plt.subplots()
        ax1.plot(wp_history)
        ax1.set_xlabel("eval")
        ax1.set_ylabel("seat-swap WP vs random")
        ax2 = ax1.twinx()
        ax2.plot(adp_history, color="tab:orange")
        ax2.set_ylabel("seat-swap ADP vs random")
        fig.suptitle("DMC self-play vs random")
        fig.tight_layout()
        fig.savefig(HERE / SAVE_CURVE)
        plt.close(fig)


if __name__ == "__main__":
    main()
