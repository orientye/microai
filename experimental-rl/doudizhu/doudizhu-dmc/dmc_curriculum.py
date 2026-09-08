"""DMC curriculum: landlord Q vs random farmers. Best on fixed deals."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

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
    LR,
    QNet,
    Replay,
    X_LANDLORD,
    collect_landlord_vs_random,
    dmc_update,
)
from dmc_agent import eval_landlord_vs_random_deals
from ruler_metrics import is_better

MAX_UPDATES = 80
MIN_GAMES = 16
EVAL_EVERY = 10
EVAL_DEALS = 40
REPLAY_BATCH = 256
UPDATE_EPOCHS = 4
SAVE_BEST = "dmc_landlord.pth"
SAVE_CURVE = "reward_history_curriculum.png"
EVAL_DATA = RULER / "eval_data.pkl"


def _load_eval_deals(n: int) -> list:
    if not EVAL_DATA.exists():
        raise SystemExit(f"missing {EVAL_DATA}; run generate_eval_data.py in eval-ruler")
    with EVAL_DATA.open("rb") as f:
        deals = pickle.load(f)
    return deals[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_updates", type=int, default=MAX_UPDATES)
    parser.add_argument("--min_games", type=int, default=MIN_GAMES)
    parser.add_argument("--eval_deals", type=int, default=EVAL_DEALS)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="explore rate; DouZero uses 0.01 at huge scale",
    )
    parser.add_argument("--init", type=str, default="")
    parser.add_argument("--no_early_stop", action="store_true")
    args = parser.parse_args()

    from doudizhu_env import DoudizhuEnv

    env = DoudizhuEnv(objective="adp")
    model = QNet(X_LANDLORD)
    if args.init:
        init_path = Path(args.init)
        if not init_path.exists():
            raise SystemExit(f"missing init {init_path}")
        model.load_state_dict(torch.load(init_path, map_location="cpu", weights_only=True))
    opt = optim.RMSprop(model.parameters(), lr=LR)
    replay = Replay()
    deals = _load_eval_deals(args.eval_deals)
    wp_history: list[float] = []
    adp_history: list[float] = []
    best_wp = -1.0
    best_adp = float("-inf")

    for update in range(1, args.max_updates + 1):
        fresh = collect_landlord_vs_random(env, model, args.min_games, args.epsilon)
        replay.add_many(fresh)
        loss = 0.0
        for _ in range(UPDATE_EPOCHS):
            loss = dmc_update(
                model, opt, replay.sample(REPLAY_BATCH), max_grad_norm=ADP_GRAD_CLIP
            )
        if update == 1 or update % EVAL_EVERY == 0 or update == args.max_updates:
            stats = eval_landlord_vs_random_deals(model, deals)
            model.train()
            wp, adp = stats["wp"], stats["adp"]
            wp_history.append(wp)
            adp_history.append(adp)
            if wp > best_wp or (wp == best_wp and adp > best_adp):
                best_wp, best_adp = wp, adp
                torch.save(model.state_dict(), HERE / SAVE_BEST)
            print(
                f"update {update}/{args.max_updates} loss={loss:.3f} "
                f"n={len(fresh)} replay={len(replay)} "
                f"eval_wp={wp:.3f} eval_adp={adp:.3f} "
                f"best_wp={best_wp:.3f} best_adp={best_adp:.3f}"
            )
            if (not args.no_early_stop) and is_better(wp, adp):
                print("reached ruler: WP>0.5 and ADP>0 vs random farmers")
                break

    if not (HERE / SAVE_BEST).exists():
        torch.save(model.state_dict(), HERE / SAVE_BEST)
    if wp_history:
        fig, ax1 = plt.subplots()
        ax1.plot(wp_history)
        ax1.set_xlabel("eval")
        ax1.set_ylabel("landlord WP vs random")
        ax2 = ax1.twinx()
        ax2.plot(adp_history, color="tab:orange")
        ax2.set_ylabel("landlord ADP vs random")
        fig.suptitle("DMC curriculum: landlord vs random")
        fig.tight_layout()
        fig.savefig(HERE / SAVE_CURVE)
        plt.close(fig)


if __name__ == "__main__":
    main()
