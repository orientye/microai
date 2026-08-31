"""Self-play PPO with ADP reward (2^bombs). Grad clip 40 like DouZero."""

from __future__ import annotations

import argparse
import importlib.util
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
for _p in (SELFPLAY, ENV_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

spec = importlib.util.spec_from_file_location("doudizhu_selfplay", SELFPLAY / "ppo_train.py")
_sp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(_sp)

EVAL_EVERY = _sp.EVAL_EVERY
EVAL_EPISODES = _sp.EVAL_EPISODES
MAX_UPDATES = _sp.MAX_UPDATES
MIN_GAMES = _sp.MIN_GAMES
POSITIONS = _sp.POSITIONS
SAVE_THRESHOLD = _sp.SAVE_THRESHOLD
TripleModels = _sp.TripleModels
collect_games = _sp.collect_games
evaluate_landlord_vs_random = _sp.evaluate_landlord_vs_random
ppo_update_seat = _sp.ppo_update_seat

ADP_GRAD_CLIP = 40.0
SAVE_BEST = "ppo_adp.pth"
SAVE_CURVE = "reward_history.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_updates", type=int, default=MAX_UPDATES)
    parser.add_argument("--min_games", type=int, default=MIN_GAMES)
    args = parser.parse_args()

    from doudizhu_env import DoudizhuEnv

    env = DoudizhuEnv(objective="adp")
    models = TripleModels()
    opts = models.optimizers()
    wp_history: list[float] = []
    best_wp = -1.0

    for update in range(1, args.max_updates + 1):
        batch = collect_games(env, models, args.min_games)
        losses = [
            ppo_update_seat(
                models[pos], opts[pos], batch[pos], max_grad_norm=ADP_GRAD_CLIP
            )
            for pos in POSITIONS
        ]
        if update == 1 or update % EVAL_EVERY == 0 or update == args.max_updates:
            wp = evaluate_landlord_vs_random(env, models, EVAL_EPISODES)
            wp_history.append(wp)
            if wp > best_wp:
                best_wp = wp
                torch.save(models.state_dict(), HERE / SAVE_BEST)
            print(
                f"update {update}/{args.max_updates} "
                f"loss_ll={losses[0]:.3f} eval_wp={wp:.3f} best={best_wp:.3f}"
            )
            if wp >= SAVE_THRESHOLD:
                print(f"reached threshold WP>={SAVE_THRESHOLD}")
                break

    if not (HERE / SAVE_BEST).exists():
        torch.save(models.state_dict(), HERE / SAVE_BEST)
    if wp_history:
        plt.plot(wp_history)
        plt.xlabel("eval")
        plt.ylabel("landlord WP vs random")
        plt.title("Doudizhu ADP self-play")
        plt.tight_layout()
        plt.savefig(HERE / SAVE_CURVE)
        plt.close()


if __name__ == "__main__":
    main()
