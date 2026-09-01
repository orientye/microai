"""ADP curriculum: landlord PPO vs random farmers. Best saved on fixed deals."""

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
import torch.optim as optim

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

HERE = Path(__file__).resolve().parent
CRITIC = HERE.parent / "doudizhu-ppo-critic"
ENV_DIR = HERE.parent / "doudizhu-env"
RULER = HERE.parent / "eval-ruler"
for _p in (HERE, CRITIC, ENV_DIR, RULER):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

spec = importlib.util.spec_from_file_location("doudizhu_perfect_ppo", CRITIC / "ppo_train.py")
_cr = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(_cr)

EVAL_EVERY = _cr.EVAL_EVERY
LR = _cr.LR
MAX_UPDATES = _cr.MAX_UPDATES
PerfectLegalAC = _cr.PerfectLegalAC
collect_episodes = _cr.collect_episodes
ppo_update = _cr.ppo_update

from ruler_metrics import eval_landlord_vs_random_deals, is_better

ADP_GRAD_CLIP = 40.0
EVAL_DEALS = 40
MIN_LANDLORD_STEPS = 512
SAVE_BEST = "ppo_adp_landlord.pth"
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
    parser.add_argument("--min_steps", type=int, default=MIN_LANDLORD_STEPS)
    parser.add_argument("--eval_deals", type=int, default=EVAL_DEALS)
    args = parser.parse_args()

    from doudizhu_env import DoudizhuEnv

    env = DoudizhuEnv(objective="adp")
    model = PerfectLegalAC()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    deals = _load_eval_deals(args.eval_deals)
    wp_history: list[float] = []
    adp_history: list[float] = []
    best_wp = -1.0
    best_adp = float("-inf")

    for update in range(1, args.max_updates + 1):
        batch = collect_episodes(env, model, args.min_steps)
        loss = ppo_update(model, optimizer, batch, max_grad_norm=ADP_GRAD_CLIP)
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
                f"eval_wp={wp:.3f} eval_adp={adp:.3f} "
                f"best_wp={best_wp:.3f} best_adp={best_adp:.3f} "
                f"steps={len(batch)}"
            )
            if is_better(wp, adp):
                print("reached ruler: WP>0.5 and ADP>0 vs random farmers")
                break

    if not (HERE / SAVE_BEST).exists():
        torch.save(model.state_dict(), HERE / SAVE_BEST)
    if wp_history:
        fig, ax1 = plt.subplots()
        ax1.plot(wp_history, label="WP")
        ax1.set_xlabel("eval")
        ax1.set_ylabel("landlord WP vs random")
        ax2 = ax1.twinx()
        ax2.plot(adp_history, color="tab:orange", label="ADP")
        ax2.set_ylabel("landlord ADP vs random")
        fig.suptitle("ADP curriculum: landlord vs random")
        fig.tight_layout()
        fig.savefig(HERE / SAVE_CURVE)
        plt.close(fig)


if __name__ == "__main__":
    main()
