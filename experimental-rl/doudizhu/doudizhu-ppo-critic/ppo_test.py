"""Greedy landlord vs random farmers (perfect critic unused at play time)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from ppo_train import PerfectLegalAC, evaluate_wp

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(HERE / "ppo_landlord_critic.pth"))
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    from doudizhu_env import DoudizhuEnv

    env = DoudizhuEnv(objective="wp")
    model = PerfectLegalAC()
    path = Path(args.ckpt)
    if not path.exists():
        raise SystemExit(f"missing {path}; run ppo_train.py first")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    wp = evaluate_wp(env, model, args.episodes)
    print(f"landlord WP vs random farmers ({args.episodes} games): {wp:.3f}")


if __name__ == "__main__":
    random.seed(0)
    main()
