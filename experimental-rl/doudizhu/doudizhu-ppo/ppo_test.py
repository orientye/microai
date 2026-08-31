"""Greedy landlord vs random farmers."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ENV_DIR = HERE.parent / "doudizhu-env"
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

from doudizhu_env import DoudizhuEnv
from ppo_train import LegalActorCritic, evaluate_wp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(HERE / "ppo_landlord.pth"))
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    env = DoudizhuEnv(objective="wp")
    model = LegalActorCritic()
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
