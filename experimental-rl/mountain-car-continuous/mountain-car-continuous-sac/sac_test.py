"""Load SAC actor and drive MountainCarContinuous with human render."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from sac_train import GaussianActor, SAVE_BEST

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
NUM_EPISODES = 5
STEP_DELAY = 0.01


def load_actor(env: gym.Env) -> GaussianActor:
    path = HERE / SAVE_BEST
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}; run sac_train.py first.")
    actor = GaussianActor(
        state_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.shape[0]),
        action_high=float(env.action_space.high[0]),
    )
    actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    actor.eval()
    return actor


def main() -> None:
    try:
        env = gym.make("MountainCarContinuous-v0", render_mode="human")
        render_mode = "human"
    except Exception:
        print("注意：无法使用 render_mode=human，回退到无渲染。")
        env = gym.make("MountainCarContinuous-v0")
        render_mode = None

    actor = load_actor(env)
    print(f"Loaded {SAVE_BEST} (render_mode={render_mode!r})")

    for ep in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        total = 0.0
        steps = 0
        done = False
        while not done:
            action = actor.act(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            if render_mode == "human":
                time.sleep(STEP_DELAY)
            done = terminated or truncated
        print(f"Episode {ep}: return={total:.1f}  steps={steps}")

    env.close()


if __name__ == "__main__":
    main()
