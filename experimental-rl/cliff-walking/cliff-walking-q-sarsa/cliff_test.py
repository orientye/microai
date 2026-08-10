"""Evaluate saved Q-learning / SARSA tables on CliffWalking-v1.

Prints policies + greedy stats, then plays Gymnasium pixel render
(same style as the official cliff_walking.gif / CartPole human mode).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from cliff_env import evaluate_greedy, make_env, print_policy

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
Q_QL = HERE / "q_qlearning.npy"
Q_SARSA = HERE / "q_sarsa.npy"
N_RENDER_EPISODES = 3
STEP_SLEEP = 0.25  # cliff is short; slower than CartPole so steps are visible
GAP_SLEEP = 1.0


def load_q(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}; run cliff_train.py first.")
    return np.load(path)


def print_summary(q: np.ndarray, name: str) -> None:
    env = make_env()
    print_policy(q, name)
    stats = evaluate_greedy(env, q, n_episodes=50)
    env.close()
    print(
        f"{name} greedy eval (50 eps): success={stats['success_rate']:.0%}  "
        f"mean_return={stats['mean_return']:.1f}  min={stats['min_return']:.1f}"
    )
    print()


def render_greedy(q: np.ndarray, name: str, *, n_episodes: int = N_RENDER_EPISODES) -> None:
    # Same human window as official docs GIF (pygame pixel art).
    env = make_env(render_mode="human")
    print(f"=== {name}: human render ({n_episodes} eps) ===")
    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        total = 0.0
        steps = 0
        print(f"\n--- {name} episode {ep}/{n_episodes} ---")
        while True:
            action = int(np.argmax(q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            time.sleep(STEP_SLEEP)
            if terminated or truncated:
                mark = "goal" if terminated else "truncated"
                print(f"done ({mark}): return={total:.1f}  steps={steps}")
                time.sleep(GAP_SLEEP)
                break
    env.close()
    print()


def main() -> None:
    agents = [
        (load_q(Q_QL), "Q-learning"),
        (load_q(Q_SARSA), "SARSA"),
    ]
    for q, name in agents:
        print_summary(q, name)
    for q, name in agents:
        render_greedy(q, name)


if __name__ == "__main__":
    main()
