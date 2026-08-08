"""Load a trained Q-table and roll out greedy episodes on GridWorld."""

from __future__ import annotations

import sys

import numpy as np

from grid_world_env import ACTION_ARROWS, GridWorldEnv
from q_learning_train import print_policy

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

Q_PATH = "q_table.npy"
N_EPISODES = 5


def main() -> None:
    q = np.load(Q_PATH)
    env = GridWorldEnv(render_mode="ansi")
    print_policy(env, q)
    print()

    for i in range(1, N_EPISODES + 1):
        state, _ = env.reset()
        total = 0.0
        path = [env.index_to_state(state)]
        done = False
        while not done:
            action = int(np.argmax(q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            path.append(env.index_to_state(state))
            done = terminated or truncated

        print(f"episode {i}: return={total:.3f}  steps={len(path) - 1}")
        print("path:", " -> ".join(f"({r},{c})" for r, c in path))
        print(env.render())
        print(f"final action preference at start: {ACTION_ARROWS[int(np.argmax(q[0]))]}")
        print("-" * 40)


if __name__ == "__main__":
    main()
