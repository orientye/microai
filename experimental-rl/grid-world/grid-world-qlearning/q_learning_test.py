"""Evaluate a trained Q-table with varied starts (not a fixed corner only)."""

from __future__ import annotations

import sys

import numpy as np

from grid_world_env import ACTION_ARROWS, GridWorldEnv
from q_learning_train import evaluate_all_starts, print_policy, run_greedy_episode

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

Q_PATH = "q_table.npy"
N_RANDOM_DEMOS = 5
RANDOM_SEED = 1


def main() -> None:
    q = np.load(Q_PATH)
    env = GridWorldEnv(render_mode="ansi")
    print_policy(env, q)
    print()

    # 1) Exhaustive: every free cell as start (deterministic, most persuasive).
    starts = env.free_cells(include_goal=False)
    print(f"=== All-start greedy eval ({len(starts)} starts) ===")
    rows = []
    for start in starts:
        ret, ok, steps = run_greedy_episode(env, q, start=start)
        rows.append((start, ret, ok, steps))
        mark = "OK" if ok else "FAIL"
        print(f"  start={start}  return={ret:.3f}  steps={steps:2d}  {mark}")

    stats = evaluate_all_starts(env, q)
    print(
        f"summary: success={stats['success_rate']:.0%}  "
        f"mean_return={stats['mean_return']:.3f}  "
        f"min_return={stats['min_return']:.3f}"
    )
    print()

    # 2) Random demos: sample starts and print paths (illustrative, not the score).
    rng = np.random.default_rng(RANDOM_SEED)
    demo_starts = [starts[i] for i in rng.choice(len(starts), size=N_RANDOM_DEMOS, replace=False)]
    print(f"=== Random-start path demos (seed={RANDOM_SEED}) ===")
    for i, start in enumerate(demo_starts, start=1):
        state, info = env.reset(options={"start": start})
        total = 0.0
        path = [info["start"]]
        done = False
        while not done:
            action = int(np.argmax(q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            path.append(env.index_to_state(state))
            done = terminated or truncated

        print(f"demo {i}: start={start}  return={total:.3f}  steps={len(path) - 1}")
        print("path:", " -> ".join(f"({r},{c})" for r, c in path))
        print(env.render())
        print(f"action at this start: {ACTION_ARROWS[int(np.argmax(q[env.state_to_index(*start)]))]}")
        print("-" * 40)


if __name__ == "__main__":
    main()
