"""Evaluate CNN-PPO on freshly sampled random seeds (not a fixed exam set)."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

from grid_world_env import GridWorldEnv
from ppo_train import ActorCritic, select_action

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

PPO_PATH = "ppo_random_layout.pth"
Q_TABLE_PATH = os.path.join("..", "grid-world-qlearning", "q_table.npy")
N_SEEDS = 5
LAYOUTS_PER_SEED = 200
N_DEMOS = 5


def rollout_ppo(
    env: GridWorldEnv,
    net: ActorCritic,
    *,
    seed: int,
) -> tuple[float, bool, dict, list[tuple[int, int]]]:
    state, info = env.reset(
        seed=seed,
        options={"randomize_layout": True, "random_start": True, "n_obstacles": 3},
    )
    path = [info["start"]]
    total = 0.0
    done = False
    reached = False
    while not done:
        action, _, _ = select_action(
            net,
            state,
            env.legal_actions(avoid_revisit=True),
            greedy=True,
        )
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        path.append(env.pos)
        reached = terminated
        done = terminated or truncated
    return total, reached, info, path


def eval_seed(env: GridWorldEnv, net: ActorCritic, seed: int, n_layouts: int) -> dict[str, float]:
    returns = []
    successes = 0
    for i in range(n_layouts):
        total, reached, _, _ = rollout_ppo(env, net, seed=seed + i)
        returns.append(total)
        successes += int(reached)
    return {
        "success_rate": successes / n_layouts,
        "successes": float(successes),
        "n_layouts": float(n_layouts),
        "mean_return": float(np.mean(returns)),
        "min_return": float(np.min(returns)),
    }


def eval_tabular_seed(env: GridWorldEnv, q: np.ndarray, seed: int, n_layouts: int) -> float:
    ok = 0
    for i in range(n_layouts):
        _, info = env.reset(
            seed=seed + i,
            options={"randomize_layout": True, "random_start": True, "n_obstacles": 3},
        )
        pos = info["start"]
        obstacles = set(info["obstacles"])
        steps = 0
        reached = False
        while steps < env.max_steps:
            state_id = pos[0] * env.size + pos[1]
            action = int(np.argmax(q[state_id]))
            dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
            nr = min(max(pos[0] + dr, 0), env.size - 1)
            nc = min(max(pos[1] + dc, 0), env.size - 1)
            if (nr, nc) not in obstacles:
                pos = (nr, nc)
            steps += 1
            if pos == env.goal:
                reached = True
                break
        ok += int(reached)
    return ok / n_layouts


def main() -> None:
    if not os.path.exists(PPO_PATH):
        raise SystemExit(f"missing {PPO_PATH}; run ppo_train.py first")

    env = GridWorldEnv(randomize_layout=True, n_obstacles=3, render_mode="ansi")
    net = ActorCritic(env.size, env.action_space.n)
    net.load_state_dict(torch.load(PPO_PATH, map_location="cpu", weights_only=True))
    net.eval()

    rng = np.random.default_rng()
    seeds = [int(rng.integers(0, 1_000_000_000)) for _ in range(N_SEEDS)]

    print(
        f"=== CNN-PPO multi-seed eval: {N_SEEDS} seeds × {LAYOUTS_PER_SEED} layouts ==="
    )
    print(f"seeds={seeds}")

    print(f"\n--- demos from seed={seeds[0]} ---")
    for i in range(N_DEMOS):
        total, reached, info, path = rollout_ppo(env, net, seed=seeds[0] + i)
        mark = "OK" if reached else "FAIL"
        print(
            f"demo {i + 1}: start={info['start']} "
            f"obstacles={sorted(info['obstacles'])}"
        )
        print(f"  return={total:.3f} steps={len(path) - 1} {mark}")
        print(env.render())
        print("-" * 40)

    seed_rates = []
    seed_means = []
    total_ok = 0
    total_n = 0
    print("\n--- per-seed summary ---")
    for seed in seeds:
        stats = eval_seed(env, net, seed, LAYOUTS_PER_SEED)
        seed_rates.append(stats["success_rate"])
        seed_means.append(stats["mean_return"])
        total_ok += int(stats["successes"])
        total_n += int(stats["n_layouts"])
        print(
            f"seed={seed}: success={stats['success_rate']:.1%} "
            f"({int(stats['successes'])}/{int(stats['n_layouts'])})  "
            f"mean_return={stats['mean_return']:.3f}  "
            f"min_return={stats['min_return']:.3f}"
        )

    rates = np.asarray(seed_rates, dtype=np.float64)
    print("\n=== PPO multi-seed summary ===")
    print(
        f"overall success={total_ok / total_n:.1%} ({total_ok}/{total_n})  "
        f"per-seed mean={rates.mean():.1%} ± {rates.std():.1%}  "
        f"worst={rates.min():.1%}  best={rates.max():.1%}"
    )
    print(f"mean_return across seeds={float(np.mean(seed_means)):.3f}")

    if os.path.exists(Q_TABLE_PATH):
        q = np.load(Q_TABLE_PATH)
        tab_rates = [eval_tabular_seed(env, q, seed, LAYOUTS_PER_SEED) for seed in seeds]
        tab = np.asarray(tab_rates, dtype=np.float64)
        print(
            f"Tabular Q multi-seed: mean={tab.mean():.1%} ± {tab.std():.1%}  "
            f"worst={tab.min():.1%}"
        )
        print("→ 布局一变，只看坐标的 Q 表没有墙的信息，泛化会明显变差。")


if __name__ == "__main__":
    main()
