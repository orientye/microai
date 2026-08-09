"""Test CNN-DQN on fresh random layouts; contrast with fixed-map tabular Q."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

from dqn_train import QNet
from grid_world_env import GridWorldEnv

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

DQN_PATH = "dqn_random_layout.pth"
Q_TABLE_PATH = os.path.join("..", "grid-world-qlearning", "q_table.npy")
N_LAYOUTS = 200
SEED = 42


def main() -> None:
    if not os.path.exists(DQN_PATH):
        raise SystemExit(f"missing {DQN_PATH}; run dqn_train.py first")

    env = GridWorldEnv(randomize_layout=True, n_obstacles=3, render_mode="ansi")
    net = QNet(env.size, env.action_space.n)
    net.load_state_dict(torch.load(DQN_PATH, map_location="cpu", weights_only=True))
    net.eval()

    successes = 0
    returns = []
    print(f"=== CNN-DQN greedy on {N_LAYOUTS} fresh random layouts (seed={SEED}) ===")
    for i in range(N_LAYOUTS):
        state, info = env.reset(
            seed=SEED + i, options={"randomize_layout": True, "random_start": True}
        )
        path = [info["start"]]
        total = 0.0
        done = False
        reached = False
        while not done:
            with torch.no_grad():
                x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                action = int(net(x).argmax(dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            path.append(env.pos)
            reached = terminated
            done = terminated or truncated
        successes += int(reached)
        returns.append(total)
        if i < 5:
            print(f"demo {i + 1}: start={info['start']} obstacles={sorted(info['obstacles'])}")
            print(f"  return={total:.3f} steps={len(path) - 1} {'OK' if reached else 'FAIL'}")
            print(env.render())
            print("-" * 40)

    print(
        f"DQN summary: success={successes / N_LAYOUTS:.0%}  "
        f"mean_return={float(np.mean(returns)):.3f}  "
        f"min_return={float(np.min(returns)):.3f}"
    )

    if os.path.exists(Q_TABLE_PATH):
        q = np.load(Q_TABLE_PATH)
        tab_ok = 0
        for i in range(N_LAYOUTS):
            _, info = env.reset(
                seed=SEED + i, options={"randomize_layout": True, "random_start": True}
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
            tab_ok += int(reached)
        print(
            f"Tabular Q (../grid-world-qlearning, position-only) on random layouts: "
            f"success={tab_ok / N_LAYOUTS:.0%}"
        )
        print("→ 布局一变，只看坐标的 Q 表没有墙的信息，泛化会明显变差。")


if __name__ == "__main__":
    main()
