"""Inspect remaining DQN failures on the same seed suite as dqn_test."""

from __future__ import annotations

from collections import Counter

import torch

from dqn_train import QNet
from grid_world_env import GridWorldEnv

N_LAYOUTS = 200
SEED = 42


def main() -> None:
    env = GridWorldEnv(randomize_layout=True, n_obstacles=3, render_mode="ansi")
    net = QNet(env.size, env.action_space.n)
    net.load_state_dict(torch.load("dqn_random_layout.pth", map_location="cpu", weights_only=True))
    net.eval()

    fails = []
    for i in range(N_LAYOUTS):
        state, info = env.reset(
            seed=SEED + i,
            options={"randomize_layout": True, "random_start": True, "n_obstacles": 3},
        )
        path = [info["start"]]
        visits = Counter([info["start"]])
        bumps = 0
        done = False
        reached = False
        while not done:
            legal = env.legal_actions(avoid_revisit=True)
            with torch.no_grad():
                x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                q = net(x).squeeze(0).clone()
                mask = torch.full_like(q, -1e9)
                idx = torch.as_tensor(legal, dtype=torch.int64)
                mask[idx] = q[idx]
                action = int(mask.argmax().item())
            state, _, terminated, truncated, step_info = env.step(action)
            path.append(env.pos)
            visits[env.pos] += 1
            bumps += int(step_info.get("bumped", False))
            reached = terminated
            done = terminated or truncated
        if not reached:
            fails.append(
                {
                    "i": i,
                    "seed": SEED + i,
                    "start": info["start"],
                    "obstacles": sorted(info["obstacles"]),
                    "steps": len(path) - 1,
                    "bumps": bumps,
                    "max_visit": max(visits.values()),
                    "path": path,
                }
            )

    print(f"fails={len(fails)}/{N_LAYOUTS}")
    for f in fails:
        print("---")
        print(
            f"index={f['i']} seed={f['seed']} start={f['start']} "
            f"obstacles={f['obstacles']}"
        )
        print(f"steps={f['steps']} bumps={f['bumps']} max_visit={f['max_visit']}")
        path = f["path"]
        head = " -> ".join(str(p) for p in path[:16])
        if len(path) > 16:
            head += " ..."
        print("path:", head)
        print("tail:", " -> ".join(str(p) for p in path[-12:]))
        env.reset(
            seed=f["seed"],
            options={"randomize_layout": True, "random_start": True, "n_obstacles": 3},
        )
        print(env.render())


if __name__ == "__main__":
    main()
