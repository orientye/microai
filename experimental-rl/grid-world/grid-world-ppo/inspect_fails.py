"""Inspect remaining PPO failures on a fixed seed suite."""

from __future__ import annotations

from collections import Counter

import torch

from grid_world_env import GridWorldEnv
from ppo_train import ActorCritic, select_action

N_LAYOUTS = 200
SEED = 42


def main() -> None:
    env = GridWorldEnv(randomize_layout=True, n_obstacles=3, render_mode="ansi")
    net = ActorCritic(env.size, env.action_space.n)
    net.load_state_dict(torch.load("ppo_random_layout.pth", map_location="cpu", weights_only=True))
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
            action, _, _ = select_action(
                net,
                state,
                env.legal_actions(avoid_revisit=True),
                greedy=True,
            )
            state, _, terminated, truncated, step_info = env.step(action)
            path.append(env.pos)
            visits[env.pos] += 1
            bumps += int(step_info["bumped"])
            reached = terminated
            done = terminated or truncated
        if not reached:
            fails.append((info, path, visits, bumps))

    print(f"seed={SEED} layouts={N_LAYOUTS} fails={len(fails)}")
    for i, (info, path, visits, bumps) in enumerate(fails[:10], start=1):
        print(
            f"\nfail {i}: start={info['start']} "
            f"obstacles={sorted(info['obstacles'])} "
            f"steps={len(path) - 1} bumps={bumps}"
        )
        print(f"  path={path}")
        print(f"  top visits={visits.most_common(5)}")
        # Replay last frame for visual context.
        env.reset(
            seed=SEED,
            options={
                "randomize_layout": False,
                "obstacles": info["obstacles"],
                "start": info["start"],
            },
        )
        for pos in path[1:]:
            # Walk path only for render; ignore legality.
            env.pos = pos
        print(env.render())


if __name__ == "__main__":
    main()
