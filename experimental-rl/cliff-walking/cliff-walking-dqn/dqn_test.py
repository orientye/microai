"""Load Double DQN weights and play CliffWalking with human pixel render."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from dqn_train import (
    N_ACTIONS,
    N_STATES,
    QNet,
    encode,
    evaluate_greedy,
    make_env,
    net_to_q_table,
    print_policy,
)

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "dqn_cliff.pth"
N_RENDER_EPISODES = 3
STEP_SLEEP = 0.25
GAP_SLEEP = 1.0


def load_net() -> QNet:
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Missing {WEIGHTS.name}; run dqn_train.py first.")
    net = QNet(N_STATES, N_ACTIONS)
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    net.eval()
    return net


def render_greedy(net: QNet, *, n_episodes: int = N_RENDER_EPISODES) -> None:
    env = make_env(render_mode="human")
    print(f"=== Double DQN: human render ({n_episodes} eps) ===")
    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        total = 0.0
        steps = 0
        print(f"\n--- episode {ep}/{n_episodes} ---")
        while True:
            with torch.no_grad():
                action = int(net(torch.FloatTensor(encode(state)).unsqueeze(0)).argmax().item())
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


def main() -> None:
    net = load_net()
    q = net_to_q_table(net)
    print_policy(q, "Double DQN greedy policy")
    print()

    env = make_env()
    stats = evaluate_greedy(env, net, n_episodes=50)
    env.close()
    print(
        f"greedy eval (50 eps): success={stats['success_rate']:.0%}  "
        f"mean_return={stats['mean_return']:.1f}  min={stats['min_return']:.1f}"
    )
    print()
    render_greedy(net)


if __name__ == "__main__":
    main()
