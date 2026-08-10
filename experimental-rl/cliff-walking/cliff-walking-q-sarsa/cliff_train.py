"""Train tabular Q-learning and SARSA side-by-side on CliffWalking-v1."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cliff_env import (
    choose_action,
    epsilon_by_episode,
    evaluate_greedy,
    make_env,
    print_policy,
    save_policy_figure,
)
from td_updates import q_learning_update, sarsa_update

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ----------------- Hyperparameters (shared) -----------------
ALPHA = 0.5
GAMMA = 1.0
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY_EPISODES = 400
MAX_EPISODES = 500
EVAL_EVERY = 50
MAX_EPISODE_STEPS = 100
SEED_Q = 0
SEED_SARSA = 1

SAVE_Q_QL = "q_qlearning.npy"
SAVE_Q_SARSA = "q_sarsa.npy"
SAVE_CURVE = "reward_history.png"
SAVE_POLICY_QL = "policy_qlearning.png"
SAVE_POLICY_SARSA = "policy_sarsa.png"

UpdateFn = Callable[..., None]


def train_one(
    name: str,
    update_fn: UpdateFn,
    *,
    seed: int,
) -> tuple[np.ndarray, list[float], list[tuple[int, float]]]:
    rng = np.random.default_rng(seed)
    env = make_env(max_episode_steps=MAX_EPISODE_STEPS)
    n_states = int(env.observation_space.n)
    n_actions = int(env.action_space.n)
    q = np.zeros((n_states, n_actions), dtype=np.float64)

    episode_returns: list[float] = []
    eval_returns: list[tuple[int, float]] = []

    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset(seed=seed + episode)
        epsilon = epsilon_by_episode(
            episode - 1,
            eps_start=EPS_START,
            eps_end=EPS_END,
            decay_episodes=EPS_DECAY_EPISODES,
        )
        action = choose_action(q, state, epsilon, rng)
        done = False
        ep_return = 0.0

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            next_action = 0 if done else choose_action(q, int(next_state), epsilon, rng)

            update_fn(
                q,
                int(state),
                int(action),
                float(reward),
                int(next_state),
                int(next_action),
                done,
                alpha=ALPHA,
                gamma=GAMMA,
            )

            state = int(next_state)
            action = int(next_action)
            ep_return += float(reward)

        episode_returns.append(ep_return)

        if episode % EVAL_EVERY == 0 or episode == MAX_EPISODES:
            stats = evaluate_greedy(env, q, n_episodes=20)
            eval_returns.append((episode, stats["mean_return"]))
            print(
                f"[{name}] episode={episode:4d}  eps={epsilon:.3f}  "
                f"train_return={ep_return:7.1f}  "
                f"greedy_mean={stats['mean_return']:7.1f}  "
                f"success={stats['success_rate']:.0%}"
            )

    env.close()
    return q, episode_returns, eval_returns


def save_curves(
    ql_returns: list[float],
    sarsa_returns: list[float],
    ql_eval: list[tuple[int, float]],
    sarsa_eval: list[tuple[int, float]],
    path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    window = 20

    def smooth(xs: list[float]) -> tuple[range, np.ndarray]:
        if len(xs) < window:
            return range(1, len(xs) + 1), np.asarray(xs)
        kernel = np.ones(window) / window
        return range(window, len(xs) + 1), np.convolve(xs, kernel, mode="valid")

    x_q, y_q = smooth(ql_returns)
    x_s, y_s = smooth(sarsa_returns)
    ax.plot(x_q, y_q, label=f"Q-learning train (MA{window})", alpha=0.85)
    ax.plot(x_s, y_s, label=f"SARSA train (MA{window})", alpha=0.85)

    if ql_eval:
        xs, ys = zip(*ql_eval)
        ax.plot(xs, ys, "o-", label="Q-learning greedy mean", markersize=4)
    if sarsa_eval:
        xs, ys = zip(*sarsa_eval)
        ax.plot(xs, ys, "s-", label="SARSA greedy mean", markersize=4)

    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title("CliffWalking-v1: Q-learning vs SARSA")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    out = Path(__file__).resolve().parent
    print("Training Q-learning (off-policy, max bootstrap)...")
    q_ql, ret_ql, eval_ql = train_one("Q-learning", q_learning_update, seed=SEED_Q)
    print()
    print("Training SARSA (on-policy, next-action bootstrap)...")
    q_sarsa, ret_sarsa, eval_sarsa = train_one("SARSA", sarsa_update, seed=SEED_SARSA)

    np.save(out / SAVE_Q_QL, q_ql)
    np.save(out / SAVE_Q_SARSA, q_sarsa)
    print(f"saved Q-tables -> {SAVE_Q_QL}, {SAVE_Q_SARSA}")

    print()
    print_policy(q_ql, "Q-learning greedy policy")
    print()
    print_policy(q_sarsa, "SARSA greedy policy")

    save_policy_figure(q_ql, str(out / SAVE_POLICY_QL), "Q-learning greedy policy")
    save_policy_figure(q_sarsa, str(out / SAVE_POLICY_SARSA), "SARSA greedy policy")
    save_curves(ret_ql, ret_sarsa, eval_ql, eval_sarsa, str(out / SAVE_CURVE))
    print(f"saved figures -> {SAVE_POLICY_QL}, {SAVE_POLICY_SARSA}, {SAVE_CURVE}")
    print()
    print(
        "Expect: Q-learning hugs the cliff (optimal under greedy); "
        "SARSA stays higher / safer while ε-greedy exploration is on."
    )


if __name__ == "__main__":
    main()
