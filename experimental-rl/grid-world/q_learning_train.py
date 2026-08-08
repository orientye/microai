"""Tabular Q-learning on the custom 5x5 GridWorld."""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grid_world_env import ACTION_ARROWS, GridWorldEnv

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ----------------- Hyperparameters -----------------
ALPHA = 0.1
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 3000
MAX_EPISODES = 4000
EVAL_EVERY = 200
SAVE_Q = "q_table.npy"
SAVE_CURVE = "reward_history.png"
SAVE_POLICY = "policy_map.png"


def epsilon_by_episode(episode: int) -> float:
    # Linear decay: keep some exploration late in training.
    if episode >= EPS_DECAY_EPISODES:
        return EPS_END
    frac = episode / EPS_DECAY_EPISODES
    return EPS_START + frac * (EPS_END - EPS_START)


def choose_action(q: np.ndarray, state: int, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, q.shape[1]))
    return int(np.argmax(q[state]))


def run_greedy_episode(env: GridWorldEnv, q: np.ndarray) -> float:
    state, _ = env.reset()
    total = 0.0
    done = False
    while not done:
        action = int(np.argmax(q[state]))
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        done = terminated or truncated
    return total


def policy_grid(env: GridWorldEnv, q: np.ndarray) -> list[list[str]]:
    grid: list[list[str]] = []
    for r in range(env.size):
        row: list[str] = []
        for c in range(env.size):
            if (r, c) in env.obstacles:
                row.append("#")
            elif (r, c) == env.goal:
                row.append("G")
            else:
                s = env.state_to_index(r, c)
                row.append(ACTION_ARROWS[int(np.argmax(q[s]))])
        grid.append(row)
    return grid


def print_policy(env: GridWorldEnv, q: np.ndarray) -> None:
    grid = policy_grid(env, q)
    print("Learned greedy policy (↑→↓←):")
    for row in grid:
        print(" ".join(row))


def save_policy_figure(env: GridWorldEnv, q: np.ndarray, path: str) -> None:
    values = np.full((env.size, env.size), np.nan, dtype=np.float64)
    arrows = np.full((env.size, env.size), "", dtype=object)
    for r in range(env.size):
        for c in range(env.size):
            if (r, c) in env.obstacles:
                continue
            s = env.state_to_index(r, c)
            values[r, c] = float(np.max(q[s]))
            if (r, c) != env.goal:
                arrows[r, c] = ACTION_ARROWS[int(np.argmax(q[s]))]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(values, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V(s)=max_a Q(s,a)")
    for r in range(env.size):
        for c in range(env.size):
            if (r, c) in env.obstacles:
                ax.text(c, r, "#", ha="center", va="center", color="white", fontsize=16)
            elif (r, c) == env.goal:
                ax.text(c, r, "G", ha="center", va="center", color="white", fontsize=16, fontweight="bold")
            else:
                ax.text(c, r, arrows[r, c], ha="center", va="center", color="white", fontsize=16)
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title("GridWorld greedy policy")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(0)
    env = GridWorldEnv()
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q = np.zeros((n_states, n_actions), dtype=np.float64)

    episode_returns: list[float] = []
    eval_returns: list[tuple[int, float]] = []

    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        done = False
        ep_return = 0.0
        epsilon = epsilon_by_episode(episode - 1)

        while not done:
            action = choose_action(q, state, epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Classic tabular Q-learning:
            # Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') * (1-done) - Q(s,a))
            td_target = reward
            if not done:
                td_target += GAMMA * float(np.max(q[next_state]))
            td_error = td_target - q[state, action]
            q[state, action] += ALPHA * td_error

            state = next_state
            ep_return += reward

        episode_returns.append(ep_return)

        if episode % EVAL_EVERY == 0 or episode == MAX_EPISODES:
            greedy_scores = [run_greedy_episode(env, q) for _ in range(20)]
            mean_score = float(np.mean(greedy_scores))
            eval_returns.append((episode, mean_score))
            print(
                f"episode={episode:4d}  eps={epsilon:.3f}  "
                f"train_return={ep_return:.3f}  greedy_mean={mean_score:.3f}"
            )

    np.save(SAVE_Q, q)
    print(f"saved Q-table -> {SAVE_Q}")
    print_policy(env, q)
    save_policy_figure(env, q, SAVE_POLICY)
    print(f"saved policy map -> {SAVE_POLICY}")

    fig, ax = plt.subplots(figsize=(8, 4))
    window = 50
    if len(episode_returns) >= window:
        kernel = np.ones(window) / window
        smooth = np.convolve(episode_returns, kernel, mode="valid")
        ax.plot(range(window, len(episode_returns) + 1), smooth, label=f"train return (MA{window})")
    else:
        ax.plot(episode_returns, label="train return")
    if eval_returns:
        xs, ys = zip(*eval_returns)
        ax.plot(xs, ys, "o-", label="greedy eval mean")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title("GridWorld Q-learning")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SAVE_CURVE, dpi=140)
    plt.close(fig)
    print(f"saved reward curve -> {SAVE_CURVE}")


if __name__ == "__main__":
    main()
