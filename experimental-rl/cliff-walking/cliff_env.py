"""CliffWalking-v1 factory, geometry, and policy visualization helpers."""

from __future__ import annotations

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Gymnasium CliffWalking actions: 0=Up, 1=Right, 2=Down, 3=Left
ACTION_ARROWS = ("↑", "→", "↓", "←")
NROWS, NCOLS = 4, 12
START = (3, 0)
GOAL = (3, 11)
CLIFF_COLS = range(1, 11)  # bottom row cells that are the cliff


def make_env(
    *,
    max_episode_steps: int = 100,
    render_mode: str | None = None,
) -> gym.Env:
    return gym.make(
        "CliffWalking-v1",
        is_slippery=False,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )


def state_to_rc(state: int) -> tuple[int, int]:
    return divmod(int(state), NCOLS)


def rc_to_state(row: int, col: int) -> int:
    return row * NCOLS + col


def is_cliff(row: int, col: int) -> bool:
    return row == NROWS - 1 and col in CLIFF_COLS


def epsilon_by_episode(
    episode: int,
    *,
    eps_start: float,
    eps_end: float,
    decay_episodes: int,
) -> float:
    if episode >= decay_episodes:
        return eps_end
    frac = episode / decay_episodes
    return eps_start + frac * (eps_end - eps_start)


def choose_action(
    q: np.ndarray,
    state: int,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, q.shape[1]))
    return int(np.argmax(q[state]))


def run_greedy_episode(env: gym.Env, q: np.ndarray) -> tuple[float, bool, int]:
    state, _ = env.reset()
    total = 0.0
    steps = 0
    done = False
    reached_goal = False
    while not done:
        action = int(np.argmax(q[state]))
        state, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        steps += 1
        reached_goal = bool(terminated)
        done = terminated or truncated
    return total, reached_goal, steps


def evaluate_greedy(
    env: gym.Env,
    q: np.ndarray,
    *,
    n_episodes: int = 20,
) -> dict[str, float]:
    returns: list[float] = []
    successes = 0
    for _ in range(n_episodes):
        ret, ok, _ = run_greedy_episode(env, q)
        returns.append(ret)
        successes += int(ok)
    return {
        "n_episodes": float(n_episodes),
        "success_rate": successes / n_episodes,
        "mean_return": float(np.mean(returns)),
        "min_return": float(np.min(returns)),
    }


def policy_grid(q: np.ndarray) -> list[list[str]]:
    grid: list[list[str]] = []
    for r in range(NROWS):
        row: list[str] = []
        for c in range(NCOLS):
            if is_cliff(r, c):
                row.append("C")
            elif (r, c) == GOAL:
                row.append("G")
            else:
                s = rc_to_state(r, c)
                row.append(ACTION_ARROWS[int(np.argmax(q[s]))])
        grid.append(row)
    return grid


def print_policy(q: np.ndarray, title: str) -> None:
    print(f"{title} (↑→↓←; C=cliff, G=goal):")
    for row in policy_grid(q):
        print(" ".join(row))


def save_policy_figure(q: np.ndarray, path: str, title: str) -> None:
    values = np.full((NROWS, NCOLS), np.nan, dtype=np.float64)
    arrows = np.full((NROWS, NCOLS), "", dtype=object)
    for r in range(NROWS):
        for c in range(NCOLS):
            if is_cliff(r, c):
                continue
            s = rc_to_state(r, c)
            values[r, c] = float(np.max(q[s]))
            if (r, c) != GOAL:
                arrows[r, c] = ACTION_ARROWS[int(np.argmax(q[s]))]

    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(values, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="V(s)=max_a Q(s,a)")
    for r in range(NROWS):
        for c in range(NCOLS):
            if is_cliff(r, c):
                ax.text(c, r, "C", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
            elif (r, c) == GOAL:
                ax.text(c, r, "G", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
            else:
                ax.text(c, r, arrows[r, c], ha="center", va="center", color="white", fontsize=12)
    ax.set_xticks(range(NCOLS))
    ax.set_yticks(range(NROWS))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
