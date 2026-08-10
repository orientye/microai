"""Tabular TD updates: Q-learning (off-policy) vs SARSA (on-policy)."""

from __future__ import annotations

import numpy as np


def q_learning_update(
    q: np.ndarray,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    next_action: int,
    done: bool,
    *,
    alpha: float,
    gamma: float,
) -> None:
    """Off-policy: bootstrap with max_a' Q(s', a'). ``next_action`` is unused."""
    del next_action  # Q-learning ignores the behavior next action.
    td_target = reward
    if not done:
        td_target += gamma * float(np.max(q[next_state]))
    q[state, action] += alpha * (td_target - q[state, action])


def sarsa_update(
    q: np.ndarray,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    next_action: int,
    done: bool,
    *,
    alpha: float,
    gamma: float,
) -> None:
    """On-policy: bootstrap with Q(s', a') where a' is the actual next action."""
    td_target = reward
    if not done:
        td_target += gamma * float(q[next_state, next_action])
    q[state, action] += alpha * (td_target - q[state, action])
