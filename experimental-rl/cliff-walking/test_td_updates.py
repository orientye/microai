"""Unit tests for tabular Q-learning vs SARSA TD updates."""

from __future__ import annotations

import unittest

import numpy as np

from td_updates import q_learning_update, sarsa_update


class TestTdUpdates(unittest.TestCase):
    def test_q_learning_uses_max_next_q_not_behavior_action(self) -> None:
        q = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 5.0, 2.0, 0.0],  # next state: max is action 1 -> 5
            ],
            dtype=np.float64,
        )
        # Behavior next action would be 0 (Q=1), but Q-learning must use max=5.
        q_learning_update(
            q,
            state=0,
            action=1,
            reward=-1.0,
            next_state=1,
            next_action=0,
            done=False,
            alpha=0.5,
            gamma=1.0,
        )
        # td_target = -1 + 1.0 * 5 = 4; td_error = 4 - 0 = 4; new Q = 0 + 0.5*4 = 2
        self.assertEqual(q[0, 1], 2.0)

    def test_sarsa_uses_actual_next_action(self) -> None:
        q = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 5.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )
        sarsa_update(
            q,
            state=0,
            action=1,
            reward=-1.0,
            next_state=1,
            next_action=0,  # actual next action Q=1
            done=False,
            alpha=0.5,
            gamma=1.0,
        )
        # td_target = -1 + 1.0 * 1 = 0; new Q = 0 + 0.5*0 = 0
        self.assertEqual(q[0, 1], 0.0)

    def test_both_cut_bootstrap_when_done(self) -> None:
        q_q = np.zeros((2, 4), dtype=np.float64)
        q_s = np.zeros((2, 4), dtype=np.float64)
        q_q[1] = [10.0, 10.0, 10.0, 10.0]
        q_s[1] = [10.0, 10.0, 10.0, 10.0]

        q_learning_update(
            q_q, 0, 0, reward=-1.0, next_state=1, next_action=0, done=True, alpha=1.0, gamma=1.0
        )
        sarsa_update(
            q_s, 0, 0, reward=-1.0, next_state=1, next_action=0, done=True, alpha=1.0, gamma=1.0
        )

        self.assertEqual(q_q[0, 0], -1.0)
        self.assertEqual(q_s[0, 0], -1.0)


if __name__ == "__main__":
    unittest.main()
