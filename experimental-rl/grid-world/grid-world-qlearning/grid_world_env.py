"""Fixed-layout 5x5 GridWorld for tabular Q-learning.

Layout (row 0 at top):

    S . . # .
    . . . # .
    . . . . .
    . # . . .
    . . . . G

Observation: Discrete cell id 0..24.
Actions: 0=Up, 1=Right, 2=Down, 3=Left.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces


ACTION_DELTAS = (
    (-1, 0),  # Up
    (0, 1),   # Right
    (1, 0),   # Down
    (0, -1),  # Left
)
ACTION_ARROWS = ("↑", "→", "↓", "←")
DEFAULT_OBSTACLES = frozenset({(0, 3), (1, 3), (3, 1)})


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        size: int = 5,
        step_penalty: float = -0.01,
        goal_reward: float = 1.0,
        max_steps: int = 100,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.size = size
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles = set(DEFAULT_OBSTACLES)

        self.observation_space = spaces.Discrete(size * size)
        self.action_space = spaces.Discrete(4)

        self.pos: tuple[int, int] = self.start
        self.steps = 0

    def state_to_index(self, row: int, col: int) -> int:
        return row * self.size + col

    def index_to_state(self, index: int) -> tuple[int, int]:
        return divmod(index, self.size)

    def free_cells(self, *, include_goal: bool = False) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) in self.obstacles:
                    continue
                if not include_goal and (r, c) == self.goal:
                    continue
                cells.append((r, c))
        return cells

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        self.obstacles = set(DEFAULT_OBSTACLES)

        if "start" in options:
            start = tuple(options["start"])
            if start in self.obstacles or start == self.goal:
                raise ValueError(f"invalid start cell: {start}")
            if not (0 <= start[0] < self.size and 0 <= start[1] < self.size):
                raise ValueError(f"start out of bounds: {start}")
            self.pos = start  # type: ignore[assignment]
        elif options.get("random_start", False):
            cells = self.free_cells(include_goal=False)
            idx = int(self.np_random.integers(0, len(cells)))
            self.pos = cells[idx]
        else:
            self.pos = self.start

        self.steps = 0
        return self.state_to_index(*self.pos), {"start": self.pos}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")

        dr, dc = ACTION_DELTAS[action]
        nr = min(max(self.pos[0] + dr, 0), self.size - 1)
        nc = min(max(self.pos[1] + dc, 0), self.size - 1)
        if (nr, nc) not in self.obstacles:
            self.pos = (nr, nc)

        self.steps += 1
        terminated = self.pos == self.goal
        truncated = self.steps >= self.max_steps
        reward = self.goal_reward if terminated else self.step_penalty
        return self.state_to_index(*self.pos), reward, terminated, truncated, {}

    def render(self) -> str:
        lines = []
        for r in range(self.size):
            cells = []
            for c in range(self.size):
                if (r, c) == self.pos:
                    cells.append("A")
                elif (r, c) == self.goal:
                    cells.append("G")
                elif (r, c) == self.start:
                    cells.append("S")
                elif (r, c) in self.obstacles:
                    cells.append("#")
                else:
                    cells.append(".")
            lines.append(" ".join(cells))
        text = "\n".join(lines)
        if self.render_mode == "ansi":
            return text
        print(text)
        return text
