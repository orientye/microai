"""Random-layout 5x5 GridWorld for DQN.

Each reset can sample a new obstacle set (BFS-guaranteed solvable).
Observation is a flat 3-channel grid: [agent | obstacle | goal].
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


ACTION_DELTAS = (
    (-1, 0),  # Up
    (0, 1),   # Right
    (1, 0),   # Down
    (0, -1),  # Left
)
ACTION_ARROWS = ("↑", "→", "↓", "←")


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        size: int = 5,
        step_penalty: float = -0.01,
        goal_reward: float = 1.0,
        max_steps: int = 100,
        n_obstacles: int = 3,
        randomize_layout: bool = True,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.size = size
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.n_obstacles = n_obstacles
        self.randomize_layout = randomize_layout
        self.render_mode = render_mode

        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles: set[tuple[int, int]] = set()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3 * size * size,),
            dtype=np.float32,
        )

        self.pos: tuple[int, int] = self.start
        self.steps = 0

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

    def _path_exists(self, obstacles: set[tuple[int, int]]) -> bool:
        if self.start in obstacles or self.goal in obstacles:
            return False
        q: deque[tuple[int, int]] = deque([self.start])
        seen = {self.start}
        while q:
            r, c = q.popleft()
            if (r, c) == self.goal:
                return True
            for dr, dc in ACTION_DELTAS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    continue
                nxt = (nr, nc)
                if nxt in obstacles or nxt in seen:
                    continue
                seen.add(nxt)
                q.append(nxt)
        return False

    def _sample_obstacles(self) -> set[tuple[int, int]]:
        candidates = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in (self.start, self.goal)
        ]
        for _ in range(200):
            idx = self.np_random.choice(len(candidates), size=self.n_obstacles, replace=False)
            obstacles = {candidates[int(i)] for i in np.asarray(idx).reshape(-1)}
            if self._path_exists(obstacles):
                return obstacles
        return set()

    def _observe(self) -> np.ndarray:
        agent = np.zeros((self.size, self.size), dtype=np.float32)
        obstacle = np.zeros((self.size, self.size), dtype=np.float32)
        goal = np.zeros((self.size, self.size), dtype=np.float32)
        agent[self.pos] = 1.0
        for r, c in self.obstacles:
            obstacle[r, c] = 1.0
        goal[self.goal] = 1.0
        return np.concatenate([agent.ravel(), obstacle.ravel(), goal.ravel()])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}

        if options.get("randomize_layout", self.randomize_layout):
            self.obstacles = self._sample_obstacles()
        elif "obstacles" in options:
            self.obstacles = {tuple(x) for x in options["obstacles"]}
            if not self._path_exists(self.obstacles):
                raise ValueError("provided obstacles block start→goal")

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
        return self._observe(), {
            "start": self.pos,
            "obstacles": frozenset(self.obstacles),
        }

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
        return self._observe(), reward, terminated, truncated, {}

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
