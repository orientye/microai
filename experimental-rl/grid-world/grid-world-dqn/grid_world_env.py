"""Random-layout 5x5 GridWorld for DQN.

Observation channels (flat): agent | obstacle | goal | visited.
The visited channel breaks greedy A↔B / bump loops at test time.
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
N_CHANNELS = 4


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        size: int = 5,
        step_penalty: float = -0.01,
        bump_penalty: float = -0.05,
        revisit_penalty: float = -0.03,
        goal_reward: float = 1.0,
        max_steps: int = 50,
        n_obstacles: int = 3,
        randomize_layout: bool = True,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.size = size
        self.step_penalty = step_penalty
        self.bump_penalty = bump_penalty
        self.revisit_penalty = revisit_penalty
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.n_obstacles = n_obstacles
        self.randomize_layout = randomize_layout
        self.render_mode = render_mode

        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles: set[tuple[int, int]] = set()
        self.visit_count = np.zeros((size, size), dtype=np.float32)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(N_CHANNELS * size * size,),
            dtype=np.float32,
        )

        self.pos: tuple[int, int] = self.start
        self.steps = 0

    def manhattan(self, pos: tuple[int, int] | None = None) -> int:
        r, c = self.pos if pos is None else pos
        return abs(r - self.goal[0]) + abs(c - self.goal[1])

    def legal_actions(self, *, avoid_revisit: bool = False) -> list[int]:
        """Actions that move into a free cell (not border/obstacle bumps).

        If ``avoid_revisit`` and some legal moves enter a never/less-visited cell,
        only those are returned — breaks greedy A↔B loops during eval/test.
        """
        candidates: list[tuple[int, float]] = []
        for action, (dr, dc) in enumerate(ACTION_DELTAS):
            nr, nc = self.pos[0] + dr, self.pos[1] + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.obstacles:
                candidates.append((action, float(self.visit_count[nr, nc])))
        if not candidates:
            return list(range(self.action_space.n))
        if avoid_revisit:
            fresh = [a for a, v in candidates if v < 1.0]
            if fresh:
                return fresh
            min_v = min(v for _, v in candidates)
            return [a for a, v in candidates if v == min_v]
        return [a for a, _ in candidates]

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

    def cells_reaching_goal(self) -> list[tuple[int, int]]:
        q: deque[tuple[int, int]] = deque([self.goal])
        seen = {self.goal}
        while q:
            r, c = q.popleft()
            for dr, dc in ACTION_DELTAS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    continue
                nxt = (nr, nc)
                if nxt in self.obstacles or nxt in seen:
                    continue
                seen.add(nxt)
                q.append(nxt)
        return [p for p in seen if p != self.goal]

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

    def _sample_obstacles(self, n_obstacles: int) -> set[tuple[int, int]]:
        candidates = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in (self.start, self.goal)
        ]
        n_obstacles = min(n_obstacles, len(candidates))
        for _ in range(300):
            idx = self.np_random.choice(len(candidates), size=n_obstacles, replace=False)
            obstacles = {candidates[int(i)] for i in np.asarray(idx).reshape(-1)}
            if self._path_exists(obstacles):
                return obstacles
        return set()

    def _observe(self) -> np.ndarray:
        agent = np.zeros((self.size, self.size), dtype=np.float32)
        obstacle = np.zeros((self.size, self.size), dtype=np.float32)
        goal = np.zeros((self.size, self.size), dtype=np.float32)
        # Cap so the channel stays in [0, 1]; saturation still signals "looping hard".
        visited = np.clip(self.visit_count / 5.0, 0.0, 1.0)
        agent[self.pos] = 1.0
        for r, c in self.obstacles:
            obstacle[r, c] = 1.0
        goal[self.goal] = 1.0
        return np.concatenate([agent.ravel(), obstacle.ravel(), goal.ravel(), visited.ravel()])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        n_obstacles = int(options.get("n_obstacles", self.n_obstacles))

        if options.get("randomize_layout", self.randomize_layout):
            self.obstacles = self._sample_obstacles(n_obstacles)
        elif "obstacles" in options:
            self.obstacles = {tuple(x) for x in options["obstacles"]}
            if not self._path_exists(self.obstacles):
                raise ValueError("provided obstacles block start→goal")

        reachable = self.cells_reaching_goal()
        if not reachable:
            self.obstacles = self._sample_obstacles(n_obstacles)
            reachable = self.cells_reaching_goal()

        if "start" in options:
            start = tuple(options["start"])
            if start not in reachable:
                raise ValueError(f"invalid/unreachable start cell: {start}")
            self.pos = start  # type: ignore[assignment]
        elif options.get("random_start", False):
            idx = int(self.np_random.integers(0, len(reachable)))
            self.pos = reachable[idx]
        else:
            self.pos = self.start if self.start in reachable else reachable[0]

        self.steps = 0
        self.visit_count = np.zeros((self.size, self.size), dtype=np.float32)
        self.visit_count[self.pos] = 1.0
        return self._observe(), {
            "start": self.pos,
            "obstacles": frozenset(self.obstacles),
            "manhattan": self.manhattan(),
        }

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")

        prev = self.pos
        dr, dc = ACTION_DELTAS[action]
        intended = (prev[0] + dr, prev[1] + dc)
        bumped = False
        if not (0 <= intended[0] < self.size and 0 <= intended[1] < self.size):
            bumped = True
        elif intended in self.obstacles:
            bumped = True
        else:
            self.pos = intended

        self.steps += 1
        prev_visits = float(self.visit_count[self.pos])
        self.visit_count[self.pos] += 1.0

        terminated = self.pos == self.goal
        truncated = self.steps >= self.max_steps
        if terminated:
            reward = self.goal_reward
        else:
            reward = self.bump_penalty if bumped else self.step_penalty
            # Penalize revisits to discourage A↔B / stay-put loops.
            if prev_visits >= 1.0:
                reward += self.revisit_penalty * prev_visits

        return self._observe(), reward, terminated, truncated, {
            "bumped": bumped,
            "manhattan": self.manhattan(),
            "revisits": prev_visits,
        }

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
