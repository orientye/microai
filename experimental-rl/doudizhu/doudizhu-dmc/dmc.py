"""Deep Monte-Carlo on DouZero features: Q(s,a) → whole-episode G."""

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

HERE = Path(__file__).resolve().parent
ENV_DIR = HERE.parent / "doudizhu-env"
DOUZERO = HERE.parent / "DouZero"
for _p in (ENV_DIR, DOUZERO, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

POSITIONS = ("landlord", "landlord_up", "landlord_down")
X_LANDLORD = 373
X_FARMER = 484
Z_STEPS = 5
Z_DIM = 162
LSTM_HID = 128
HID = 256
LR = 1e-4
EPSILON = 0.01
ADP_GRAD_CLIP = 40.0


def module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class QNet(nn.Module):
    """LSTM over move history, MLP on (h, x) → Q(s,a). Same width as our PPO scorer."""

    def __init__(self, x_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(Z_DIM, LSTM_HID, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(x_dim + LSTM_HID, HID),
            nn.ReLU(),
            nn.Linear(HID, HID),
            nn.ReLU(),
            nn.Linear(HID, 1),
        )

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = self.lstm(z)[0][:, -1]
        return self.mlp(torch.cat([h, x], dim=-1)).squeeze(-1)


class TripleQ:
    def __init__(self):
        self.models = {
            "landlord": QNet(X_LANDLORD),
            "landlord_up": QNet(X_FARMER),
            "landlord_down": QNet(X_FARMER),
        }

    def __getitem__(self, position: str) -> QNet:
        return self.models[position]

    def to(self, device: str | torch.device) -> TripleQ:
        for model in self.models.values():
            model.to(device)
        return self

    def state_dict(self) -> dict:
        return {p: m.state_dict() for p, m in self.models.items()}

    def load_state_dict(self, state: dict) -> None:
        for p, m in self.models.items():
            m.load_state_dict(state[p])

    def optimizers(self) -> dict:
        return {p: optim.RMSprop(m.parameters(), lr=LR) for p, m in self.models.items()}


def assign_dmc_targets(bufs: dict, landlord_g: float) -> None:
    """Every step of a seat carries the same terminal G (farmers get -G)."""
    for pos, steps in bufs.items():
        g = float(landlord_g) if pos == "landlord" else -float(landlord_g)
        for step in steps:
            step["target"] = g


def epsilon_greedy_index(q: torch.Tensor, epsilon: float) -> int:
    k = int(q.numel())
    if k < 1:
        raise ValueError("empty Q vector")
    if epsilon > 0.0 and random.random() < epsilon:
        return random.randrange(k)
    return int(torch.argmax(q).item())


def score_legal(model: QNet, obs: dict) -> torch.Tensor:
    dev = module_device(model)
    z = torch.as_tensor(obs["z_batch"], dtype=torch.float32, device=dev)
    x = torch.as_tensor(obs["x_batch"], dtype=torch.float32, device=dev)
    with torch.no_grad():
        return model(z, x)


def dmc_update(
    model: QNet,
    optimizer: optim.Optimizer,
    batch: list[dict],
    max_grad_norm: float = ADP_GRAD_CLIP,
) -> float:
    if len(batch) < 1:
        return 0.0
    dev = module_device(model)
    z = torch.stack([s["z"] for s in batch]).to(dev)
    x = torch.stack([s["x"] for s in batch]).to(dev)
    target = torch.tensor([s["target"] for s in batch], dtype=torch.float32, device=dev)
    pred = model(z, x)
    loss = F.mse_loss(pred, target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    return float(loss.item())


class Replay:
    def __init__(self, capacity: int = 50_000):
        self.buf: deque = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buf)

    def add_many(self, steps: list[dict]) -> None:
        self.buf.extend(steps)

    def sample(self, n: int) -> list[dict]:
        if len(self.buf) <= n:
            return list(self.buf)
        return random.sample(list(self.buf), n)


def collect_dmc_games(
    env,
    models: TripleQ,
    min_games: int,
    epsilon: float = EPSILON,
) -> dict:
    pooled = {p: [] for p in POSITIONS}
    for _ in range(min_games):
        obs = env.reset()
        ep = {p: [] for p in POSITIONS}
        done = False
        reward = 0.0
        while not done:
            pos = env.position
            legal = env.legal_actions
            q = score_legal(models[pos], obs)
            idx = epsilon_greedy_index(q, epsilon)
            action = legal[idx]
            ep[pos].append(
                {
                    "z": torch.as_tensor(obs["z"], dtype=torch.float32),
                    "x": torch.as_tensor(obs["x_batch"][idx], dtype=torch.float32),
                }
            )
            obs, reward, done, _info = env.step(action)
        assign_dmc_targets(ep, landlord_g=float(reward))
        for p in POSITIONS:
            pooled[p].extend(ep[p])
    return pooled


def collect_landlord_vs_random(
    env,
    model: QNet,
    min_games: int,
    epsilon: float = EPSILON,
) -> list[dict]:
    pooled: list[dict] = []
    for _ in range(min_games):
        obs = env.reset()
        ep: list[dict] = []
        done = False
        reward = 0.0
        while not done:
            pos = env.position
            legal = env.legal_actions
            if pos == "landlord":
                q = score_legal(model, obs)
                idx = epsilon_greedy_index(q, epsilon)
                action = legal[idx]
                ep.append(
                    {
                        "z": torch.as_tensor(obs["z"], dtype=torch.float32),
                        "x": torch.as_tensor(obs["x_batch"][idx], dtype=torch.float32),
                    }
                )
            else:
                action = random.choice(legal)
            obs, reward, done, _info = env.step(action)
        for step in ep:
            step["target"] = float(reward)
        pooled.extend(ep)
    return pooled
