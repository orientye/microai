"""Three-seat PPO: public combo feat on both heads; perfect cards on critic only."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

HERE = Path(__file__).resolve().parent
ENV_DIR = HERE.parent / "doudizhu-env"
PPO_DIR = HERE.parent / "doudizhu-ppo"
CRITIC_DIR = HERE.parent / "doudizhu-ppo-critic"
DOUZERO_ROOT = HERE.parent / "DouZero"
for _p in (HERE, ENV_DIR, DOUZERO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_base = _load("doudizhu_legal_ppo", PPO_DIR / "ppo_train.py")
_critic = _load("doudizhu_perfect_ppo", CRITIC_DIR / "ppo_train.py")

CLIP_EPS = _base.CLIP_EPS
ENT_COEF = _base.ENT_COEF
LR = _base.LR
UPDATE_EPOCHS = _base.UPDATE_EPOCHS
VF_COEF = _base.VF_COEF
X_ACTION = _base.X_ACTION
X_STATE = _base.X_STATE
compute_gae = _base.compute_gae
legal_logits = _base.legal_logits
LstmScorer = _base.LstmScorer
normalize_adv = _base.normalize_adv
pad_legal_batch = _base.pad_legal_batch
encode_perfect_hands = _critic.encode_perfect_hands
PERFECT_DIM = _critic.PERFECT_DIM

from feat import FEAT_DIM, feat_from_infoset

FARMER_X_ACTION = 484
FARMER_X_STATE = 430
POSITIONS = ("landlord", "landlord_up", "landlord_down")
ADP_GRAD_CLIP = 40.0
BELIEF_DIM = 108
BELIEF_COEF = 0.5
OTHER_SEATS = {
    "landlord": ("landlord_up", "landlord_down"),
    "landlord_up": ("landlord", "landlord_down"),
    "landlord_down": ("landlord", "landlord_up"),
}


def encode_other_hands(all_handcards: dict, position: str) -> np.ndarray:
    """54-dim unary encoding of the two seats that are not `position`."""
    from douzero.env.env import _cards2array

    a, b = OTHER_SEATS[position]
    return np.concatenate(
        [_cards2array(all_handcards[a]), _cards2array(all_handcards[b])]
    ).astype(np.float32)


def module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def attach_feat(x: torch.Tensor | np.ndarray, feat: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.as_tensor(x, dtype=torch.float32)
    if isinstance(feat, np.ndarray):
        feat = torch.as_tensor(feat, dtype=torch.float32)
    feat = feat.to(device=x.device, dtype=x.dtype)
    if feat.dim() == 1:
        feat = feat.unsqueeze(0)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() == 2:
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        if feat.size(0) == 1 and x.size(0) != 1:
            feat = feat.expand(x.size(0), -1)
        return torch.cat([x, feat], dim=-1)
    if x.dim() == 3:
        b, k, _ = x.shape
        return torch.cat([x, feat.expand(b, k, -1)], dim=-1)
    raise ValueError(f"bad x dim {x.dim()}")


class BeliefHead(nn.Module):
    def __init__(self, x_state: int):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(x_state + FEAT_DIM + 128, 256),
            nn.ReLU(),
            nn.Linear(256, BELIEF_DIM),
        )

    def forward(self, z: torch.Tensor, x_pub: torch.Tensor) -> torch.Tensor:
        h = self.lstm(z)[0][:, -1]
        return self.mlp(torch.cat([h, x_pub], dim=-1))


class SeatAC(nn.Module):
    def __init__(self, x_action: int, x_state: int):
        super().__init__()
        self.x_state = x_state
        self.actor_head = LstmScorer(x_action + FEAT_DIM + BELIEF_DIM)
        self.critic_head = LstmScorer(x_state + FEAT_DIM + PERFECT_DIM)
        self.belief_head = BeliefHead(x_state)

    def value(
        self,
        obs: dict,
        feat: torch.Tensor | np.ndarray,
        perfect: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        dev = module_device(self)
        z = torch.as_tensor(obs["z"], dtype=torch.float32, device=dev).unsqueeze(0)
        x_no = torch.as_tensor(obs["x_no_action"], dtype=torch.float32, device=dev).unsqueeze(0)
        x_pub = attach_feat(x_no, feat).to(dev)
        if isinstance(perfect, np.ndarray):
            perfect = torch.as_tensor(perfect, dtype=torch.float32, device=dev)
        else:
            perfect = perfect.to(dev)
        if perfect.dim() == 1:
            perfect = perfect.unsqueeze(0)
        return self.critic_head(z, torch.cat([x_pub, perfect], dim=-1))

    def belief_logits(
        self,
        obs: dict,
        feat: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        dev = module_device(self)
        z = torch.as_tensor(obs["z"], dtype=torch.float32, device=dev)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        x_no = torch.as_tensor(obs["x_no_action"], dtype=torch.float32, device=dev)
        x_pub = attach_feat(x_no, feat).to(dev)
        return self.belief_head(z, x_pub)

    @torch.no_grad()
    def act(
        self,
        obs: dict,
        feat: np.ndarray | torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        dev = module_device(self)
        z = torch.as_tensor(obs["z"], dtype=torch.float32, device=dev).unsqueeze(0)
        belief = torch.sigmoid(self.belief_logits(obs, feat)).squeeze(0)
        x = attach_feat(attach_feat(obs["x_batch"], feat), belief).to(dev).unsqueeze(0)
        mask = torch.ones(1, x.size(1), dtype=torch.bool, device=dev)
        logits = legal_logits(self, z, x, mask)[0]
        dist = Categorical(logits=logits)
        idx = logits.argmax() if deterministic else dist.sample()
        return int(idx.item()), float(dist.log_prob(idx).item()), 0.0


class TripleModels:
    def __init__(self):
        self.models = {
            "landlord": SeatAC(X_ACTION, X_STATE),
            "landlord_up": SeatAC(FARMER_X_ACTION, FARMER_X_STATE),
            "landlord_down": SeatAC(FARMER_X_ACTION, FARMER_X_STATE),
        }

    def __getitem__(self, position: str) -> SeatAC:
        return self.models[position]

    def to(self, device: str | torch.device) -> TripleModels:
        for model in self.models.values():
            model.to(device)
        return self

    def optimizers(self) -> dict:
        return {p: optim.Adam(m.parameters(), lr=LR) for p, m in self.models.items()}

    def state_dict(self) -> dict:
        return {p: m.state_dict() for p, m in self.models.items()}

    def load_state_dict(self, state: dict) -> None:
        for p, sd in state.items():
            self.models[p].load_state_dict(sd)


def assign_episode_returns(bufs: dict, landlord_g: float) -> None:
    for pos, steps in bufs.items():
        g = float(landlord_g) if pos == "landlord" else -float(landlord_g)
        for step in steps:
            step["reward"] = 0.0
            step["done"] = False
        if steps:
            steps[-1]["reward"] = g
            steps[-1]["done"] = True


def collect_games(env, models: TripleModels, min_games: int) -> dict:
    pooled = {p: [] for p in POSITIONS}
    for _ in range(min_games):
        obs = env.reset()
        ep = {p: [] for p in POSITIONS}
        done = False
        reward = 0.0
        while not done:
            pos = env.position
            infoset = env._env.infoset
            feat = feat_from_infoset(infoset)
            perfect = encode_perfect_hands(env.all_handcards)
            belief_target = encode_other_hands(env.all_handcards, pos)
            with torch.no_grad():
                belief = torch.sigmoid(models[pos].belief_logits(obs, feat)).squeeze(0).cpu()
            idx, log_prob, _v = models[pos].act(obs, feat)
            value = float(models[pos].value(obs, feat, perfect).item())
            action = env.legal_actions[idx]
            next_obs, reward, done, _info = env.step(action)
            ep[pos].append(
                {
                    "z": torch.as_tensor(obs["z"], dtype=torch.float32),
                    "x_batch": torch.as_tensor(obs["x_batch"], dtype=torch.float32),
                    "x_no_action": torch.as_tensor(
                        obs["x_no_action"], dtype=torch.float32
                    ),
                    "feat": torch.as_tensor(feat, dtype=torch.float32),
                    "perfect": torch.as_tensor(perfect, dtype=torch.float32),
                    "belief": belief.detach().clone(),
                    "belief_target": torch.as_tensor(belief_target, dtype=torch.float32),
                    "action_idx": idx,
                    "log_prob": log_prob,
                    "value": value,
                    "reward": 0.0,
                    "done": False,
                }
            )
            obs = next_obs
        assign_episode_returns(ep, landlord_g=float(reward))
        for p in POSITIONS:
            pooled[p].extend(ep[p])
    return pooled


def ppo_update_seat(
    model: SeatAC,
    optimizer: optim.Optimizer,
    batch: list[dict],
    max_grad_norm: float = ADP_GRAD_CLIP,
) -> float:
    if len(batch) < 2:
        return 0.0
    dev = module_device(model)
    x_act = [
        attach_feat(attach_feat(s["x_batch"], s["feat"]), s["belief"]).to(dev)
        for s in batch
    ]
    z_b, x_b, mask = pad_legal_batch(
        [s["z"].to(dev) for s in batch],
        x_act,
    )
    mask = mask.to(dev)
    x_crit = torch.cat(
        [
            attach_feat(
                torch.stack([s["x_no_action"] for s in batch]),
                torch.stack([s["feat"] for s in batch]),
            ),
            torch.stack([s["perfect"] for s in batch]),
        ],
        dim=-1,
    ).to(dev)
    idx = torch.tensor([s["action_idx"] for s in batch], dtype=torch.long, device=dev)
    old_lp = torch.tensor([s["log_prob"] for s in batch], dtype=torch.float32, device=dev)
    rewards = torch.tensor([s["reward"] for s in batch], dtype=torch.float32, device=dev)
    dones = torch.tensor([float(s["done"]) for s in batch], dtype=torch.float32, device=dev)
    values = torch.tensor([s["value"] for s in batch], dtype=torch.float32, device=dev)

    last_value = torch.zeros((), device=dev)
    if not batch[-1]["done"]:
        with torch.no_grad():
            last_value = model.critic_head(
                batch[-1]["z"].to(dev).unsqueeze(0),
                torch.cat(
                    [
                        attach_feat(
                            batch[-1]["x_no_action"].unsqueeze(0),
                            batch[-1]["feat"].unsqueeze(0),
                        ),
                        batch[-1]["perfect"].unsqueeze(0),
                    ],
                    dim=-1,
                ).to(dev),
            ).squeeze(0)

    advantages, returns = compute_gae(rewards, dones, values, last_value)
    advantages = normalize_adv(advantages)
    loss_value = 0.0
    for _ in range(UPDATE_EPOCHS):
        logits = legal_logits(model, z_b, x_b, mask)
        dist = Categorical(logits=logits)
        new_lp = dist.log_prob(idx)
        entropy = dist.entropy().mean()
        v_pred = model.critic_head(z_b, x_crit)
        x_pub = attach_feat(
            torch.stack([s["x_no_action"] for s in batch]),
            torch.stack([s["feat"] for s in batch]),
        ).to(dev)
        belief_pred = model.belief_head(z_b, x_pub)
        belief_tgt = torch.stack([s["belief_target"] for s in batch]).to(dev)
        belief_loss = F.binary_cross_entropy_with_logits(belief_pred, belief_tgt)
        ratio = torch.exp(new_lp - old_lp)
        surrogate = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
        policy_loss = -torch.min(surrogate, clipped).mean()
        value_loss = F.mse_loss(v_pred, returns)
        loss = (
            policy_loss
            + VF_COEF * value_loss
            + BELIEF_COEF * belief_loss
            - ENT_COEF * entropy
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        loss_value = float(loss.item())
    return loss_value
