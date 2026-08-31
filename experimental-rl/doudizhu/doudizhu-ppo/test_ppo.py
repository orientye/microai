"""Tests for legal-set PPO helpers (landlord vs random farmers)."""

import numpy as np
import torch
from torch.distributions import Categorical

from ppo_train import (
    LegalActorCritic,
    apply_opponent_terminal,
    compute_gae,
    legal_logits,
    pad_legal_batch,
)


def test_legal_softmax_log_prob():
    logits = torch.tensor([0.0, 2.0, 0.0])
    dist = Categorical(logits=logits)
    assert dist.log_prob(torch.tensor(1)).item() > dist.log_prob(torch.tensor(0)).item()
    assert abs(dist.probs.sum().item() - 1.0) < 1e-5


def test_pad_legal_batch_masks_dummy_rows():
    z = torch.zeros(5, 162)
    x2 = torch.zeros(2, 373)
    x3 = torch.zeros(3, 373)
    z_b, x_b, mask = pad_legal_batch(
        [z, z],
        [x2, x3],
    )
    assert z_b.shape == (2, 5, 162)
    assert x_b.shape == (2, 3, 373)
    assert mask.tolist() == [[True, True, False], [True, True, True]]


def test_legal_logits_masked_neg_inf():
    model = LegalActorCritic()
    model.eval()
    z = torch.zeros(2, 5, 162)
    x = torch.zeros(2, 4, 373)
    mask = torch.tensor(
        [[True, True, False, False], [True, True, True, False]],
        dtype=torch.bool,
    )
    logits = legal_logits(model, z, x, mask)
    assert torch.all(logits[:, 0:2] > -1e8)
    assert torch.all(logits[0, 2:] < -1e8)
    assert torch.isfinite(logits[1, 2])
    assert logits[1, 3] < -1e8
    dist0 = Categorical(logits=logits[0])
    assert dist0.probs[2] < 1e-6
    assert dist0.probs[3] < 1e-6


def test_apply_opponent_terminal_credits_last_landlord_step():
    steps = [
        {"reward": 0.0, "done": False},
        {"reward": 0.0, "done": False},
    ]
    apply_opponent_terminal(steps, reward=-1.0)
    assert steps[-1]["reward"] == -1.0
    assert steps[-1]["done"] is True
    assert steps[0]["reward"] == 0.0


def test_gae_terminal_only_reward():
    rewards = torch.tensor([0.0, 0.0, 1.0])
    dones = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([0.1, 0.2, 0.3])
    adv, ret = compute_gae(rewards, dones, values, last_value=torch.tensor(0.0))
    assert abs(ret[-1].item() - 1.0) < 1e-5
    assert ret[0].item() > 0.5


def test_act_and_one_update_smoke():
    from pathlib import Path
    import sys

    env_dir = Path(__file__).resolve().parent.parent / "doudizhu-env"
    if str(env_dir) not in sys.path:
        sys.path.insert(0, str(env_dir))
    from doudizhu_env import DoudizhuEnv
    from ppo_train import collect_episodes, ppo_update

    env = DoudizhuEnv(objective="wp")
    model = LegalActorCritic()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch = collect_episodes(env, model, min_landlord_steps=8, farmer="random")
    assert len(batch) >= 1
    for step in batch:
        assert step["x_batch"].shape[0] == step["x_batch"].shape[0]
        assert 0 <= step["action_idx"] < step["x_batch"].shape[0]
    loss = ppo_update(model, opt, batch)
    assert np.isfinite(loss)


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
    print("all passed")
