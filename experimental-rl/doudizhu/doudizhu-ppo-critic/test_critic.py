"""Tests: critic sees three hands; actor does not."""

import numpy as np
import torch

from ppo_train import (
    PERFECT_DIM,
    PerfectLegalAC,
    encode_perfect_hands,
)


def test_encode_perfect_hands_shape():
    hands = {
        "landlord": [3, 3, 3, 3],
        "landlord_up": [4, 4],
        "landlord_down": [20, 30],
    }
    vec = encode_perfect_hands(hands)
    assert vec.shape == (PERFECT_DIM,)
    other = encode_perfect_hands(
        {
            "landlord": [5],
            "landlord_up": [4, 4],
            "landlord_down": [20, 30],
        }
    )
    assert not np.allclose(vec, other)


def test_actor_logits_ignore_perfect_info():
    model = PerfectLegalAC()
    model.eval()
    k = 3
    obs = {
        "z": np.zeros((5, 162), dtype=np.float32),
        "x_batch": np.zeros((k, 373), dtype=np.float32),
        "x_no_action": np.zeros(319, dtype=np.float32),
    }
    p0 = torch.zeros(PERFECT_DIM)
    p1 = torch.ones(PERFECT_DIM)
    z = torch.as_tensor(obs["z"]).unsqueeze(0)
    x = torch.as_tensor(obs["x_batch"]).unsqueeze(0)
    mask = torch.ones(1, k, dtype=torch.bool)
    from ppo_train import legal_logits

    logits = legal_logits(model, z, x, mask)
    v0 = model.value(obs, p0)
    v1 = model.value(obs, p1)
    assert logits.shape == (1, k)
    assert not torch.allclose(v0, v1)


def test_collect_stores_perfect_and_update():
    from pathlib import Path
    import sys

    env_dir = Path(__file__).resolve().parent.parent / "doudizhu-env"
    if str(env_dir) not in sys.path:
        sys.path.insert(0, str(env_dir))
    from doudizhu_env import DoudizhuEnv
    from ppo_train import collect_episodes, ppo_update

    env = DoudizhuEnv(objective="wp")
    model = PerfectLegalAC()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch = collect_episodes(env, model, min_landlord_steps=8)
    assert batch[0]["perfect"].shape == (PERFECT_DIM,)
    loss = ppo_update(model, opt, batch)
    assert np.isfinite(loss)


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
    print("all passed")
