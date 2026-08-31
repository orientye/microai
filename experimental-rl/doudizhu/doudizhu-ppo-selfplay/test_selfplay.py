"""Tests for three-seat self-play PPO (WP, farmers also learn)."""

from collections import defaultdict

import numpy as np
import torch

from ppo_train import (
    FARMER_X_ACTION,
    FARMER_X_STATE,
    POSITIONS,
    SeatAC,
    TripleModels,
    assign_episode_returns,
)


def test_assign_episode_returns_farmers_get_negated_g():
    bufs = {
        "landlord": [{"reward": 0.0, "done": False}, {"reward": 1.0, "done": True}],
        "landlord_up": [{"reward": 0.0, "done": False}],
        "landlord_down": [{"reward": 0.0, "done": False}, {"reward": 0.0, "done": False}],
    }
    assign_episode_returns(bufs, landlord_g=1.0)
    assert bufs["landlord"][-1]["reward"] == 1.0
    assert bufs["landlord"][0]["reward"] == 0.0
    assert bufs["landlord_up"][-1]["reward"] == -1.0
    assert bufs["landlord_down"][-1]["reward"] == -1.0
    assert bufs["landlord_down"][0]["reward"] == 0.0
    assert bufs["landlord_down"][-1]["done"] is True


def test_farmer_network_dims():
    m = SeatAC(x_action=FARMER_X_ACTION, x_state=FARMER_X_STATE)
    z = torch.zeros(1, 5, 162)
    x_a = torch.zeros(1, FARMER_X_ACTION)
    x_c = torch.zeros(1, FARMER_X_STATE + 162)
    assert m.actor_head(z, x_a).shape == (1,)
    assert m.critic_head(z, x_c).shape == (1,)


def test_collect_all_seats_and_update():
    from pathlib import Path
    import sys

    env_dir = Path(__file__).resolve().parent.parent / "doudizhu-env"
    if str(env_dir) not in sys.path:
        sys.path.insert(0, str(env_dir))
    from doudizhu_env import DoudizhuEnv
    from ppo_train import collect_games, ppo_update_seat

    env = DoudizhuEnv(objective="wp")
    models = TripleModels()
    opts = models.optimizers()
    batch = collect_games(env, models, min_games=2)
    for pos in POSITIONS:
        assert pos in batch
        assert len(batch[pos]) >= 1
        assert batch[pos][0]["x_batch"].shape[-1] in (
            373,
            FARMER_X_ACTION,
        )
        assert batch[pos][0]["perfect"].shape == (162,)
    for pos in POSITIONS:
        if len(batch[pos]) >= 1:
            loss = ppo_update_seat(models[pos], opts[pos], batch[pos])
            assert np.isfinite(loss)


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
    print("all passed")
