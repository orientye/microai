"""Tests for public combo features + feat PPO (no leak)."""

from __future__ import annotations

import inspect

import numpy as np

from feat import (
    FEAT_DIM,
    LEAD_NAMES,
    encode_public_feat,
    feat_from_infoset,
    hand_stats,
    lead_onehot,
)


def test_bomb_own_hand():
    feat = encode_public_feat([3, 3, 3, 3], [], [])
    assert feat.shape == (FEAT_DIM,)
    assert feat.dtype == np.float32
    assert abs(float(feat[4]) - 1.0 / 13.0) < 1e-6


def test_rocket_own_hand():
    feat = encode_public_feat([20, 30], [], [])
    assert abs(float(feat[6]) - 1.0) < 1e-6
    assert abs(float(feat[5]) - 1.0) < 1e-6


def test_union_is_merged_counts_not_sum_of_stats():
    left, right = [3, 3], [3, 3]
    merged = left + right
    feat = encode_public_feat([], merged, [])
    union_stats = feat[7:14]
    split_sum = hand_stats(left) + hand_stats(right)
    assert abs(float(union_stats[2]) - 0.0) < 1e-6
    assert abs(float(union_stats[4]) - 1.0 / 13.0) < 1e-6
    assert abs(float(split_sum[2]) - 2.0 / 13.0) < 1e-6


def test_lead_empty_vs_single():
    empty = lead_onehot([])
    single = lead_onehot([3])
    assert empty[LEAD_NAMES.index("empty")] == 1.0
    assert single[LEAD_NAMES.index("single")] == 1.0
    assert empty.sum() == 1.0 and single.sum() == 1.0
    assert not np.array_equal(empty, single)


def test_lead_rocket_and_straight():
    rocket = lead_onehot([20, 30])
    straight = lead_onehot([3, 4, 5, 6, 7])
    assert rocket[LEAD_NAMES.index("rocket")] == 1.0
    assert straight[LEAD_NAMES.index("straight")] == 1.0


def test_act_signature_has_no_perfect():
    from ppo_feat import SeatAC

    params = inspect.signature(SeatAC.act).parameters
    assert "perfect" not in params
    assert "feat" in params


def test_actor_forward_ignores_perfect_dim():
    from ppo_feat import FEAT_DIM, SeatAC, X_ACTION, X_STATE

    m = SeatAC(X_ACTION, X_STATE)
    z = np.zeros((5, 162), dtype=np.float32)
    x_batch = np.zeros((3, X_ACTION), dtype=np.float32)
    feat = np.zeros(FEAT_DIM, dtype=np.float32)
    obs = {
        "z": z,
        "x_batch": x_batch,
        "x_no_action": np.zeros(X_STATE, dtype=np.float32),
    }
    idx, logp, value = m.act(obs, feat, deterministic=True)
    assert idx in (0, 1, 2)
    assert value == 0.0
    assert m.actor_head.mlp[0].in_features == X_ACTION + FEAT_DIM + 128
    assert m.critic_head.mlp[0].in_features == X_STATE + FEAT_DIM + 162 + 128


def test_collect_stores_feat_and_update():
    import sys
    from pathlib import Path

    env_dir = Path(__file__).resolve().parent.parent / "doudizhu-env"
    if str(env_dir) not in sys.path:
        sys.path.insert(0, str(env_dir))
    from doudizhu_env import DoudizhuEnv
    from ppo_feat import FEAT_DIM, POSITIONS, TripleModels, collect_games, ppo_update_seat

    env = DoudizhuEnv(objective="adp")
    models = TripleModels()
    opts = models.optimizers()
    batch = collect_games(env, models, min_games=2)
    for pos in POSITIONS:
        assert len(batch[pos]) >= 1
        assert batch[pos][0]["feat"].shape == (FEAT_DIM,)
        assert batch[pos][0]["perfect"].shape == (162,)
        assert "perfect" not in inspect.signature(models[pos].act).parameters
    for pos in POSITIONS:
        loss = ppo_update_seat(models[pos], opts[pos], batch[pos])
        assert np.isfinite(loss)


def test_feat_agent_uses_public_fields_only():
    from feat_agent import FeatSeatAgent
    from ppo_feat import SeatAC, X_ACTION, X_STATE
    import douzero.env.env as douzero_env

    class _Info:
        legal_actions = [[3], [4]]
        player_hand_cards = [3, 4]
        other_hand_cards = [5]
        last_move = []
        player_position = "landlord"

        @property
        def all_handcards(self):
            raise AssertionError("eval must not read all_handcards")

    dummy_obs = {
        "z": np.zeros((5, 162), dtype=np.float32),
        "x_batch": np.zeros((2, X_ACTION), dtype=np.float32),
        "x_no_action": np.zeros(X_STATE, dtype=np.float32),
    }
    orig_get_obs = douzero_env.get_obs
    douzero_env.get_obs = lambda infoset: dummy_obs
    try:
        agent = FeatSeatAgent(SeatAC(X_ACTION, X_STATE))
        move = agent.act(_Info())
        assert move in ([3], [4])
    finally:
        douzero_env.get_obs = orig_get_obs


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
    print("all passed")
