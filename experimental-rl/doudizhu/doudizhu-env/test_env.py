"""Tests for the thin Dou Dizhu rule wrapper (no PPO)."""

from collections import Counter
from pathlib import Path
import sys

EVAL_RULER = Path(__file__).resolve().parent.parent / "eval-ruler"
if str(EVAL_RULER) not in sys.path:
    sys.path.insert(0, str(EVAL_RULER))

from generate_eval_data import FULL_DECK, generate_deals
from doudizhu_env import DoudizhuEnv, make_deal


def _has_move(legal, cards: list[int]) -> bool:
    target = sorted(cards)
    return any(sorted(m) == target for m in legal)


def test_reset_fixed_deal_landlord_acts():
    env = DoudizhuEnv(objective="wp")
    deal = generate_deals(1, seed=0)[0]
    obs = env.reset(deal=deal)
    assert env.position == "landlord"
    assert len(env.legal_actions) >= 1
    assert obs["x_batch"].shape[0] == len(env.legal_actions)


def test_empty_table_cannot_pass():
    env = DoudizhuEnv()
    env.reset(deal=generate_deals(1, seed=1)[0])
    assert [] not in env.legal_actions
    assert not _has_move(env.legal_actions, [])


def test_bomb_straight_plane_king_in_legal():
    landlord = (
        [3, 3, 3, 3]
        + [4, 4, 4]
        + [5, 6, 7]
        + [8, 9, 10, 11, 12, 13, 14]
        + [17]
        + [20, 30]
    )
    assert len(landlord) == 20
    env = DoudizhuEnv()
    env.reset(deal=make_deal(landlord))
    legal = env.legal_actions
    assert _has_move(legal, [3, 3, 3, 3])
    assert _has_move(legal, [3, 4, 5, 6, 7])
    assert _has_move(legal, [3, 3, 3, 4, 4, 4, 5, 6])
    assert _has_move(legal, [20, 30])


def test_reset_does_not_mutate_deal():
    deal = generate_deals(1, seed=2)[0]
    before = list(deal["landlord"])
    env = DoudizhuEnv()
    env.reset(deal=deal)
    env.step(env.legal_actions[0])
    assert deal["landlord"] == before


def test_all_handcards_is_perfect_info():
    deal = generate_deals(1, seed=3)[0]
    env = DoudizhuEnv()
    env.reset(deal=deal)
    hands = env.all_handcards
    assert set(hands) == {"landlord", "landlord_up", "landlord_down"}
    pooled = hands["landlord"] + hands["landlord_up"] + hands["landlord_down"]
    assert Counter(pooled) == Counter(FULL_DECK)


def test_wp_episode_reward_is_pm1():
    env = DoudizhuEnv(objective="wp")
    env.reset(deal=generate_deals(1, seed=4)[0])
    done = False
    reward = 0.0
    steps = 0
    obs = None
    while not done:
        obs, reward, done, _info = env.step(env.legal_actions[0])
        steps += 1
        assert steps < 200
    assert done
    assert reward in (1.0, -1.0)
    assert obs is None


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
    print("all passed")
