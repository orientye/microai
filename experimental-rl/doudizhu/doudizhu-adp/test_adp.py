"""Tests for ADP training returns and PPO agents on the seat-swap ruler."""

from pathlib import Path
import sys

SELFPLAY = Path(__file__).resolve().parent.parent / "doudizhu-ppo-selfplay"
if str(SELFPLAY) not in sys.path:
    sys.path.insert(0, str(SELFPLAY))

from ppo_train import TripleModels, assign_episode_returns
from ppo_agent import PpoSeatAgent, load_ppo_players


def test_adp_assign_uses_signed_power_of_two():
    bufs = {
        "landlord": [{"reward": 0.0, "done": False}],
        "landlord_up": [{"reward": 0.0, "done": False}],
        "landlord_down": [{"reward": 0.0, "done": False}],
    }
    assign_episode_returns(bufs, landlord_g=4.0)
    assert bufs["landlord"][-1]["reward"] == 4.0
    assert bufs["landlord_up"][-1]["reward"] == -4.0
    assert bufs["landlord_down"][-1]["reward"] == -4.0


def test_ppo_agent_returns_legal_action():
    env_dir = Path(__file__).resolve().parent.parent / "doudizhu-env"
    if str(env_dir) not in sys.path:
        sys.path.insert(0, str(env_dir))
    from doudizhu_env import DoudizhuEnv

    env = DoudizhuEnv(objective="adp")
    env.reset(seed=0)
    models = TripleModels()
    agent = PpoSeatAgent(models["landlord"])
    action = agent.act(env._env.infoset)
    assert action in env.legal_actions


def test_landlord_metrics_use_paper_adp():
    from ruler_metrics import metrics_from_landlord_results

    out = metrics_from_landlord_results([(True, 0), (True, 0), (False, 1)])
    assert out["games"] == 3
    assert abs(out["wp"] - 2 / 3) < 1e-9
    assert abs(out["adp"] - (1 + 1 - 2) / 3) < 1e-9


def test_is_better_needs_wp_over_half_and_positive_adp():
    from ruler_metrics import is_better

    assert is_better(0.51, 0.1) is True
    assert is_better(0.51, 0.0) is False
    assert is_better(0.5, 1.0) is False


def test_is_strong_vs_random_needs_wp_at_least_point_nine():
    from ruler_metrics import is_strong_vs_random

    assert is_strong_vs_random(0.90, 0.1) is True
    assert is_strong_vs_random(0.89, 1.0) is False
    assert is_strong_vs_random(0.95, 0.0) is False


def test_eval_landlord_vs_random_deals_plays_fixed_pickle():
    import importlib.util
    import pickle
    from ruler_metrics import eval_landlord_vs_random_deals

    critic_path = Path(__file__).resolve().parent.parent / "doudizhu-ppo-critic" / "ppo_train.py"
    spec = importlib.util.spec_from_file_location("doudizhu_perfect_ppo", critic_path)
    critic = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(critic)

    eval_path = Path(__file__).resolve().parent.parent / "eval-ruler" / "eval_data.pkl"
    with eval_path.open("rb") as f:
        deals = pickle.load(f)[:1]
    out = eval_landlord_vs_random_deals(critic.PerfectLegalAC(), deals)
    assert out["games"] == 1
    assert out["wp"] in (0.0, 1.0)


def test_load_landlord_curriculum_into_triple_keeps_farmers():
    import importlib.util
    import tempfile

    import torch
    from selfplay_init import load_landlord_curriculum

    critic_path = Path(__file__).resolve().parent.parent / "doudizhu-ppo-critic" / "ppo_train.py"
    spec = importlib.util.spec_from_file_location("doudizhu_perfect_ppo", critic_path)
    critic = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(critic)

    src = critic.PerfectLegalAC()
    models = TripleModels()
    farmer_before = {
        k: v.detach().clone() for k, v in models["landlord_up"].state_dict().items()
    }
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ll.pth"
        torch.save(src.state_dict(), ckpt)
        load_landlord_curriculum(models, ckpt)
    for k, v in src.state_dict().items():
        assert torch.equal(models["landlord"].state_dict()[k], v)
    for k, v in farmer_before.items():
        assert torch.equal(models["landlord_up"].state_dict()[k], v)


def test_eval_trio_vs_random_deals_seat_swap_two_games():
    import pickle
    from selfplay_init import eval_trio_vs_random_deals

    eval_path = Path(__file__).resolve().parent.parent / "eval-ruler" / "eval_data.pkl"
    with eval_path.open("rb") as f:
        deals = pickle.load(f)[:1]
    out = eval_trio_vs_random_deals(TripleModels(), deals)
    assert out["games"] == 2
    assert out["num_deals"] == 1
    assert 0.0 <= out["wp_a"] <= 1.0


def test_assign_mc_returns_every_step_gets_same_g():
    from vs_douzero import assign_mc_returns

    bufs = {
        "landlord": [{"x": 0}, {"x": 1}],
        "landlord_up": [{"x": 0}],
        "landlord_down": [{"x": 0}, {"x": 1}, {"x": 2}],
    }
    assign_mc_returns(bufs, landlord_g=4.0)
    assert [s["mc_return"] for s in bufs["landlord"]] == [4.0, 4.0]
    assert bufs["landlord_up"][0]["mc_return"] == -4.0
    assert [s["mc_return"] for s in bufs["landlord_down"]] == [-4.0, -4.0, -4.0]


def test_collect_vs_douzero_landlord_role_stores_only_landlord():
    env_dir = Path(__file__).resolve().parent.parent / "doudizhu-env"
    if str(env_dir) not in sys.path:
        sys.path.insert(0, str(env_dir))
    from doudizhu_env import DoudizhuEnv
    from vs_douzero import collect_vs_douzero, load_douzero_players

    dz_dir = Path(__file__).resolve().parent.parent / "DouZero" / "baselines" / "douzero_ADP"
    env = DoudizhuEnv(objective="adp")
    models = TripleModels()
    dz = load_douzero_players(dz_dir)
    batch = collect_vs_douzero(env, models, dz, min_games=1, ours="landlord")
    assert len(batch["landlord"]) >= 1
    assert "mc_return" in batch["landlord"][0]
    assert batch["landlord_up"] == []
    assert batch["landlord_down"] == []


def test_expert_action_index_matches_legal_row():
    from clone import expert_action_index

    legal = [[3, 3], [4], []]
    assert expert_action_index(legal, [4]) == 1
    assert expert_action_index(legal, []) == 2


def test_collect_expert_stores_douzero_index():
    env_dir = Path(__file__).resolve().parent.parent / "doudizhu-env"
    if str(env_dir) not in sys.path:
        sys.path.insert(0, str(env_dir))
    from doudizhu_env import DoudizhuEnv
    from clone import collect_expert_games
    from vs_douzero import load_douzero_players

    dz_dir = Path(__file__).resolve().parent.parent / "DouZero" / "baselines" / "douzero_ADP"
    env = DoudizhuEnv(objective="adp")
    dz = load_douzero_players(dz_dir)
    env.reset(seed=0)
    batch = collect_expert_games(env, dz, min_games=1)
    for pos in ("landlord", "landlord_up", "landlord_down"):
        assert len(batch[pos]) >= 1
        step = batch[pos][0]
        k = step["x_batch"].shape[0]
        assert 0 <= step["action_idx"] < k


def test_adp_terminal_reward_is_power_of_two():
    import math

    env_dir = Path(__file__).resolve().parent.parent / "doudizhu-env"
    if str(env_dir) not in sys.path:
        sys.path.insert(0, str(env_dir))
    from doudizhu_env import DoudizhuEnv

    env = DoudizhuEnv(objective="adp")
    env.reset(seed=1)
    done = False
    reward = 0.0
    while not done:
        _obs, reward, done, _ = env.step(env.legal_actions[0])
    mag = abs(float(reward))
    k = round(math.log2(mag))
    assert k >= 0
    assert abs(mag - 2**k) < 1e-6


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
    print("all passed")
