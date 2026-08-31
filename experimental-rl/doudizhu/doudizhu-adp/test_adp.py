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
