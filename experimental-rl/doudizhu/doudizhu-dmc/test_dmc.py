"""Tests for Deep Monte-Carlo: same-G targets, MSE(Q,G), legal ε-greedy."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def test_assign_dmc_targets_every_step_same_g():
    from dmc import assign_dmc_targets

    bufs = {
        "landlord": [{"x": 0}, {"x": 1}],
        "landlord_up": [{"x": 0}],
        "landlord_down": [{"x": 0}, {"x": 1}, {"x": 2}],
    }
    assign_dmc_targets(bufs, landlord_g=4.0)
    assert [s["target"] for s in bufs["landlord"]] == [4.0, 4.0]
    assert bufs["landlord_up"][0]["target"] == -4.0
    assert [s["target"] for s in bufs["landlord_down"]] == [-4.0, -4.0, -4.0]


def test_epsilon_greedy_zero_is_argmax():
    import torch
    from dmc import epsilon_greedy_index

    q = torch.tensor([0.1, 3.0, 1.2])
    assert epsilon_greedy_index(q, epsilon=0.0) == 1


def test_dmc_mse_fits_constant_target():
    import torch
    from dmc import QNet, dmc_update

    model = QNet(x_dim=8)
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-2)
    z = [torch.zeros(5, 162) for _ in range(4)]
    x = [torch.zeros(8) for _ in range(4)]
    batch = [{"z": z[i], "x": x[i], "target": 2.0} for i in range(4)]
    before = dmc_update(model, opt, batch, max_grad_norm=40.0)
    after = before
    for _ in range(20):
        after = dmc_update(model, opt, batch, max_grad_norm=40.0)
    assert after < before
    with torch.no_grad():
        pred = model(z[0].unsqueeze(0), x[0].unsqueeze(0)).item()
    assert abs(pred - 2.0) < 0.5


def test_dmc_agent_returns_legal_action():
    env_dir = HERE.parent / "doudizhu-env"
    if str(env_dir) not in sys.path:
        sys.path.insert(0, str(env_dir))
    from doudizhu_env import DoudizhuEnv
    from dmc import QNet, TripleQ
    from dmc_agent import DmcSeatAgent

    env = DoudizhuEnv(objective="adp")
    env.reset(seed=0)
    models = TripleQ()
    agent = DmcSeatAgent(models["landlord"])
    action = agent.act(env._env.infoset)
    assert action in env.legal_actions


def test_collect_dmc_stores_chosen_row_and_target():
    env_dir = HERE.parent / "doudizhu-env"
    if str(env_dir) not in sys.path:
        sys.path.insert(0, str(env_dir))
    from doudizhu_env import DoudizhuEnv
    from dmc import TripleQ, collect_dmc_games

    env = DoudizhuEnv(objective="adp")
    models = TripleQ()
    batch = collect_dmc_games(env, models, min_games=1, epsilon=0.0)
    for pos in ("landlord", "landlord_up", "landlord_down"):
        assert len(batch[pos]) >= 1
        step = batch[pos][0]
        assert step["z"].shape == (5, 162)
        assert step["x"].dim() == 1
        assert "target" in step
        assert abs(abs(step["target"]) - 2 ** round(__import__("math").log2(abs(step["target"])))) < 1e-6


def test_cuda_update_if_available():
    import torch
    from dmc import QNet, dmc_update, resolve_device

    if not torch.cuda.is_available():
        print("skip test_cuda_update_if_available")
        return
    device = resolve_device()
    model = QNet(x_dim=8).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-2)
    batch = [
        {
            "z": torch.zeros(5, 162),
            "x": torch.zeros(8),
            "target": 2.0,
        }
        for _ in range(4)
    ]
    loss = dmc_update(model, opt, batch, max_grad_norm=40.0)
    assert loss == loss


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
    print("all passed")
