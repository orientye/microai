"""Tests for parallel DMC: queue packing, checkpoint resume, actors fill replay."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
DMC = HERE.parent / "doudizhu-dmc"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(DMC) not in sys.path:
    sys.path.insert(0, str(DMC))


def test_pack_games_roundtrip_keeps_target():
    import torch
    from scale_io import pack_games, unpack_games

    fresh = {
        "landlord": [{"z": torch.zeros(5, 162), "x": torch.ones(373), "target": 4.0}],
        "landlord_up": [{"z": torch.zeros(5, 162), "x": torch.ones(484), "target": -4.0}],
        "landlord_down": [],
    }
    back = unpack_games(pack_games(fresh))
    assert back["landlord"][0]["target"] == 4.0
    assert back["landlord_up"][0]["target"] == -4.0
    assert back["landlord_down"] == []
    assert back["landlord"][0]["z"].shape == (5, 162)
    assert back["landlord"][0]["x"].shape == (373,)


def test_checkpoint_roundtrip_restores_update_and_weights():
    import tempfile
    import torch
    from dmc import TripleQ
    from scale_io import load_checkpoint, save_checkpoint

    models = TripleQ()
    opts = models.optimizers()
    before = {k: v.detach().clone() for k, v in models["landlord"].state_dict().items()}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pth"
        save_checkpoint(path, models, opts, update=7, extra={"best_wp": 0.12})
        models2 = TripleQ()
        opts2 = models2.optimizers()
        meta = load_checkpoint(path, models2, opts2)
        assert meta["update"] == 7
        assert meta["best_wp"] == 0.12
        for k, v in before.items():
            assert torch.equal(models2["landlord"].state_dict()[k], v)


def test_drain_queue_into_replay():
    import torch
    from queue import Queue
    from dmc import POSITIONS, Replay
    from scale_io import drain_queue, pack_games

    q = Queue()
    q.put(
        pack_games(
            {
                "landlord": [{"z": torch.zeros(5, 162), "x": torch.zeros(373), "target": 1.0}],
                "landlord_up": [],
                "landlord_down": [],
            }
        )
    )
    replays = {p: Replay() for p in POSITIONS}
    n = drain_queue(q, replays)
    assert n == 1
    assert len(replays["landlord"]) == 1
    assert replays["landlord"].sample(1)[0]["target"] == 1.0


def test_two_actors_put_at_least_one_game_each():
    import multiprocessing as mp
    from scale_actor import run_actor

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    stop = ctx.Event()
    weights = HERE / "_test_actor_weights.pth"
    from dmc import TripleQ
    import torch

    models = TripleQ()
    torch.save(models.state_dict(), weights)
    procs = [
        ctx.Process(
            target=run_actor,
            args=(i, str(weights), q, stop, 0.0, 1),
        )
        for i in range(2)
    ]
    for p in procs:
        p.start()
    got = [q.get(timeout=120) for _ in range(2)]
    stop.set()
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    weights.unlink(missing_ok=True)
    assert len(got) == 2
    for packed in got:
        n = sum(len(packed[p]) for p in packed)
        assert n >= 1


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
    print("all passed")
