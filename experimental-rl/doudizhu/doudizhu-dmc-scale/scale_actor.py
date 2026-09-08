"""Actor process: load latest Q weights, play games, push packed transitions."""

from __future__ import annotations

import sys
from pathlib import Path


def _setup_paths() -> None:
    here = Path(__file__).resolve().parent
    for extra in (
        here,
        here.parent / "doudizhu-dmc",
        here.parent / "doudizhu-env",
        here.parent / "DouZero",
    ):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))


def run_actor(
    actor_id: int,
    weights_path: str,
    queue,
    stop_event,
    epsilon: float,
    games: int,
) -> None:
    _setup_paths()
    import torch
    from dmc import TripleQ, collect_dmc_games
    from doudizhu_env import DoudizhuEnv
    from scale_io import pack_games

    env = DoudizhuEnv(objective="adp")
    models = TripleQ()
    for m in models.models.values():
        m.eval()
    last_mtime = -1.0
    weights = Path(weights_path)
    while not stop_event.is_set():
        if weights.exists():
            mtime = weights.stat().st_mtime
            if mtime > last_mtime:
                try:
                    state = torch.load(weights, map_location="cpu", weights_only=True)
                    models.load_state_dict(state)
                    last_mtime = mtime
                except (OSError, RuntimeError, EOFError):
                    pass
        fresh = collect_dmc_games(env, models, games, epsilon)
        if stop_event.is_set():
            break
        queue.put(pack_games(fresh))
