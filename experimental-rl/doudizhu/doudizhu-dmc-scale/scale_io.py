"""Pack transitions for queues and save / load scale checkpoints."""

from __future__ import annotations

import os
import time
from pathlib import Path
from queue import Empty

import torch

from dmc import POSITIONS, Replay


def pack_games(fresh: dict) -> dict:
    packed = {}
    for pos in POSITIONS:
        packed[pos] = [
            {
                "z": s["z"].detach().cpu().contiguous(),
                "x": s["x"].detach().cpu().contiguous(),
                "target": float(s["target"]),
            }
            for s in fresh.get(pos, [])
        ]
    return packed


def unpack_games(packed: dict) -> dict:
    out = {}
    for pos in POSITIONS:
        out[pos] = [
            {
                "z": s["z"].clone() if torch.is_tensor(s["z"]) else torch.as_tensor(s["z"]),
                "x": s["x"].clone() if torch.is_tensor(s["x"]) else torch.as_tensor(s["x"]),
                "target": float(s["target"]),
            }
            for s in packed.get(pos, [])
        ]
    return out


def drain_queue(q, replays: dict[str, Replay], max_items: int | None = None) -> int:
    n = 0
    while True:
        try:
            packed = q.get_nowait()
        except Empty:
            break
        fresh = unpack_games(packed)
        for pos in POSITIONS:
            replays[pos].add_many(fresh[pos])
        n += 1
        if max_items is not None and n >= max_items:
            break
    return n


def atomic_save(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    last_err: OSError | None = None
    for _ in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last_err = exc
            time.sleep(0.05)
    if last_err is not None:
        raise last_err


def save_checkpoint(
    path: Path,
    models,
    opts: dict,
    update: int,
    extra: dict | None = None,
) -> None:
    payload = {
        "models": models.state_dict(),
        "optimizers": {p: opt.state_dict() for p, opt in opts.items()},
        "update": int(update),
    }
    if extra:
        payload.update(extra)
    atomic_save(Path(path), payload)


def load_checkpoint(path: Path, models, opts: dict) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    models.load_state_dict(payload["models"])
    saved_opts = payload.get("optimizers", {})
    for pos, opt in opts.items():
        if pos in saved_opts:
            opt.load_state_dict(saved_opts[pos])
    return payload
