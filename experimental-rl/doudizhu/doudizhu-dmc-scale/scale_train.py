"""Learner: drain actor queue, MSE(Q,G), publish weights, resume from checkpoint."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

HERE = Path(__file__).resolve().parent
DMC = HERE.parent / "doudizhu-dmc"
ENV_DIR = HERE.parent / "doudizhu-env"
RULER = HERE.parent / "eval-ruler"
ADP = HERE.parent / "doudizhu-adp"
DOUZERO = HERE.parent / "DouZero"
for _p in (HERE, DMC, ENV_DIR, RULER, ADP, DOUZERO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dmc import ADP_GRAD_CLIP, POSITIONS, Replay, TripleQ, dmc_update, resolve_device
from dmc_agent import eval_trio_vs_douzero_deals, eval_trio_vs_random_deals
from scale_actor import run_actor
from scale_io import atomic_save, drain_queue, load_checkpoint, save_checkpoint

DEFAULT_INIT = DMC / "dmc_adp.pth"
ACTOR_WEIGHTS = HERE / "actor_weights.pth"
CKPT = HERE / "dmc_scale_last.pth"
SAVE_BEST = HERE / "dmc_scale.pth"
EVAL_DATA = RULER / "eval_data.pkl"
DZ_DIR = DOUZERO / "baselines" / "douzero_ADP"


def _load_init(models: TripleQ, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    landlord = state.get("landlord") if isinstance(state, dict) else None
    if isinstance(landlord, dict) and "lstm.weight_ih_l0" in landlord:
        models.load_state_dict(state)
        return
    models["landlord"].load_state_dict(state)


def _load_eval_deals(n: int, start: int) -> list:
    if not EVAL_DATA.exists():
        raise SystemExit(f"missing {EVAL_DATA}")
    with EVAL_DATA.open("rb") as f:
        deals = pickle.load(f)
    return deals[start : start + n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actors", type=int, default=4)
    parser.add_argument("--max_updates", type=int, default=200)
    parser.add_argument("--min_games", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--eval_deals", type=int, default=10)
    parser.add_argument("--eval_start", type=int, default=800)
    parser.add_argument("--init", type=str, default=str(DEFAULT_INIT))
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--ckpt", type=str, default=str(CKPT))
    parser.add_argument("--vs_douzero", action="store_true")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    device = resolve_device(args.device or None)
    print(f"device={device}")
    ctx = mp.get_context("spawn")
    models = TripleQ()
    models.to(device)
    opts = models.optimizers()
    start_update = 0
    best_wp = -1.0
    best_adp = float("-inf")
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise SystemExit(f"missing resume {resume_path}")
        meta = load_checkpoint(resume_path, models, opts)
        start_update = int(meta.get("update", 0))
        best_wp = float(meta.get("best_wp", best_wp))
        best_adp = float(meta.get("best_adp", best_adp))
        print(f"resumed {resume_path} update={start_update} best_wp={best_wp:.3f}")
    else:
        init_ckpt = Path(args.init)
        if init_ckpt.exists():
            _load_init(models, init_ckpt)
            print(f"init from {init_ckpt}")

    atomic_save(ACTOR_WEIGHTS, models.state_dict())
    replays = {p: Replay() for p in POSITIONS}
    q = ctx.Queue(maxsize=64)
    stop = ctx.Event()
    procs = [
        ctx.Process(
            target=run_actor,
            args=(i, str(ACTOR_WEIGHTS), q, stop, args.epsilon, 1),
        )
        for i in range(args.actors)
    ]
    for p in procs:
        p.start()

    deals = _load_eval_deals(args.eval_deals, args.eval_start)
    dz_ok = (DZ_DIR / "landlord.ckpt").exists()
    use_dz = args.vs_douzero and dz_ok
    if args.vs_douzero and not dz_ok:
        print("DouZero ADP missing; eval vs random")

    try:
        for update in range(start_update + 1, args.max_updates + 1):
            t0 = time.time()
            got = 0
            deadline = t0 + 120.0
            while got < args.min_games and time.time() < deadline:
                got += drain_queue(q, replays)
                if got < args.min_games:
                    time.sleep(0.05)
            got += drain_queue(q, replays)
            losses = []
            for pos in POSITIONS:
                loss = 0.0
                if len(replays[pos]) >= 1:
                    for _ in range(args.epochs):
                        loss = dmc_update(
                            models[pos],
                            opts[pos],
                            replays[pos].sample(args.batch),
                            max_grad_norm=ADP_GRAD_CLIP,
                        )
                losses.append(loss)
            atomic_save(ACTOR_WEIGHTS, models.state_dict())
            save_checkpoint(
                Path(args.ckpt),
                models,
                opts,
                update,
                extra={"best_wp": best_wp, "best_adp": best_adp},
            )
            do_eval = update == start_update + 1 or update % args.eval_every == 0
            if do_eval or update == args.max_updates:
                if use_dz:
                    stats = eval_trio_vs_douzero_deals(models, deals, DZ_DIR)
                else:
                    stats = eval_trio_vs_random_deals(models, deals)
                wp, adp = stats["wp_a"], stats["adp_a"]
                if wp > best_wp or (wp == best_wp and adp > best_adp):
                    best_wp, best_adp = wp, adp
                    atomic_save(SAVE_BEST, models.state_dict())
                vs = "douzero" if use_dz else "random"
                print(
                    f"update {update}/{args.max_updates} "
                    f"games={got} replay_ll={len(replays['landlord'])} "
                    f"loss_ll={losses[0]:.3f} "
                    f"eval_{vs}_wp={wp:.3f} eval_adp={adp:.3f} "
                    f"best_wp={best_wp:.3f} sec={time.time() - t0:.1f}"
                )
            else:
                print(
                    f"update {update}/{args.max_updates} "
                    f"games={got} replay_ll={len(replays['landlord'])} "
                    f"loss_ll={losses[0]:.3f} sec={time.time() - t0:.1f}"
                )
    finally:
        stop.set()
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()


if __name__ == "__main__":
    main()
