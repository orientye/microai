"""GameEnv-compatible greedy agents wrapping DMC Q-nets."""

from __future__ import annotations

from pathlib import Path

import torch

from dmc import POSITIONS, TripleQ


class DmcSeatAgent:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def act(self, infoset):
        from douzero.env.env import get_obs

        legal = infoset.legal_actions
        if len(legal) == 1:
            return legal[0]
        obs = get_obs(infoset)
        z = torch.as_tensor(obs["z_batch"], dtype=torch.float32)
        x = torch.as_tensor(obs["x_batch"], dtype=torch.float32)
        with torch.no_grad():
            q = self.model(z, x)
        return legal[int(torch.argmax(q).item())]


def load_dmc_players(ckpt_path: str | Path) -> dict:
    path = Path(ckpt_path)
    models = TripleQ()
    state = torch.load(path, map_location="cpu", weights_only=True)
    models.load_state_dict(state)
    return {pos: DmcSeatAgent(models[pos]) for pos in POSITIONS}


def eval_landlord_vs_random_deals(model, deals: list) -> dict:
    sys_paths()
    from eval_ruler import _load_players, play_deals_with_players
    from ruler_metrics import metrics_from_landlord_results

    rnd = _load_players("random", "random", "random")
    players = {
        "landlord": DmcSeatAgent(model),
        "landlord_up": rnd["landlord_up"],
        "landlord_down": rnd["landlord_down"],
    }
    results = play_deals_with_players(deals, players)
    out = metrics_from_landlord_results(results)
    out["num_deals"] = len(deals)
    return out


def eval_trio_vs_random_deals(models: TripleQ, deals: list) -> dict:
    sys_paths()
    from eval_ruler import _load_players, evaluate_seat_swap_players

    players_a = {pos: DmcSeatAgent(models[pos]) for pos in POSITIONS}
    rnd = _load_players("random", "random", "random")
    out = evaluate_seat_swap_players(
        players_a, rnd, deals, label_a="dmc", label_b="random"
    )
    for pos in POSITIONS:
        models[pos].train()
    return out


def eval_trio_vs_douzero_deals(models: TripleQ, deals: list, dz_dir: str | Path) -> dict:
    sys_paths()
    from eval_ruler import evaluate_seat_swap_players
    from vs_douzero import load_douzero_players

    players_a = {pos: DmcSeatAgent(models[pos]) for pos in POSITIONS}
    dz = load_douzero_players(dz_dir)
    out = evaluate_seat_swap_players(
        players_a, dz, deals, label_a="dmc", label_b=str(dz_dir)
    )
    for pos in POSITIONS:
        models[pos].train()
    return out


def sys_paths() -> None:
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    for extra in (
        here.parent / "eval-ruler",
        here.parent / "doudizhu-adp",
    ):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
