"""Run: python test_ppo_helpers.py  (from this directory)"""
import torch
from ppo_train import compute_returns, normalize_adv


def test_compute_returns_stops_at_done():
    rewards = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    dones = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    # gamma=1: after done at t=1, t=0 return = 1 + 1 = 2; t=1 return = 1; t=2 return = 1
    out = compute_returns(rewards, dones, gamma=1.0)
    assert torch.allclose(out, torch.tensor([2.0, 1.0, 1.0])), out


def test_normalize_adv_zero_mean_unit_std():
    adv = torch.tensor([1.0, 2.0, 3.0, 4.0])
    norm = normalize_adv(adv)
    assert abs(norm.mean().item()) < 1e-5
    assert abs(norm.std(unbiased=False).item() - 1.0) < 1e-4


if __name__ == "__main__":
    test_compute_returns_stops_at_done()
    test_normalize_adv_zero_mean_unit_std()
    print("OK")
