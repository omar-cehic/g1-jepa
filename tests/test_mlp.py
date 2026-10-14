"""Tests for src/models/mlp.py and src/train.py.

Uses data/processed/2026-10-07_182522.

(a) make_mlp(46, 34, 256, 3) has 152,354 parameters
    (46*256+256 + 2*(256*256+256) + 256*34+34) and maps (N, 46) to (N, 34).
(b) make_step with a fake model that returns the true normalized delta (normalized here by
    hand, cast to float32) gives back s_{k+1} within 1e-5 * delta_std per feature, on the
    valid transitions of the first 3 tuning runs. The fake model also checks that its input
    equals the hand normalized [s_k, a_k] cast to float32.
(c) make_step with a fake model that returns zeros gives s_k + delta_mean exactly.
(d) transitions() from train.py: 226,099 train pairs and 28,381 tuning pairs; the normalized
    train inputs and targets have mean ~0 and std ~1 per feature.
(e) Overfit: on CPU, a fresh MLP trained with train_step on one fixed batch of 512 train
    pairs for 500 steps (AdamW, lr 0.001) lowers the loss by at least 100x.
(f) Determinism: on CPU, train() twice with the same seed (20 steps on 640 train pairs)
    gives identical weights; a different seed gives different weights.

Nothing is written. Run: python tests/test_mlp.py
"""

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from models.mlp import make_mlp, make_step  # noqa: E402
import train  # noqa: E402

DATASET = os.path.join(REPO_ROOT, "data", "processed", "2026-10-07_182522", "dataset.npz")
N_RUNS = 3
CPU = torch.device("cpu")
CFG = {"width": 256, "depth": 3, "lr": 0.001, "weight_decay": 0.001, "grad_clip": 1.0}


def tuning_transitions(data):
    """s_k, a_k, s_{k+1} of the valid transitions of the first N_RUNS tuning runs."""
    runs = np.flatnonzero(data["split"] == "tuning")[:N_RUNS]
    states, valid = data["states"][runs], data["valid"][runs]
    return states[:, :-1][valid], data["actions"][runs][valid], states[:, 1:][valid]


def test_shape():
    model = make_mlp(46, 34, 256, 3)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 152354, n_params
    out = model(torch.zeros(5, 46))
    assert out.shape == (5, 34), out.shape
    print(f"(a) make_mlp(46, 34, 256, 3): {n_params} parameters, (5, 46) -> {tuple(out.shape)}")


def test_make_step(data):
    s, a, s_next = tuning_transitions(data)
    expected_x = np.concatenate([(s - data["state_mean"]) / data["state_std"],
                                 (a.astype(np.float64) - data["action_mean"]) / data["action_std"]],
                                axis=1).astype(np.float32)
    true_delta = ((s_next - s - data["delta_mean"]) / data["delta_std"]).astype(np.float32)

    def perfect(x):
        assert x.dtype == torch.float32
        assert np.array_equal(x.numpy(), expected_x)
        return torch.from_numpy(true_delta)
    pred = make_step(perfect, data, CPU)(s, a)
    assert pred.dtype == np.float64
    rel = np.abs(pred - s_next) / data["delta_std"]       # error in units of delta_std
    assert np.all(rel <= 1e-5), rel.max()
    print(f"(b) true delta through make_step: max |error| / delta_std = {rel.max():.1e} "
          f"on {len(s)} tuning transitions (max |normalized delta| {np.abs(true_delta).max():.1f})")

    def zeros(x):
        return torch.zeros(len(x), 34)
    pred = make_step(zeros, data, CPU)(s, a)
    assert np.array_equal(pred, s + data["delta_mean"])
    print("(c) zero output through make_step: s_k + delta_mean exactly")


def test_transitions(data):
    for split, n_expected in [("train", 226099), ("tuning", 28381)]:
        x, y = train.transitions(data, split)
        assert x.dtype == np.float32 and y.dtype == np.float32
        assert x.shape == (n_expected, 46) and y.shape == (n_expected, 34), (x.shape, y.shape)
    x, y = train.transitions(data, "train")
    worst = []
    for name, v in [("inputs", x), ("targets", y)]:
        v = v.astype(np.float64)
        mean_err = np.abs(v.mean(axis=0)).max()
        std_err = np.abs(v.std(axis=0) - 1).max()
        assert mean_err < 1e-4 and std_err < 1e-4, (name, mean_err, std_err)
        worst.append(f"{name} max |mean| {mean_err:.1e}, max |std - 1| {std_err:.1e}")
    print("(d) 226099 train pairs, 28381 tuning pairs; normalized train " + "; ".join(worst))


def test_overfit(data):
    x, y = train.transitions(data, "train")
    x, y = torch.from_numpy(x[:512]), torch.from_numpy(y[:512])
    train.set_seed(0)
    model = make_mlp(46, 34, CFG["width"], CFG["depth"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    losses = [train.train_step(model, optimizer, x, y, CFG["grad_clip"])[0] for _ in range(500)]
    with torch.no_grad():
        final = torch.nn.functional.mse_loss(model(x), y).item()
    assert final < losses[0] / 100, (losses[0], final)
    print(f"(e) overfit one batch of 512 for 500 steps: loss {losses[0]:.4f} -> {final:.6f} "
          f"({losses[0] / final:.0f}x lower)")


def test_determinism(data):
    x, y = train.transitions(data, "train")
    x, y = torch.from_numpy(x[:640]), torch.from_numpy(y[:640])
    cfg = dict(CFG, batch_size=64, epochs=2, warmup_epochs=1, seed=0)   # 2 x 10 = 20 steps
    print("    determinism runs:")
    weights = []
    for seed in [0, 0, 1]:
        model, _ = train.train(dict(cfg, seed=seed), x, y, x[:64], y[:64], CPU)
        weights.append(model.state_dict())
    same = all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])
    different = not any(torch.equal(weights[0][k], weights[2][k]) for k in weights[0])
    assert same and different
    print("(f) 20 steps on CPU: seed 0 twice gives identical weights; seed 1 gives different weights")


def main():
    data = dict(np.load(DATASET))
    test_shape()
    test_make_step(data)
    test_transitions(data)
    test_overfit(data)
    test_determinism(data)
    print("PASS test_mlp")


if __name__ == "__main__":
    main()
