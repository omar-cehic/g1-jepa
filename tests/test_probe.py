"""Tests for src/eval/probe.py and scripts/score_jepa.py.

(a) Exact recovery: synthetic float64 Z (1000, 64), known W (64, 34) and b (34,), Y = Z W + b.
    The fitted W and b match within 1e-8 and the mean squared error of the probe is below 1e-16.
(b) Collapsed input: Z (1000, 64) with every row identical (what a collapsed encoder gives).
    No crash; the probe output equals the mean of Y in every row, so the mean squared error of
    each feature equals the variance of Y (both within 1e-10).
(c) score_run() on one real run (the 2 epoch smoke run at sigreg_weight 0) gives exactly the same
    row twice: there is no randomness in scoring. Uses the GPU if there is one.
(d) MLP probe identity check: fit_mlp_probe with Z = the normalized train states (34 numbers) and
    Y = the same states. On the tuning states the mean squared error is below 0.01: the probe
    setup can learn an easy map. Uses the GPU if there is one.
(e) MLP probe determinism on the CPU: fit_mlp_probe twice with the same seed on 2000 synthetic
    rows for 2 epochs gives identical weights.
(f) MLP probe default size: on the same synthetic rows, fit_mlp_probe with explicit width=256,
    depth=2 gives weights identical to the call with the defaults (CPU, seed 0, 2 epochs).

Nothing is written. Run: python tests/test_probe.py
"""

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from eval.probe import fit_linear_probe, apply_probe, fit_mlp_probe, apply_mlp_probe  # noqa: E402
from score_jepa import score_run  # noqa: E402
from train_jepa import pairs  # noqa: E402

RUN_DIR = os.path.join(REPO_ROOT, "results", "train", "2026-10-25_121950_jepa_d128_sw0_lr0.001_s0")
DATASET = os.path.join(REPO_ROOT, "data", "processed", "2026-10-07_182522", "dataset.npz")


def test_exact_recovery():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(1000, 64))
    W_true = rng.normal(size=(64, 34))
    b_true = rng.normal(size=34)
    Y = Z @ W_true + b_true
    W, b = fit_linear_probe(Z, Y)
    w_err = np.abs(W - W_true).max()
    b_err = np.abs(b - b_true).max()
    mse = np.mean((apply_probe(Z, W, b) - Y) ** 2)
    assert w_err < 1e-8 and b_err < 1e-8 and mse < 1e-16, (w_err, b_err, mse)
    print(f"(a) exact recovery: max |W error| {w_err:.1e}, max |b error| {b_err:.1e}, mse {mse:.1e}")


def test_collapsed():
    rng = np.random.default_rng(1)
    Z = np.tile(rng.normal(size=64), (1000, 1))           # every row the same
    Y = rng.normal(size=(1000, 34)) * rng.uniform(0.5, 2, size=34) + rng.normal(size=34)
    W, b = fit_linear_probe(Z, Y)
    out = apply_probe(Z, W, b)
    mean_err = np.abs(out - Y.mean(axis=0)).max()
    mse = np.mean((out - Y) ** 2, axis=0)                 # per feature
    var_err = np.abs(mse - Y.var(axis=0)).max()
    assert mean_err < 1e-10 and var_err < 1e-10, (mean_err, var_err)
    print(f"(b) collapsed Z: max |output - mean of Y| {mean_err:.1e}, "
          f"max |per feature mse - variance of Y| {var_err:.1e} (variances {Y.var(axis=0).min():.2f} "
          f"to {Y.var(axis=0).max():.2f})")


def test_deterministic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first = score_run(RUN_DIR, device)
    second = score_run(RUN_DIR, device)
    assert first == second, (first, second)
    print(f"(c) {first['folder']} scored twice on {device}: identical rows, "
          f"score {first['score']!r}, present_mse {first['present_mse']!r}, "
          f"no_change_mse {first['no_change_mse']!r}")


def test_mlp_identity():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = dict(np.load(DATASET))
    s_train, _, _ = pairs(data, "train")
    s_tune, _, _ = pairs(data, "tuning")
    model = fit_mlp_probe(s_train, s_train, device, seed=0)
    mse = np.mean((apply_mlp_probe(model, s_tune, device) - s_tune.astype(np.float64)) ** 2)
    assert mse < 0.01, mse
    print(f"(d) MLP probe identity on {device}: {len(s_train)} train states, "
          f"tuning mse {mse:.2e} (must be below 0.01)")


def test_mlp_deterministic():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(2000, 64))
    Y = rng.normal(size=(2000, 34))
    device = torch.device("cpu")
    first = fit_mlp_probe(Z, Y, device, seed=0, epochs=2).state_dict()
    second = fit_mlp_probe(Z, Y, device, seed=0, epochs=2).state_dict()
    assert first.keys() == second.keys()
    for key in first:
        assert torch.equal(first[key], second[key]), key
    print(f"(e) MLP probe fit twice on cpu with seed 0: identical weights ({len(first)} tensors)")


def test_mlp_default_size():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(2000, 64))
    Y = rng.normal(size=(2000, 34))
    device = torch.device("cpu")
    default = fit_mlp_probe(Z, Y, device, seed=0, epochs=2).state_dict()
    explicit = fit_mlp_probe(Z, Y, device, seed=0, epochs=2, width=256, depth=2).state_dict()
    assert default.keys() == explicit.keys()
    for key in default:
        assert torch.equal(default[key], explicit[key]), key
    print(f"(f) MLP probe with explicit width=256, depth=2 on cpu: identical weights to the default "
          f"call ({len(default)} tensors)")


def main():
    test_exact_recovery()
    test_collapsed()
    test_deterministic()
    test_mlp_identity()
    test_mlp_deterministic()
    test_mlp_default_size()
    print("PASS test_probe")


if __name__ == "__main__":
    main()
