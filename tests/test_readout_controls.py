"""Tests for scripts/readout_controls.py.

Uses data/processed/2026-10-07_182522.

(a) one_step_mse with a fake model that returns the true normalized delta (cast to float32),
    wrapped by make_step, on the first 3 tuning runs: the error is below 1e-8 for every feature.
    What is left is the float32 rounding of the delta.
(b) standardize uses train statistics only: on small synthetic embeddings (std about 0.03, like an
    untrained encoder), the standardized train embeddings have mean 0 and std 1 per number (within
    1e-6), and the tuning embeddings are shifted and scaled with those same train numbers.

Nothing is written. Run: python tests/test_readout_controls.py
"""

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from models.mlp import make_step, normalize  # noqa: E402
from readout_controls import one_step_mse, standardize, DATASET  # noqa: E402


def test_one_step_perfect_model():
    data = dict(np.load(os.path.join(REPO_ROOT, DATASET)))
    first3 = np.flatnonzero(data["split"] == "tuning")[:3]
    small = dict(data)
    for key in ["states", "actions", "valid", "split"]:
        small[key] = data[key][first3]
    states, valid = small["states"], small["valid"]
    delta = (states[:, 1:] - states[:, :-1])[valid]       # same order as in one_step_mse
    true_delta = torch.from_numpy(normalize(delta, small, "delta").astype(np.float32))

    def fake_model(x):
        assert len(x) == len(true_delta)
        return true_delta

    step = make_step(fake_model, small, torch.device("cpu"))
    mse = one_step_mse(step, small, "tuning")
    assert mse.shape == (34,) and mse.max() < 1e-8, mse
    print(f"(a) fake model with the true normalized delta (float32) on tuning runs {first3.tolist()} "
          f"({len(true_delta)} transitions): max per feature error {mse.max():.1e}, mean {mse.mean():.1e}")


def test_standardize_train_stats():
    rng = np.random.default_rng(0)
    z_train = rng.normal(size=(500, 16)) * 0.03 + rng.normal(size=16)
    z_tune = rng.normal(size=(200, 16)) * 0.05 + rng.normal(size=16)   # a different distribution
    st_train, st_tune = standardize(z_train, z_tune)
    mean_err = np.abs(st_train.mean(axis=0)).max()
    std_err = np.abs(st_train.std(axis=0) - 1).max()
    tune_err = np.abs(st_tune - (z_tune - z_train.mean(axis=0)) / z_train.std(axis=0)).max()
    assert mean_err < 1e-6 and std_err < 1e-6 and tune_err < 1e-12, (mean_err, std_err, tune_err)
    print(f"(b) standardize: train max |mean| {mean_err:.1e}, max |std - 1| {std_err:.1e}; "
          f"tuning uses the train mean and std (max difference {tune_err:.1e})")


def main():
    test_one_step_perfect_model()
    test_standardize_train_stats()
    print("PASS test_readout_controls")


if __name__ == "__main__":
    main()
