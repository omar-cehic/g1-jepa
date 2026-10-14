"""Tests for the robust statistics of src/eval/rollout.py evaluate (median, p90, diverged_frac).

Uses data/processed/2026-10-07_182522 (tuning split).

(a) The median and 90th percentile of the per window group error at j = 10, 50, 100 for the w1024
    lr 0.001 MLP equal the ones computed from results/analysis/2026-10-14_135432_mlp_w1024
    (per_window.npz of scripts/analyze_divergence.py), to 1e-12 relative. "all" as asked; contact
    and rest too, with the contact label of each window taken from the dataset.
(b) Perfect model (looks up the true next state of its input): diverged_frac is 0 at every step,
    and the median and p90 are 0 (true states never leave the train range).
(c) diverged_frac never decreases with j (w1024 MLP and constant velocity).
(d) outside_range on hand built predictions: gravity_z is ignored, NaN and inf count as outside,
    a value equal to the train min or max is inside.

Nothing is written. Run: python tests/test_robust_stats.py
"""

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from eval_rollouts import load_mlp_step, DATASET  # noqa: E402
from eval.rollout import evaluate, outside_range, train_range, SUBSETS, GRAVITY_Z  # noqa: E402
from features import STATE_GROUPS  # noqa: E402
from models import constant_velocity  # noqa: E402

H = 100
RUN_DIR = os.path.join(REPO_ROOT, "results", "train", "2026-10-14_123253_mlp_w1024_lr0.001_s0")
PER_WINDOW = os.path.join(REPO_ROOT, "results", "analysis", "2026-10-14_135432_mlp_w1024", "per_window.npz")
STEPS = [10, 50, 100]  # the steps saved in per_window.npz


def test_matches_per_window(m, data):
    pw = np.load(PER_WINDOW)
    assert pw["report_steps"].tolist() == STEPS
    assert len(pw["window_run"]) == m["n_windows"]
    worst = 0.0
    for i, j in enumerate(STEPS):
        contact = data["contact_step"][pw["window_run"], pw["window_start"] + j - 1]  # transition k+j-1
        for subset, mask in [("all", np.ones_like(contact)), ("contact", contact), ("rest", ~contact)]:
            for group in STATE_GROUPS:
                err = np.sqrt(pw[f"mse_{group}"][mask, i])     # per window group error at step j
                for stat, expected in [("median", np.median(err)), ("p90", np.percentile(err, 90))]:
                    got = m[f"{stat}_{subset}_{group}"][j - 1]
                    rel = abs(got - expected) / expected
                    assert rel <= 1e-12, (subset, group, stat, j, got, expected)
                    worst = max(worst, rel)
    print(f"(a) w1024 median and p90 at j = {STEPS} (all, contact, rest) == per_window.npz:"
          f" max relative diff {worst:.1e}")


def test_perfect(data, bounds):
    tuning = np.flatnonzero(data["split"] == "tuning")
    next_row = {}
    for run in tuning:
        for k in range(data["states"].shape[1] - 1):
            next_row[data["states"][run, k].tobytes()] = (run, k)

    def lookup(s, actions):
        return np.array([data["states"][run, k + 1] for run, k in (next_row[x.tobytes()] for x in s)])

    m = evaluate(lookup, data, "tuning", H, bounds=bounds)
    assert np.all(m["diverged_frac"] == 0.0)
    for subset in SUBSETS:
        for group in STATE_GROUPS:
            assert np.all(m[f"median_{subset}_{group}"] == 0.0) and np.all(m[f"p90_{subset}_{group}"] == 0.0)
    print(f"(b) perfect model: diverged_frac 0 at every step, median and p90 exactly 0, {m['n_windows']} windows")


def test_monotone(name, m):
    f = m["diverged_frac"]
    assert f.shape == (H,) and np.all(np.diff(f) >= 0), f
    print(f"(c) {name}: diverged_frac never decreases; j = 1, 10, 50, 100: "
          + ", ".join(f"{f[j - 1]:.4f}" for j in [1, 10, 50, 100]))


def test_outside_range():
    lo, hi = np.zeros(34), np.ones(34)
    preds = np.full((5, 2, 34), 0.5)          # 5 windows, 2 steps, all inside
    preds[0, 0] = 0.0                         # window 0: equal to the min: inside
    preds[1, 1] = 1.0                         # window 1: equal to the max: inside
    preds[2, 0, GRAVITY_Z] = -5.0             # window 2: only gravity_z outside: ignored
    preds[3, 1, 0] = 1.5                      # window 3: feature 0 above the max at step 2
    preds[4, 0, 7] = np.nan                   # window 4: NaN at step 1
    preds[4, 1, 7] = np.inf                   #           and inf at step 2
    expected = np.array([[False, False], [False, False], [False, False], [False, True], [True, True]])
    assert np.array_equal(outside_range(preds, lo, hi), expected)
    print("(d) outside_range on hand built predictions: as expected")


def main():
    data = dict(np.load(os.path.join(REPO_ROOT, DATASET)))
    bounds = train_range(data)
    test_outside_range()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    m = evaluate(load_mlp_step(RUN_DIR, data, device), data, "tuning", H, bounds=bounds)
    test_matches_per_window(m, data)
    test_perfect(data, bounds)
    test_monotone("w1024 lr 0.001", m)
    test_monotone("constant velocity", evaluate(constant_velocity.step, data, "tuning", H, bounds=bounds))
    print("PASS test_robust_stats")


if __name__ == "__main__":
    main()
