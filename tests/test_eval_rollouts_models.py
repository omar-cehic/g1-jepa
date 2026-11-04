"""Tests for the window rollouts in scripts/eval_rollouts.py and src/eval/rollout.py
(rollout_window, evaluate(..., predict=...), the transformer world model, the JEPAs and the train mean).

Uses data/processed/2026-10-07_182522, the first 3 tuning runs, and the saved runs in MLP_DIR,
TRANSFORMER_DIR and JEPA_DIRS below. Nothing is written.

(a) The MLP through rollout_window (window of 1) and evaluate(None, ..., predict=...) gives exactly the
    same evaluate() output as the old evaluate(step, ...) (np.array_equal on every key).
(b) Window sliding with a fake step that returns the oldest state of its window (n = 10):
    prediction j equals s_{k-10+j} for j = 1..10, then repeats with period 10, and the action window
    given for prediction j is exactly a_{k+j-10}, ..., a_{k+j-1}. A start k < n - 1 is refused.
(c) Transformer world model, j = 1: the rollout's first prediction equals s_k + the denormalized output
    of the training code path (batch() and forward() in src/train.py) on the same pairs.
(d) Both JEPAs, j = 1: e_hat_{k+1} from the rollout equals z_hat of embed_windows() in score_jepa.py on
    the same pairs; and the floor's embedding for prediction j equals encode(normalized s_{k+j}).
(e) Train mean: every prediction equals the mean of data["states"][run, k] over the valid train
    transitions k (computed here with a loop over runs), and evaluate()'s RMS error at every step j
    (per feature and per group) equals the RMS of (truth - mean) computed directly.
(c) and (d) print the max abs difference and fail above 1e-5 (float32 results can differ in the last
bits with the batch size on the GPU).

Run: python tests/test_eval_rollouts_models.py (about a minute with the GPU)
"""

import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from eval_rollouts import (load_mlp_step, load_transformer_predict, encode_run, jepa_rollout,  # noqa: E402
                           true_embeddings, train_mean_predict, DATASET)
from eval.rollout import evaluate, rollout_window, train_range, usable_starts  # noqa: E402
from features import STATE_GROUPS  # noqa: E402
from models.mlp import normalize, denormalize  # noqa: E402
from score_jepa import load_jepa, embed_windows  # noqa: E402
from train import window_transitions, batch, forward  # noqa: E402
from train_jepa import window_pairs  # noqa: E402

H = 100
N_RUNS = 3
TOL = 1e-5
MLP_DIR = "results/train/2026-10-14_143647_mlp_w1024_lr0.001_s0"
TRANSFORMER_DIR = "results/train/2026-11-01_162810_mlp_w1024_lr0.001_tf_h10_s0"
JEPA_DIRS = ["results/train/2026-11-03_104900_jepa_d256_sw0.09_lr0.001_s3",
             "results/train/2026-11-03_113237_jepa_d256_sw0.09_lr0.001_tf_h10_s0"]


def first_tuning_runs(data):
    """Indices of the first N_RUNS tuning runs, and their positions among all tuning runs (0 .. N_RUNS - 1)."""
    runs = np.flatnonzero(data["split"] == "tuning")[:N_RUNS]
    return runs, np.arange(N_RUNS)


def pair_index(rows, local_run, starts, steps):
    """Positions in window_pairs()'s rows (sorted) of the pairs with s_k at k = starts in the local_run-th run of the split."""
    want = local_run * steps + starts
    idx = np.searchsorted(rows, want)
    assert np.array_equal(rows[idx], want)
    return idx


def test_mlp_new_path(data, device):
    """(a) The MLP through rollout_window (n = 1) and evaluate(predict=...) == the old evaluate(step, ...)."""
    runs, _ = first_tuning_runs(data)
    sub = dict(data)
    sub["split"] = np.where(np.isin(np.arange(len(data["split"])), runs), "tuning", "unused")
    bounds = train_range(data)
    step = load_mlp_step(os.path.join(REPO_ROOT, MLP_DIR), data, device)

    def predict(run, starts):
        window_step = lambda s, a: step(s[:, -1], a[:, -1])  # noqa: E731  window of 1: s_k and a_k
        return rollout_window(window_step, data["states"][run], data["actions"][run], starts, 1, H)

    old = evaluate(step, sub, "tuning", H, bounds=bounds)
    new = evaluate(None, sub, "tuning", H, bounds=bounds, predict=predict)
    assert old.keys() == new.keys()
    for key in old:
        assert np.array_equal(old[key], new[key], equal_nan=True), key
    print(f"(a) MLP: evaluate() through predict == old path exactly, on all {len(old)} keys, "
          f"{old['n_windows']} windows")


def test_window_sliding(data):
    """(b) A fake step that returns the oldest state of its window, n = 10."""
    n = 10
    runs, _ = first_tuning_runs(data)
    n_windows = 0
    for run in runs:
        x, actions = data["states"][run], data["actions"][run]
        starts = usable_starts(data["valid"][run], H)
        assert starts.min() >= 49
        seen = []

        def fake_step(x_window, a_window):
            assert x_window.shape == (len(starts), n, x.shape[1]) and a_window.shape == (len(starts), n, 12)
            seen.append(a_window.copy())
            return x_window[:, 0]

        preds = rollout_window(fake_step, x, actions, starts, n, H)
        assert preds.shape == (len(starts), H, x.shape[1]) and len(seen) == H
        for j in range(1, H + 1):
            # j = 1..10: s_{k-10+j}; after that the same 10 states again (period 10)
            assert np.array_equal(preds[:, j - 1], x[starts - 9 + (j - 1) % 10]), j
            # action window of prediction j: a_{k+j-10}, ..., a_{k+j-1}
            assert np.array_equal(seen[j - 1], actions[starts[:, None] + np.arange(j - 10, j)]), j
        n_windows += len(starts)
    try:
        rollout_window(fake_step, x, actions, np.array([8, 60]), n, H)
        raise RuntimeError("a window reaching before step 0 was not refused")
    except AssertionError:
        pass
    print(f"(b) window sliding: predictions s_(k-10+j) with period 10 and action windows a_(k+j-10..k+j-1) "
          f"exact for j = 1..{H} on {n_windows} windows; a start at k = 8 with n = 10 is refused")


def test_transformer_first_step(data, device):
    """(c) Transformer world model, j = 1: rollout == s_k + denormalized forward(batch()) of src/train.py."""
    predict, model = load_transformer_predict(os.path.join(REPO_ROOT, TRANSFORMER_DIR), data, device)
    x, _ = window_transitions(data, "tuning", model.history)
    rows = x[2]
    x = tuple(torch.from_numpy(v).to(device) for v in x)
    runs, local = first_tuning_runs(data)
    steps = data["states"].shape[1]
    max_diff, n_windows = 0.0, 0
    t0 = time.time()
    for run, i in zip(runs, local):
        starts = usable_starts(data["valid"][run], H)
        preds = predict(run, starts)                                           # (W, H, 34)
        idx = torch.from_numpy(pair_index(rows, i, starts, steps)).to(device)
        with torch.no_grad():
            out = forward(model, batch(model, x, idx)).cpu().numpy()
        expected = data["states"][run, starts] + denormalize(out, data, "delta")
        max_diff = max(max_diff, np.abs(preds[:, 0] - expected).max())
        assert np.all(np.isfinite(preds))
        n_windows += len(starts)
    print(f"(c) transformer world model (history {model.history}), j = 1: max abs diff to the training path "
          f"{max_diff:.3e} (raw units) on {n_windows} windows; {time.time() - t0:.1f} s for the {H} step rollouts")
    assert max_diff <= TOL, max_diff


def test_jepa_first_step_and_floor(data, device, run_dir):
    """(d) JEPA, j = 1: rollout e_hat_{k+1} == embed_windows() z_hat; floor embedding j == encode(s_{k+j})."""
    model = load_jepa(os.path.join(REPO_ROOT, run_dir), device)
    states, rows, a, actions = window_pairs(data, "tuning", model.history)
    _, z_hat = embed_windows(model, states, rows, a, actions, device)
    runs, local = first_tuning_runs(data)
    steps = data["states"].shape[1]
    max_diff, max_floor_diff, n_windows = 0.0, 0.0, 0
    t0 = time.time()
    for run, i in zip(runs, local):
        starts = usable_starts(data["valid"][run], H)
        z_pred = jepa_rollout(model, data, run, starts, H, device)             # (W, H, embed_dim)
        assert np.all(np.isfinite(z_pred))
        max_diff = max(max_diff, np.abs(z_pred[:, 0] - z_hat[pair_index(rows, i, starts, steps)]).max())

        floor = true_embeddings(encode_run(model, data, run, device), starts, H)  # (W, H, embed_dim)
        target_rows = starts[:, None] + np.arange(1, H + 1)                    # k + j, the rows evaluate() uses
        s = normalize(data["states"][run, target_rows.ravel()], data, "state").astype(np.float32)
        with torch.no_grad():
            z_true = model.encode(torch.from_numpy(s).to(device)).cpu().numpy().reshape(floor.shape)
        max_floor_diff = max(max_floor_diff, np.abs(floor - z_true).max())
        n_windows += len(starts)
    print(f"(d) JEPA {model.predictor_type} predictor, history {model.history}: j = 1 max abs diff to "
          f"embed_windows {max_diff:.3e}; floor vs encode(s_(k+j)) for j = 1..{H}: {max_floor_diff:.3e} "
          f"(embedding units) on {n_windows} windows; {time.time() - t0:.1f} s")
    assert max_diff <= TOL and max_floor_diff <= TOL, (max_diff, max_floor_diff)


def test_train_mean(data):
    """(e) Train mean: predictions == the mean of the valid train rows; RMS at every j == RMS of (truth - mean)."""
    # the mean, computed independently: loop over train runs, keep s_k of every valid transition k
    rows = [data["states"][run, np.flatnonzero(data["valid"][run])]
            for run in np.flatnonzero(data["split"] == "train")]
    mean = np.concatenate(rows).mean(axis=0)
    predict, script_mean = train_mean_predict(data, H)
    assert np.allclose(script_mean, mean, rtol=0, atol=1e-12), np.abs(script_mean - mean).max()

    runs, _ = first_tuning_runs(data)
    sub = dict(data)
    sub["split"] = np.where(np.isin(np.arange(len(data["split"])), runs), "tuning", "unused")
    truth = []
    for run in runs:
        starts = usable_starts(data["valid"][run], H)
        preds = predict(run, starts)
        assert preds.shape == (len(starts), H, mean.shape[0])
        assert np.all(preds == script_mean)
        truth.append(data["states"][run, starts[:, None] + np.arange(1, H + 1)])  # (W, H, 34) s_(k+j)
    truth = np.concatenate(truth)
    sq = (truth - script_mean) ** 2                                            # (all windows, H, 34)
    rms = np.sqrt(sq.mean(axis=0))                                             # (H, 34)

    m = evaluate(None, sub, "tuning", H, bounds=train_range(data), predict=predict)
    assert m["n_windows"] == len(truth)
    max_diff = np.abs(m["rms_all"] - rms).max()
    for group, cols in STATE_GROUPS.items():
        group_rms = np.sqrt(sq[:, :, cols].mean(axis=(0, 2)))                  # (H,)
        max_diff = max(max_diff, np.abs(m[f"rms_all_{group}"] - group_rms).max())
    print(f"(e) train mean: every prediction equals the mean of the valid train rows on {len(truth)} windows; "
          f"evaluate() RMS vs direct RMS of (truth - mean) for j = 1..{H}, per feature and per group: "
          f"max abs diff {max_diff:.3e} (raw units)")
    assert max_diff <= 1e-10, max_diff


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    data = dict(np.load(os.path.join(REPO_ROOT, DATASET)))
    runs, _ = first_tuning_runs(data)
    print(f"first {N_RUNS} tuning runs: run ids {data['run_ids'][runs].tolist()}")
    test_mlp_new_path(data, device)
    test_window_sliding(data)
    test_transformer_first_step(data, device)
    for run_dir in JEPA_DIRS:
        test_jepa_first_step_and_floor(data, device, run_dir)
    test_train_mean(data)
    print("all window rollout tests passed")


if __name__ == "__main__":
    main()
