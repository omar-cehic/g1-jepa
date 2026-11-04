"""Tests for the foot contact force: the force file (scripts/build_force.py), the force metrics
(src/eval/rollout.py force_stats, evaluate(..., force=...)) and the force readout (scripts/eval_rollouts.py).

Uses data/processed/2026-10-07_182522, its force file FORCE, the first 3 tuning runs, and the saved runs of
tests/test_eval_rollouts_models.py. Nothing is written.

(a) Force file: for 200 random (run, k), the file value equals the mean of the raw 500 Hz foot_force over
    indices 10 (k + 1) .. 10 (k + 2) - 1, computed here directly from the raw run file (exactly).
(b) readout_rollout and readout_floor with a fake readout that returns its inputs: prediction j gets
    (true x_k, a_k) at j = 1, (x_hat of prediction j - 1, a_{k+j-1}) for j >= 2; the floor gets (x_{k+j-1}, a_{k+j-1}).
(c) force_stats on hand built errors: perfect prediction gives 0; an offset c on both feet gives rms, median
    and p90 = |c|; an offset c on one foot gives c / sqrt(2); the contact and rest subsets use the mask, and a
    step with no samples gives NaN.
(d) Train mean force: force_train_mean equals a loop over the train runs, and evaluate()'s force RMS at every j
    equals the RMS of (force over transition k+j-1 - mean) computed directly.
(e) For the MLP, the transformer world model and both JEPAs (force readout with 2 epochs to run fast):
    at j = 1 the rollout force equals the floor force exactly; and every state key of evaluate() with the force
    equals the one of the state path without it (np.array_equal): for the MLP the old evaluate(step, ...), for
    the transformer the unwrapped state predict, for the JEPAs the same predict's states without force=.
    For the JEPAs this only checks evaluate(); that the state probe itself is unchanged follows from its being
    fit (seeded) before the force readout, and is checked by comparing the full rerun with the old folders.
    Also prints the floor's force RMS at j = 1 next to the train mean's, as a sanity check that the readout learned.

Run: python tests/test_force.py (about 1.5 min with the GPU)
"""

import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from eval_rollouts import (DATASET, H, load_force, force_train_mean, readout_rollout, readout_floor,  # noqa: E402
                           add_state_force, mlp_predict, load_mlp_step, load_transformer_predict,
                           load_jepa_predicts, train_mean_predict)
from eval.rollout import evaluate, evaluate_force, force_stats, train_range, usable_starts  # noqa: E402
from test_eval_rollouts_models import MLP_DIR, TRANSFORMER_DIR, JEPA_DIRS, first_tuning_runs  # noqa: E402

READOUT_EPOCHS = 2
H_TEST = 5   # steps in the hand built indexing test (b)


def test_force_file(data, force):
    with open(os.path.join(REPO_ROOT, os.path.dirname(DATASET), "meta.json"), "r") as f:
        raw_dir = os.path.join(REPO_ROOT, json.load(f)["source_raw_dir"])
    rng = np.random.default_rng(0)
    n = 0
    for _ in range(200):
        r = rng.integers(len(data["run_ids"]))
        k = rng.integers(force.shape[1])
        raw = np.load(os.path.join(raw_dir, f"run_{int(data['run_ids'][r]):03d}.npz"))
        direct = raw["foot_force"][10 * (k + 1):10 * (k + 2)].mean(axis=0)
        assert raw["step"][k] == 10 * (k + 1) and raw["step"][k + 1] == 10 * (k + 2)
        assert np.array_equal(force[r, k], direct), (r, k, force[r, k], direct)
        n += 1
    print(f"(a) force file == mean of raw foot_force[10 (k + 1) : 10 (k + 2)] exactly, on {n} random (run, k)")


def test_readout_indexing():
    W, steps = 4, 30
    x_true = np.arange(steps, dtype=float)[:, None] * 10.0            # x of row i is 10 i
    actions = np.arange(steps - 1, dtype=float)[:, None] * np.ones(12)  # a_i is i
    starts = np.array([0, 3, 7, 12])
    x_hat = 1000.0 + np.arange(W * H_TEST, dtype=float).reshape(W, H_TEST, 1)  # prediction j at index j - 1
    fake = lambda x, a: np.concatenate([x, a[..., :1]], axis=-1)  # noqa: E731
    out = readout_rollout(fake, x_true, x_hat, actions, starts)
    floor = readout_floor(fake, x_true, actions, starts, H_TEST)
    for w, k in enumerate(starts):
        assert out[w, 0, 0] == x_true[k, 0] and out[w, 0, 1] == k
        for j in range(2, H_TEST + 1):
            assert out[w, j - 1, 0] == x_hat[w, j - 2, 0] and out[w, j - 1, 1] == k + j - 1
        for j in range(1, H_TEST + 1):
            assert floor[w, j - 1, 0] == x_true[k + j - 1, 0] and floor[w, j - 1, 1] == k + j - 1
    print("(b) readout inputs: j = 1 (true x_k, a_k), j >= 2 (prediction j - 1, a_(k+j-1)); "
          "floor (x_(k+j-1), a_(k+j-1))")



def test_force_stats():
    rng = np.random.default_rng(1)
    truth = rng.normal(300, 50, size=(6, 3, 2))
    contact = np.array([[True, False, False]] * 3 + [[False, False, False]] * 3)  # j = 3: no contact window
    m = force_stats(truth, truth, contact)
    assert all(np.all(m[f"{s}_all_force"] == 0) for s in ["rms", "median", "p90"])
    m = force_stats(truth - 7.0, truth, contact)
    for subset, steps in [("all", [0, 1, 2]), ("contact", [0]), ("rest", [0, 1, 2])]:
        for stat in ["rms", "median", "p90"]:
            assert np.allclose(m[f"{stat}_{subset}_force"][steps], 7.0, rtol=1e-12), (stat, subset)
    assert np.all(np.isnan(m["rms_contact_force"][1:])) and np.all(np.isnan(m["median_contact_force"][1:]))
    one = truth.copy()
    one[:, :, 0] += 7.0
    m = force_stats(one, truth, contact)
    assert np.allclose(m["median_all_force"], 7.0 / np.sqrt(2), rtol=1e-12)
    assert np.allclose(m["rms_all_force"], 7.0 / np.sqrt(2), rtol=1e-12)
    # mixed errors: windows 0-2 off by 3 on both feet, windows 3-5 by 4; contact at j = 1 only windows 0-2
    mixed = truth.copy()
    mixed[:3] += 3.0
    mixed[3:] += 4.0
    m = force_stats(mixed, truth, contact)
    assert np.isclose(m["rms_contact_force"][0], 3.0) and np.isclose(m["rms_rest_force"][0], 4.0)
    assert np.isclose(m["rms_all_force"][0], np.sqrt((9 + 16) / 2))
    print("(c) force_stats on hand built errors: 0, |c|, c / sqrt(2), subsets and empty steps as expected")


def test_train_mean(data, force):
    rows = [force[run][data["valid"][run]] for run in np.flatnonzero(data["split"] == "train")]
    mean = np.concatenate(rows).mean(axis=0)
    script_mean, _ = force_train_mean(data, force)
    assert np.allclose(script_mean, mean, rtol=0, atol=1e-9), (script_mean, mean)

    runs, _ = first_tuning_runs(data)
    sub = dict(data)
    sub["split"] = np.where(np.isin(np.arange(len(data["split"])), runs), "tuning", "unused")
    state_predict, _ = train_mean_predict(data, H)

    def predict(run, starts):
        return state_predict(run, starts), np.tile(script_mean, (len(starts), H, 1))
    m = evaluate(None, sub, "tuning", H, predict=predict, force=force)
    truth = np.concatenate([force[run, usable_starts(data["valid"][run], H)[:, None] + np.arange(H)]
                            for run in runs])                       # (W, H, 2): transition k+j-1
    rms = np.sqrt(((truth - script_mean) ** 2).mean(axis=(0, 2)))
    diff = np.abs(m["rms_all_force"] - rms).max()
    assert diff <= 1e-9, diff
    print(f"(d) train mean force {script_mean.round(2).tolist()} N == loop mean; evaluate() force RMS vs direct "
          f"for j = 1..{H}: max abs diff {diff:.1e} N; RMS at j = 1: {rms[0]:.2f} N")
    return rms[0]


def check_model(name, data, force, predict, floor_force, state_only, train_mean_rms):
    """(e) for one model: j = 1 rollout force == floor force; state keys with force == without."""
    runs, _ = first_tuning_runs(data)
    sub = dict(data)
    sub["split"] = np.where(np.isin(np.arange(len(data["split"])), runs), "tuning", "unused")
    max_diff = 0.0
    for run in runs:
        starts = usable_starts(data["valid"][run], H)
        _, f = predict(run, starts)
        max_diff = max(max_diff, np.abs(f[:, 0] - floor_force(run, starts)[:, 0]).max())
    assert max_diff == 0.0, max_diff

    bounds = train_range(data)
    with_force = evaluate(None, sub, "tuning", H, bounds=bounds, predict=predict, force=force)
    without = state_only(sub, bounds)
    assert set(without) == {k for k in with_force if not k.endswith("_force")}
    for key in without:
        assert np.array_equal(with_force[key], without[key], equal_nan=True), (name, key)
    floor = evaluate_force(floor_force, sub, force, "tuning", H)
    print(f"(e) {name}: j = 1 rollout force == floor force exactly; all {len(without)} state keys unchanged; "
          f"force RMS at j = 1: floor {floor['rms_all_force'][0]:.2f} N (train mean {train_mean_rms:.2f} N), "
          f"rollout at j = 10 {with_force['rms_all_force'][9]:.2f} N")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    data = dict(np.load(os.path.join(REPO_ROOT, DATASET)))
    force = load_force(data)
    test_force_file(data, force)
    test_readout_indexing()
    test_force_stats()
    train_mean_rms = test_train_mean(data, force)

    step = load_mlp_step(os.path.join(REPO_ROOT, MLP_DIR), data, device)
    predict, floor_force = add_state_force(mlp_predict(step, data), data, force, device, epochs=READOUT_EPOCHS)
    check_model("mlp", data, force, predict, floor_force,
                lambda sub, bounds: evaluate(step, sub, "tuning", H, bounds=bounds), train_mean_rms)

    state_predict, _ = load_transformer_predict(os.path.join(REPO_ROOT, TRANSFORMER_DIR), data, device)
    predict, floor_force = add_state_force(state_predict, data, force, device, epochs=READOUT_EPOCHS)
    check_model("transformer", data, force, predict, floor_force,
                lambda sub, bounds: evaluate(None, sub, "tuning", H, bounds=bounds, predict=state_predict),
                train_mean_rms)

    for run_dir in JEPA_DIRS:
        predict, _, floor_force, model = load_jepa_predicts(os.path.join(REPO_ROOT, run_dir), data, force, device,
                                                            force_epochs=READOUT_EPOCHS)
        states_only = lambda run, starts: predict(run, starts)[0]  # noqa: E731
        check_model(f"jepa {model.predictor_type} predictor", data, force, predict, floor_force,
                    lambda sub, bounds: evaluate(None, sub, "tuning", H, bounds=bounds, predict=states_only),
                    train_mean_rms)
    print("PASS test_force")


if __name__ == "__main__":
    main()
