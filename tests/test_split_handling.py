"""Tests for the split handling of scripts/eval_rollouts.py, aggregate_seeds.py and plot_rollouts.py.

(a) Test split selection, without running any model:
    - every run of the dataset has exactly one split label (train, tuning or test); the test runs are 30 and share
      no run id with the train or tuning runs;
    - the window count of the selection loop of evaluate() (np.flatnonzero(data["split"] == split), then
      usable_starts) on tuning equals n_windows of a real tuning rollout (so the loop here is the one evaluate()
      uses); prints the test window count;
    - eval_rollouts.py accepts --split test (it then stops at the missing --run_dir, before loading any data)
      and refuses --split bogus with an error that lists test.
(c) Mixed splits are refused, on copies of real tuning rollout folders (meta.json and metrics.npz) in a temporary
    folder, some relabeled split "test" in meta.json (the numbers stay the tuning numbers):
    - aggregate_seeds.check_seeds accepts the unchanged copies and refuses a mixed pair;
    - plot_rollouts.py refuses a mixed pair within one label, a train_mean label on another split than the
      mlp label, an unknown label, and direct model folders whose force floors differ;
    - none of these refused runs writes a folder to results/figures.

Nothing in the repo is written; the temporary folder is removed at the end.
Run: python tests/test_split_handling.py (about 20 s)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from eval_rollouts import DATASET, H  # noqa: E402
from eval.rollout import usable_starts  # noqa: E402
from aggregate_seeds import check_seeds  # noqa: E402

ROLLOUTS = os.path.join(REPO_ROOT, "results", "rollouts")
MLP_S0 = os.path.join(ROLLOUTS, "2026-11-04_115633_mlp_2026-10-14_143647_mlp_w1024_lr0.001_s0_tuning")
MLP_S1 = os.path.join(ROLLOUTS, "2026-11-04_115701_mlp_2026-10-14_144118_mlp_w1024_lr0.001_s1_tuning")
TRAIN_MEAN = os.path.join(ROLLOUTS, "2026-11-04_121025_train_mean_tuning")
FIGURES = os.path.join(REPO_ROOT, "results", "figures")


def count_windows(data, split):
    """Number of windows evaluate() uses on one split: the same run selection and usable_starts."""
    return sum(len(usable_starts(data["valid"][run], H)) for run in np.flatnonzero(data["split"] == split))


def test_split_selection(data):
    labels = data["split"]
    assert set(np.unique(labels)) == {"train", "tuning", "test"}, np.unique(labels)
    test_runs = np.flatnonzero(labels == "test")
    assert len(test_runs) == 30, len(test_runs)
    other_ids = set(data["run_ids"][labels != "test"].tolist())
    assert not set(data["run_ids"][test_runs].tolist()) & other_ids, "a test run id is also a train or tuning run id"
    assert len(set(data["run_ids"].tolist())) == len(labels), "run ids are not unique"

    with open(os.path.join(MLP_S0, "meta.json"), "r") as f:
        tuning_windows = json.load(f)["n_windows"]
    assert count_windows(data, "tuning") == tuning_windows, (count_windows(data, "tuning"), tuning_windows)
    n_test = count_windows(data, "test")
    assert n_test > 0
    print(f"ok split selection: {len(test_runs)} test runs, run ids disjoint from train and tuning; "
          f"tuning windows {tuning_windows} (= n_windows of a real tuning rollout); test windows {n_test}")


def run_script(*args):
    """Run a script of this repo; returns (return code, stdout + stderr)."""
    result = subprocess.run([sys.executable] + list(args), cwd=REPO_ROOT, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def test_eval_rollouts_split_choices():
    code, out = run_script("scripts/eval_rollouts.py", "--model", "mlp", "--split", "test")
    assert code == 2 and "--run_dir is required" in out, out
    code, out = run_script("scripts/eval_rollouts.py", "--model", "mlp", "--split", "bogus")
    assert code == 2 and "invalid choice: 'bogus'" in out and "'test'" in out, out
    print("ok eval_rollouts.py accepts --split test and refuses --split bogus")


def copy_folder(src, dst_root, name, split=None, change_force_floor=False):
    """Copy meta.json and metrics.npz of a rollout folder to dst_root/name. split: write this split into meta.json.
    change_force_floor: add 1 N to floor_median_all_force."""
    dst = os.path.join(dst_root, name)
    os.makedirs(dst)
    with open(os.path.join(src, "meta.json"), "r") as f:
        meta = json.load(f)
    if split is not None:
        meta["split"] = split
    with open(os.path.join(dst, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if change_force_floor:
        m = np.load(os.path.join(src, "metrics.npz"))
        arrays = {k: m[k] for k in m.files}
        arrays["floor_median_all_force"] = arrays["floor_median_all_force"] + 1.0
        np.savez(os.path.join(dst, "metrics.npz"), **arrays)
    else:
        shutil.copy(os.path.join(src, "metrics.npz"), os.path.join(dst, "metrics.npz"))
    return dst


def expect_exit(func, *args):
    """The message of the SystemExit that func(*args) raises (fails if it does not)."""
    try:
        func(*args)
    except SystemExit as e:
        return str(e.code)
    raise AssertionError(f"{func.__name__} did not stop")


def test_mixed_splits(tmp):
    s0 = copy_folder(MLP_S0, tmp, "mlp_s0_tuning")
    s1 = copy_folder(MLP_S1, tmp, "mlp_s1_tuning")
    s1_test = copy_folder(MLP_S1, tmp, "mlp_s1_test", split="test")
    s1_floor = copy_folder(MLP_S1, tmp, "mlp_s1_changed_floor", change_force_floor=True)
    train_mean_test = copy_folder(TRAIN_MEAN, tmp, "train_mean_test", split="test")

    split = check_seeds([s0, s1])[2]
    assert split == "tuning", split
    msg = expect_exit(check_seeds, [s0, s1_test])
    assert "different splits" in msg, msg
    print("ok aggregate_seeds.check_seeds: same split accepted, mixed splits refused")

    figures_before = set(os.listdir(FIGURES))
    cases = [
        ("mixed splits within a label", ["--label", "mlp", s0, s1_test], 1, "different splits"),
        ("train_mean on another split", ["--label", "mlp", s0, s1, "--label", "train_mean", train_mean_test], 1,
         "different splits"),
        ("unknown label", ["--label", "my_model", s0, s1], 2, "without a fixed color"),
        ("direct model force floors differ", ["--label", "mlp", s0, s1_floor], 1,
         "cannot be drawn once"),
    ]
    for what, args, want_code, want_text in cases:
        code, out = run_script("scripts/plot_rollouts.py", *args)
        assert code == want_code and want_text in out, f"{what}: code {code}\n{out}"
        print(f"ok plot_rollouts.py refuses: {what}")
    assert set(os.listdir(FIGURES)) == figures_before, "a refused plot_rollouts.py run wrote to results/figures"
    print("ok no refused run wrote to results/figures")


def main():
    data = dict(np.load(os.path.join(REPO_ROOT, DATASET)))
    test_split_selection(data)
    test_eval_rollouts_split_choices()
    with tempfile.TemporaryDirectory() as tmp:
        test_mixed_splits(tmp)
    assert not os.path.exists(tmp), "the temporary folder was not removed"
    print("PASS test_split_handling")


if __name__ == "__main__":
    main()
