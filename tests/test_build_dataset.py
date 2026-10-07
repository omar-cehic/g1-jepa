"""Tests for scripts/build_dataset.py.

1. No transition containing a push and no transition with k < first_valid_k is valid.
2. Valid count per run = 950 minus the pushes at rows >= 50. (A push at row 49 falls in
   transition 48, which is already invalid as k < 49, so it does not lower the count.)
3. No run id is in two splits, drop_runs are in none, and the split sizes are right.
4. For 3 random runs, stored states equal make_state on the raw npz exactly.
5. Recomputing the stats from the training runs (written again here, not with the
   script's function) gives the stored values, and stats over all splits differ.
6. Normalized valid training states have mean ~0 and std ~1 per feature.

With no argument, builds the dataset in memory from configs/build_dataset.yaml.
With a folder, checks that folder's dataset.npz (and reads its copied config).

Run: python tests/test_build_dataset.py [data/processed/<folder>]
"""

import os
import sys

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import build_dataset  # noqa: E402
from features import make_state  # noqa: E402


def raw_path(cfg, run):
    return os.path.join(REPO_ROOT, cfg["raw_dir"], f"run_{run:03d}.npz")


def main():
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        with open(os.path.join(folder, "build_dataset.yaml")) as f:
            cfg = yaml.safe_load(f)
        data = np.load(os.path.join(folder, "dataset.npz"))
        print(f"checking {folder}")
    else:
        with open(os.path.join(REPO_ROOT, "configs", "build_dataset.yaml")) as f:
            cfg = yaml.safe_load(f)
        data = build_dataset.build(cfg)
        print("checking an in-memory build")

    run_ids, split, valid = data["run_ids"], data["split"], data["valid"]
    first = cfg["first_valid_k"]
    assert first == 49
    assert valid.shape == (len(run_ids), 999) and data["states"].shape == (len(run_ids), 1000, 34)
    assert data["actions"].shape == (len(run_ids), 999, 12)

    # 1 and 2
    for r, run in enumerate(run_ids):
        push_row = np.load(raw_path(cfg, run))["push_row"]
        assert not valid[r, :first].any(), f"run {run}: valid transition with k < {first}"
        assert not valid[r, push_row - 1].any(), f"run {run}: valid push transition"
        expected = 950 - np.sum(push_row - 1 >= first)
        assert valid[r].sum() == expected, f"run {run}: {valid[r].sum()} valid, expected {expected}"
    print("1, 2 ok: no push or k < 49 transition is valid; valid count = 950 - pushes at rows >= 50")

    # 3
    assert len(np.unique(run_ids)) == len(run_ids)  # one label per run, so no run is in two splits
    groups = {name: set(run_ids[split == name].tolist()) for name in ["train", "tuning", "test"]}
    assert not (groups["train"] & groups["tuning"]) and not (groups["train"] & groups["test"])
    assert not (groups["tuning"] & groups["test"])
    assert sum(len(g) for g in groups.values()) == len(run_ids) == 300 - len(cfg["drop_runs"])
    for run in cfg["drop_runs"]:
        assert all(run not in g for g in groups.values()), f"run {run} is in a split"
    sizes = sorted([len(groups["train"]) - cfg["n_train"], len(groups["tuning"]) - cfg["n_tuning"],
                    len(groups["test"]) - cfg["n_test"]])
    assert sizes == [-1, 0, 0], sizes  # exactly one split lost exactly one run
    print(f"3 ok: splits disjoint, {cfg['drop_runs']} in none, sizes "
          f"{[len(g) for g in groups.values()]}")

    # 4
    rng = np.random.default_rng(123)
    for r in rng.choice(len(run_ids), size=3, replace=False):
        d = np.load(raw_path(cfg, run_ids[r]))
        state, _ = make_state(d["qpos"], d["qvel"])
        assert data["states"].dtype == np.float64
        assert np.array_equal(data["states"][r], state), f"run {run_ids[r]}: states differ"
        assert np.array_equal(data["actions"][r], d["target_q"][:999])
    print(f"4 ok: states and actions equal the raw data exactly for 3 random runs")

    # 5
    def stats(mask_runs):
        s, a, ds = [], [], []
        for r in np.flatnonzero(mask_runs):
            for k in np.flatnonzero(valid[r]):
                s.append(data["states"][r, k])
                a.append(data["actions"][r, k].astype(np.float64))
                ds.append(data["states"][r, k + 1] - data["states"][r, k])
        out = {}
        for name, x in [("state", s), ("action", a), ("delta", ds)]:
            x = np.array(x)
            out[name + "_mean"] = x.mean(axis=0)
            out[name + "_std"] = x.std(axis=0)
        return out

    train_stats = stats(split == "train")
    all_stats = stats(np.ones(len(run_ids), dtype=bool))
    for name, value in train_stats.items():
        assert np.allclose(data[name], value, rtol=1e-10, atol=1e-12), f"{name} differs from recomputed"
        assert not np.allclose(data[name], all_stats[name], rtol=1e-6, atol=0), f"{name} equals all-split value"
    print("5 ok: stored stats = recomputed from train runs, and differ from all-split stats")

    # 6
    train = split == "train"
    s = data["states"][train][:, :-1][valid[train]]
    z = (s - data["state_mean"]) / data["state_std"]
    assert np.allclose(z.mean(axis=0), 0, atol=1e-8), z.mean(axis=0)
    assert np.allclose(z.std(axis=0), 1, atol=1e-8), z.std(axis=0)
    print("6 ok: normalized valid training states have mean 0 and std 1 per feature")
    print("all tests passed")


if __name__ == "__main__":
    main()
