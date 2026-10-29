"""Tests for a full state dataset built by scripts/build_dataset.py (state_type: full or full_no_xy).

1. meta.json says state_type "full" (or "full_no_xy"), and states are (299, 1000, 37)
   (or (299, 1000, 35)) float64.
2. valid, split, run_ids, contact_step and actions are identical to the official
   dataset (data/processed/2026-10-07_182522), so only the state differs.
3. For 5 random runs, states equal raw qpos (qpos[:, 2:] for full_no_xy) and qvel
   concatenated, exactly.
4. The stored state stats equal the mean and std over valid training transitions
   (recomputed here), and differ from the stats over all splits.

Run: python tests/test_full_state_dataset.py data/processed/<full state folder>
"""

import json
import os
import sys

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL = os.path.join(REPO_ROOT, "data", "processed", "2026-10-07_182522")


def main():
    folder = sys.argv[1]
    with open(os.path.join(folder, "build_dataset.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(folder, "meta.json")) as f:
        meta = json.load(f)
    data = dict(np.load(os.path.join(folder, "dataset.npz")))  # read each array once
    official = dict(np.load(os.path.join(OFFICIAL, "dataset.npz")))
    print(f"checking {folder} against {OFFICIAL}")

    # 1
    kind = meta["state_type"]
    assert kind in ["full", "full_no_xy"] and cfg["state_type"] == kind
    first_qpos = 0 if kind == "full" else 2  # full_no_xy drops pelvis x and y
    dim = 37 - first_qpos
    assert len(meta["state_names"]) == dim
    assert data["states"].shape == (299, 1000, dim) and data["states"].dtype == np.float64
    print(f"1 ok: state_type {kind}, states (299, 1000, {dim}) float64")

    # 2
    for key in ["valid", "split", "run_ids", "contact_step", "actions"]:
        assert data[key].dtype == official[key].dtype, f"{key}: dtype differs"
        assert np.array_equal(data[key], official[key]), f"{key} differs from the official dataset"
    print("2 ok: valid, split, run_ids, contact_step, actions identical to the official dataset")

    # 3
    rng = np.random.default_rng(0)
    for r in rng.choice(len(data["run_ids"]), size=5, replace=False):
        run = data["run_ids"][r]
        d = np.load(os.path.join(REPO_ROOT, cfg["raw_dir"], f"run_{run:03d}.npz"))
        n_qpos = 19 - first_qpos
        assert np.array_equal(data["states"][r, :, :n_qpos], d["qpos"][:, first_qpos:]), f"run {run}: qpos differs"
        assert np.array_equal(data["states"][r, :, n_qpos:], d["qvel"]), f"run {run}: qvel differs"
    print(f"3 ok: states = [qpos[:, {first_qpos}:], qvel] exactly for 5 random runs")

    # 4
    def state_stats(run_mask):
        s = data["states"][run_mask][:, :-1][data["valid"][run_mask]]
        return s.mean(axis=0), s.std(axis=0)

    mean, std = state_stats(data["split"] == "train")
    assert np.allclose(data["state_mean"], mean, rtol=1e-10, atol=1e-12)
    assert np.allclose(data["state_std"], std, rtol=1e-10, atol=1e-12)
    all_mean, _ = state_stats(np.ones(len(data["run_ids"]), dtype=bool))
    assert not np.allclose(data["state_mean"], all_mean, rtol=1e-6, atol=0)
    print("4 ok: state stats = recomputed from valid training transitions, differ from all-split stats")
    print("all tests passed")


if __name__ == "__main__":
    main()
