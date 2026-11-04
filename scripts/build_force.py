"""Build the foot contact force file for a dataset: the mean force per foot over every transition.

What foot_force is (scripts/collect_data.py, floor_contacts): for each foot, the sum over its 4 contact
spheres of the NORMAL force of every sphere-floor contact (mj_contactForce, f6[0] in the contact frame).
The floor is flat, so the normal is vertical: this is the vertical ground reaction force per foot, in N.
Friction (tangential) forces are not included.

Index convention (as in src/contact.py and scripts/explore_contact.py): 500 Hz row i describes physics
step i, which starts from the state at t_fine[i] = i * dt; the integrator is Euler, so the contact force of
row i is computed from that start state and acts during physics step i -> i + 1. 50 Hz row k is the state at
500 Hz index step[k] = 10 (k + 1). Transition k (row k -> row k + 1, action a_k held) is made of physics
steps step[k] .. step[k+1] - 1, so

  force[run, k, foot] = mean of raw foot_force[step[k] : step[k+1], foot]   (10 samples, end excluded)

The end sample step[k+1] is left out: its force acts during the next transition. contact_steps() in
src/contact.py uses the closed range step[k] .. step[k+1] (plus the 40 ms window) to match landing indices.

Physics check: for each train run, the time mean of left + right force over the valid transitions, divided
by the robot's weight m g (m = sum of body masses of the MuJoCo model, g = |gravity|). With no net vertical
acceleration of the center of mass over the run it should be close to 1.

Usage: python scripts/build_force.py data/processed/<dataset folder>
Output: data/processed/<YYYY-MM-DD_HHMMSS>_foot_force/ with foot_force.npz (foot_force (runs, 999, 2) in N,
        left then right, rows in the dataset's run order; run_ids) and meta.json.
"""

import argparse
import datetime
import json
import os
import sys

import mujoco
import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from build_dataset import git_commit  # noqa: E402
from collect_data import read_g1_config  # noqa: E402

DECIMATION = 10  # physics steps per 50 Hz row


def transition_force(raw):
    """(n_rows - 1, 2) mean foot force over each transition k: raw foot_force[step[k] : step[k+1]]."""
    step, foot_force = raw["step"], raw["foot_force"]
    assert np.array_equal(step, DECIMATION * (np.arange(len(step)) + 1)), "step is not 10 (k + 1)"
    assert len(foot_force) == step[-1], f"{len(foot_force)} 500 Hz rows, expected {step[-1]}"
    return np.array([foot_force[step[k]:step[k + 1]].mean(axis=0) for k in range(len(step) - 1)])


def robot_weight(raw_dir):
    """Total mass (kg) and |g| of the MuJoCo model the raw batch was collected with."""
    with open(os.path.join(raw_dir, "collect_data.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    unitree_dir = os.path.abspath(os.path.join(REPO_ROOT, cfg["unitree_rl_gym_dir"]))
    m = mujoco.MjModel.from_xml_path(read_g1_config(unitree_dir, cfg["g1_config"])["xml_path"])
    return float(m.body_mass.sum()), float(np.linalg.norm(m.opt.gravity))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="data/processed/<dataset folder>")
    args = parser.parse_args()
    dataset_dir = os.path.normpath(args.dataset_dir)
    with open(os.path.join(REPO_ROOT, dataset_dir, "meta.json"), "r") as f:
        raw_dir_rel = json.load(f)["source_raw_dir"]
    raw_dir = os.path.join(REPO_ROOT, raw_dir_rel)
    data = np.load(os.path.join(REPO_ROOT, dataset_dir, "dataset.npz"))
    run_ids, split, valid = data["run_ids"], data["split"], data["valid"]

    force = np.stack([transition_force(np.load(os.path.join(raw_dir, f"run_{int(r):03d}.npz"))) for r in run_ids])
    assert force.shape == valid.shape + (2,), (force.shape, valid.shape)

    mass, g = robot_weight(raw_dir)
    with open(os.path.join(raw_dir, "meta.json"), "r") as f:
        raw_meta = json.load(f)
    assert abs(mass - raw_meta["robot_mass_kg"]) < 1e-9, (mass, raw_meta["robot_mass_kg"])
    train = np.flatnonzero(split == "train")
    ratio = np.array([force[r].sum(axis=1)[valid[r]].mean() / (mass * g) for r in train])
    print(f"mass {mass:.4f} kg (MuJoCo model), g {g:.4f} m/s^2, m g {mass * g:.2f} N")
    print(f"physics check over {len(train)} train runs: mean(left + right over valid transitions) / (m g): "
          f"median {np.median(ratio):.4f}, min {ratio.min():.4f}, max {ratio.max():.4f}")

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "data", "processed", f"{stamp}_foot_force")
    os.makedirs(out_dir)  # raises if the folder already exists: never overwrite
    np.savez(os.path.join(out_dir, "foot_force.npz"), foot_force=force, run_ids=run_ids)
    commit, dirty = git_commit()
    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_dataset": os.path.join(dataset_dir, "dataset.npz"),
        "source_raw_dir": raw_dir_rel,
        "git_commit": commit,
        "git_uncommitted_changes": dirty,
        "shape": list(force.shape),
        "feet": ["left", "right"],
        "units": "N",
        "quantity": "sum over the foot's 4 contact spheres of the floor contact normal force (vertical on the "
                    "flat floor); friction not included",
        "definition": "foot_force[run, k] = mean of raw foot_force[step[k] : step[k+1]] (500 Hz indices "
                      "10 (k + 1) .. 10 (k + 2) - 1): the 10 physics steps of transition k -> k + 1",
        "physics_check": {
            "what": "per train run: mean of left + right over valid transitions / (m g)",
            "mass_kg": mass, "g": g, "n_runs": int(len(train)),
            "median": float(np.median(ratio)), "min": float(ratio.min()), "max": float(ratio.max()),
        },
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
