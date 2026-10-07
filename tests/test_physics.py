"""Test 3: over a run with no fall, the time-averaged floor normal force on
both feet should be close to the robot's weight (sum of body masses * g).

Why: the pelvis starts and ends at about the same height, so the average
vertical acceleration of the robot is about zero, and the only vertical
forces are gravity and the floor (the floor is flat, so normal = vertical).

Run: python tests/test_physics.py
"""

import os
import sys

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
import collect_data  # noqa: E402

TOLERANCE = 0.02  # 2 %


def main():
    with open(os.path.join(REPO_ROOT, "configs", "collect_data.yaml")) as f:
        cfg = yaml.safe_load(f)
    sim = collect_data.load_sim(cfg)
    m = sim["m"]
    weight = m.body_mass.sum() * np.linalg.norm(m.opt.gravity)

    for run_index in range(5):  # use the first run that does not fall
        d = collect_data.run_one(cfg, sim, run_index)
        if not d["fell"]:
            break
    assert not d["fell"], "all tried runs fell"

    total = d["foot_force"].sum(axis=1)
    mean_force = total.mean()
    print(f"run {run_index} (seed {int(d['seed'])}), {len(total)} physics steps")
    print(f"robot mass {m.body_mass.sum():.4f} kg -> weight {weight:.2f} N")
    print(f"time-average of left + right foot normal force: {mean_force:.2f} N "
          f"({(mean_force / weight - 1) * 100:+.2f} %)")
    print(f"  average left {d['foot_force'][:, 0].mean():.1f} N, right {d['foot_force'][:, 1].mean():.1f} N, "
          f"peak total {total.max():.0f} N")
    assert abs(mean_force / weight - 1) < TOLERANCE
    print("PASS test_physics")


if __name__ == "__main__":
    main()
