"""Tests for src/features.py make_state.

(a) shape and names, joint order matches the MuJoCo model, rotation matrix matches MuJoCo
(b) invariance: shifting the robot in x, y and rotating it about the vertical
    axis (positions, orientation and world-frame velocity transformed together)
    leaves the state unchanged to ~1e-12; rotating about a horizontal axis does not
(c) the gravity direction equals the policy observation's gravity_orientation (obs[3:6])

Uses real states from a short simulation (2 s, with commands and pushes).
Run: python tests/test_features.py
"""

import os
import sys

import mujoco
import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import collect_data  # noqa: E402
from features import make_state, quat_to_rotmat, JOINT_NAMES, STATE_GROUPS  # noqa: E402


def axis_angle_quat(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def move_robot(qpos, qvel, quat_rot, offset):
    """Rotate the whole robot by quat_rot about the world origin, then shift it by offset."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat_rot)
    R = R.reshape(3, 3)
    qpos2, qvel2 = qpos.copy(), qvel.copy()
    qpos2[:, 0:3] = qpos[:, 0:3] @ R.T + offset
    for n in range(len(qpos)):
        mujoco.mju_mulQuat(qpos2[n, 3:7], quat_rot, qpos[n, 3:7])  # new orientation = rotation * old
    qvel2[:, 0:3] = qvel[:, 0:3] @ R.T   # world-frame linear velocity rotates with the robot
    # qvel[3:6] (body frame) and the joints do not change.
    return qpos2, qvel2


def main():
    with open(os.path.join(REPO_ROOT, "configs", "collect_data.yaml")) as f:
        cfg = yaml.safe_load(f)
    sim = collect_data.load_sim(cfg)
    m = sim["m"]
    rng = np.random.default_rng(0)
    cmds, pushes = collect_data.sample_schedule(rng, cfg, 1000, 0.02)
    pushes[60] = np.array([0.3, -0.2, 0.0])
    d = collect_data.run_episode(sim, cmds, pushes, 100, 250, cfg["contact_force_threshold"])
    qpos, qvel = d["qpos"], d["qvel"]

    # (a) shape, names, joint order, rotation matrix
    state, names = make_state(qpos, qvel)
    assert state.shape == (100, 34) and len(names) == 34 and len(set(names)) == 34
    assert [m.joint(j).name for j in range(1, m.njnt)] == [j + "_joint" for j in JOINT_NAMES]
    assert sum(s.stop - s.start for s in STATE_GROUPS.values()) == 34
    R_mj = np.zeros((len(qpos), 9))
    for n in range(len(qpos)):
        mujoco.mju_quat2Mat(R_mj[n], qpos[n, 3:7])
    err_R = np.abs(quat_to_rotmat(qpos[:, 3:7]) - R_mj.reshape(-1, 3, 3)).max()
    assert err_R < 1e-14
    print(f"(a) state {state.shape}, 34 unique names, joint order matches model, "
          f"rotation matrix vs mju_quat2Mat: {err_R:.1e}")
    print("    names:", names)

    # (b) invariance to x, y shift and yaw rotation
    worst = 0.0
    for angle, offset in [(0.7, [3.0, -2.0, 0.0]), (-2.9, [100.0, 50.0, 0.0]), (np.pi, [-7.0, 0.5, 0.0])]:
        q2, v2 = move_robot(qpos, qvel, axis_angle_quat([0, 0, 1], angle), np.array(offset))
        state2, _ = make_state(q2, v2)
        worst = max(worst, np.abs(state2 - state).max())
    assert worst < 1e-12, worst
    q2, v2 = move_robot(qpos, qvel, axis_angle_quat([1, 0, 0], 0.3), np.zeros(3))
    tilt_change = np.abs(make_state(q2, v2)[0] - state).max()
    assert tilt_change > 0.1  # the test can tell a real change apart
    print(f"(b) yaw + x/y shift: max |state change| = {worst:.1e}; "
          f"tilt about x by 0.3 rad (should change): {tilt_change:.3f}")

    # (c) gravity direction vs the policy observation
    gravity = state[:, STATE_GROUPS["gravity"]]
    deploy_formula = np.array([collect_data.get_gravity_orientation(q) for q in qpos[:, 3:7]])
    err_formula = np.abs(gravity - deploy_formula).max()
    err_obs = np.abs(gravity - d["obs"][:, 3:6]).max()
    quat_norm_err = np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1).max()
    assert err_formula < 1e-14 and err_obs < 1e-6
    print(f"(c) gravity vs deploy get_gravity_orientation (float64): {err_formula:.1e}; "
          f"vs saved obs[3:6] (float32): {err_obs:.1e}; max | |quat| - 1 | = {quat_norm_err:.1e}")
    print("PASS test_features")


if __name__ == "__main__":
    main()
