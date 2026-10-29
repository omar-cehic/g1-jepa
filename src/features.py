"""Turn raw 50 Hz simulator arrays into the 34-dim model state.

State layout (all in the pelvis frame except height):
   0      pelvis height                    qpos[2]
   1:4    gravity direction                R^T [0, 0, -1]
   4:7    pelvis linear velocity           R^T qvel[0:3]   (qvel[0:3] is in the world frame)
   7:10   pelvis angular velocity          qvel[3:6]       (MuJoCo gives it in the pelvis frame)
  10:22   joint angles                     qpos[7:19]
  22:34   joint velocities                 qvel[6:18]
R is the pelvis orientation (pelvis frame -> world frame) from the quaternion
qpos[3:7] = (w, x, y, z).

The state does not depend on where the robot is in x, y or which way it
faces (yaw): those are not part of the physics of walking on a flat floor.
"""

import numpy as np

# Hinge joints in qpos[7:19] / qvel[6:18] order (checked against the MuJoCo
# model in tests/test_features.py).
JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
]

STATE_NAMES = (
    ["pelvis_height"]
    + ["gravity_x", "gravity_y", "gravity_z"]
    + ["lin_vel_x", "lin_vel_y", "lin_vel_z"]
    + ["ang_vel_x", "ang_vel_y", "ang_vel_z"]
    + ["q_" + j for j in JOINT_NAMES]
    + ["dq_" + j for j in JOINT_NAMES]
)

# Which state dimensions belong to which group.
STATE_GROUPS = {
    "height": slice(0, 1),
    "gravity": slice(1, 4),
    "lin_vel": slice(4, 7),
    "ang_vel": slice(7, 10),
    "joint_pos": slice(10, 22),
    "joint_vel": slice(22, 34),
}


def quat_to_rotmat(quat):
    """Rotation matrices (N x 3 x 3) from unit quaternions (N x 4, order w, x, y, z)."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    R = np.empty((len(quat), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def make_state(qpos, qvel):
    """Model state (N x 34) from raw qpos (N x 19) and qvel (N x 18). Returns (state, STATE_NAMES)."""
    R = quat_to_rotmat(qpos[:, 3:7])
    # R^T v for every row: entry i is sum_j R[j, i] * v[j].
    gravity = np.einsum("nji,j->ni", R, np.array([0.0, 0.0, -1.0]))
    lin_vel = np.einsum("nji,nj->ni", R, qvel[:, 0:3])
    state = np.concatenate([
        qpos[:, 2:3],    # pelvis height
        gravity,         # gravity direction in the pelvis frame
        lin_vel,         # pelvis linear velocity in the pelvis frame
        qvel[:, 3:6],    # pelvis angular velocity (already in the pelvis frame)
        qpos[:, 7:19],   # joint angles
        qvel[:, 6:18],   # joint velocities
    ], axis=1)
    return state, list(STATE_NAMES)


# Raw MuJoCo state: qpos (19) then qvel (18), nothing converted.
# The free joint's qvel[0:3] is in the world frame, qvel[3:6] in the pelvis frame.
FULL_STATE_NAMES = (
    ["qpos_pelvis_x", "qpos_pelvis_y", "qpos_pelvis_z"]
    + ["qpos_quat_w", "qpos_quat_x", "qpos_quat_y", "qpos_quat_z"]
    + ["qpos_" + j for j in JOINT_NAMES]
    + ["qvel_lin_vel_x_world", "qvel_lin_vel_y_world", "qvel_lin_vel_z_world"]
    + ["qvel_ang_vel_x_pelvis", "qvel_ang_vel_y_pelvis", "qvel_ang_vel_z_pelvis"]
    + ["qvel_" + j for j in JOINT_NAMES]
)

# Which full state dimensions belong to which group.
FULL_STATE_GROUPS = {
    "pos_xy": slice(0, 2),
    "height": slice(2, 3),
    "quat": slice(3, 7),
    "joint_pos": slice(7, 19),
    "lin_vel_world": slice(19, 22),
    "ang_vel": slice(22, 25),
    "joint_vel": slice(25, 37),
}


def make_full_state(qpos, qvel):
    """Full state (N x 37): raw qpos (N x 19) and qvel (N x 18) side by side. Returns (state, FULL_STATE_NAMES)."""
    return np.concatenate([qpos, qvel], axis=1), list(FULL_STATE_NAMES)


# Full state without pelvis x and y: qpos[2:19] (17) then qvel (18), nothing converted.
FULL_NO_XY_STATE_NAMES = FULL_STATE_NAMES[2:]

# Which full_no_xy state dimensions belong to which group.
FULL_NO_XY_STATE_GROUPS = {
    "height": slice(0, 1),
    "quat": slice(1, 5),
    "joint_pos": slice(5, 17),
    "lin_vel_world": slice(17, 20),
    "ang_vel": slice(20, 23),
    "joint_vel": slice(23, 35),
}


def make_full_no_xy_state(qpos, qvel):
    """Full state without pelvis x and y (N x 35): qpos[:, 2:] (N x 17) and qvel (N x 18). Returns (state, FULL_NO_XY_STATE_NAMES)."""
    return np.concatenate([qpos[:, 2:], qvel], axis=1), list(FULL_NO_XY_STATE_NAMES)
