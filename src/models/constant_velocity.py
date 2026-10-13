"""Constant velocity baseline: a world model with no learning.

It assumes every velocity stays the same for one step and moves the
positions with those velocities. State layout is the one in src/features.py
(raw, not normalized).

  v, w, dq   unchanged
  q          q + DT * dq
  height     height - DT * dot(g, v)    (world vertical velocity is -g . v, since R^T e_z = -g)
  g          g rotated about k = w / |w| by angle -|w| * DT (Rodrigues formula); this is the
             exact solution of dg/dt = -w x g for constant w
"""

import numpy as np

from features import STATE_GROUPS

DT = 0.02  # sim dt 0.002 s x control decimation 10 (50 Hz policy)


def step(states, actions):
    """Next states (N x 34) from states (N x 34). actions (N x 12) is ignored.

    actions is in the signature so the same rollout code can call any model as
    step(states, actions).
    """
    height = states[:, STATE_GROUPS["height"]]   # (N, 1)
    g = states[:, STATE_GROUPS["gravity"]]       # (N, 3)
    v = states[:, STATE_GROUPS["lin_vel"]]       # (N, 3)
    w = states[:, STATE_GROUPS["ang_vel"]]       # (N, 3)
    q = states[:, STATE_GROUPS["joint_pos"]]     # (N, 12)
    dq = states[:, STATE_GROUPS["joint_vel"]]    # (N, 12)

    # Rotate g about k = w / |w| by angle a = -|w| * DT.
    w_norm = np.linalg.norm(w, axis=1, keepdims=True)    # (N, 1)
    moving = w_norm[:, 0] >= 1e-12
    k = np.zeros_like(w)
    k[moving] = w[moving] / w_norm[moving]                # k = 0 where |w| < 1e-12
    a = -w_norm * DT
    k_dot_g = np.sum(k * g, axis=1, keepdims=True)
    g_next = g * np.cos(a) + np.cross(k, g) * np.sin(a) + k * k_dot_g * (1 - np.cos(a))
    g_next[~moving] = g[~moving]                          # keep g unchanged where |w| < 1e-12

    next_states = states.copy()
    next_states[:, STATE_GROUPS["height"]] = height - DT * np.sum(g * v, axis=1, keepdims=True)
    next_states[:, STATE_GROUPS["gravity"]] = g_next
    next_states[:, STATE_GROUPS["joint_pos"]] = q + DT * dq
    # v, w, dq are already copied unchanged.
    return next_states
