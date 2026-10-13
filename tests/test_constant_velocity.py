"""Tests for src/models/constant_velocity.py step.

(a) synthetic checks:
    - w = 0 leaves g unchanged
    - w parallel to g leaves g unchanged
    - for random unit g and random w with |w| < 5, g_next differs from the first order
      step g - DT * (w x g) by less than (|w| * DT)^2
    - |g_next| = |g| to 1e-12, and after 1000 steps with random w, |g| is still 1 to 1e-9
    - height and joint angle updates match the formulas on a hand built state
(b) real data (train runs, valid transitions of data/processed/2026-10-07_182522):
    prints RMS error, RMS true change and their ratio per feature and per group.
    The group RMS is over all transitions and all features of the group.
    Asserts that error = -(true change) for lin_vel, ang_vel, joint_vel (to 1e-12), and
    that the ratio is below 1 for height, gravity, joint_pos (a sign error would give ~2).

Nothing is written. Run: python tests/test_constant_velocity.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from features import STATE_NAMES, STATE_GROUPS  # noqa: E402
from models.constant_velocity import step, DT  # noqa: E402

DATASET = os.path.join(REPO_ROOT, "data", "processed", "2026-10-07_182522", "dataset.npz")
G, V, W = STATE_GROUPS["gravity"], STATE_GROUPS["lin_vel"], STATE_GROUPS["ang_vel"]


def random_unit(rng, n):
    x = rng.normal(size=(n, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def make_states(g, w):
    """States (N x 34) with the given g and w and every other feature random."""
    rng = np.random.default_rng(1)
    s = rng.normal(size=(len(g), 34))
    s[:, G] = g
    s[:, W] = w
    return s


def synthetic_checks():
    rng = np.random.default_rng(0)
    n = 1000
    no_actions = np.zeros((n, 12))
    g = random_unit(rng, n)

    # w = 0 leaves g unchanged
    s = make_states(g, np.zeros((n, 3)))
    err_zero = np.abs(step(s, no_actions)[:, G] - g).max()
    assert err_zero == 0.0, err_zero

    # w parallel to g leaves g unchanged
    w = g * rng.uniform(-5, 5, size=(n, 1))
    s = make_states(g, w)
    err_par = np.abs(step(s, no_actions)[:, G] - g).max()
    assert err_par < 1e-15, err_par

    # close to the first order step g - DT * (w x g)
    w = random_unit(rng, n) * rng.uniform(0, 5, size=(n, 1))
    s = make_states(g, w)
    g_next = step(s, no_actions)[:, G]
    first_order = g - DT * np.cross(w, g)
    diff = np.linalg.norm(g_next - first_order, axis=1)
    bound = (np.linalg.norm(w, axis=1) * DT) ** 2
    assert np.all(diff < bound), (diff / bound).max()

    # |g_next| = |g|, and |g| stays 1 over 1000 steps
    err_norm1 = np.abs(np.linalg.norm(g_next, axis=1) - np.linalg.norm(g, axis=1)).max()
    assert err_norm1 < 1e-12, err_norm1
    s = make_states(g, np.zeros((n, 3)))
    for _ in range(1000):
        s[:, W] = rng.uniform(-5, 5, size=(n, 3))
        s = step(s, no_actions)
    err_norm1000 = np.abs(np.linalg.norm(s[:, G], axis=1) - 1).max()
    assert err_norm1000 < 1e-9, err_norm1000
    print(f"(a) w = 0: max |g change| = {err_zero:.1e}; w parallel to g: {err_par:.1e}")
    print(f"    vs first order step: max diff / (|w| DT)^2 = {(diff / bound).max():.3f} (must be < 1)")
    print(f"    | |g_next| - |g| | max = {err_norm1:.1e}; after 1000 steps | |g| - 1 | max = {err_norm1000:.1e}")

    # height and joint angles on a hand built state
    s = np.zeros((2, 34))
    s[:, 0] = 0.75                                    # height
    s[0, G] = [0.0, 0.0, -1.0]                        # upright: world vertical velocity = v_z
    s[0, V] = [0.4, -0.1, 0.5]
    s[1, G] = [0.6, 0.0, -0.8]                        # tilted: -g . v = -(0.6 * 1.0) + 0.8 * 0.5 = -0.2
    s[1, V] = [1.0, 2.0, 0.5]
    s[:, STATE_GROUPS["joint_pos"]] = np.linspace(-1, 1, 12)
    s[:, STATE_GROUPS["joint_vel"]] = np.arange(12) - 5.5
    s_next = step(s, np.zeros((2, 12)))
    assert np.isclose(s_next[0, 0], 0.75 + 0.02 * 0.5, rtol=0, atol=1e-15)
    assert np.isclose(s_next[1, 0], 0.75 + 0.02 * (-0.2), rtol=0, atol=1e-15)
    q_expected = np.linspace(-1, 1, 12) + 0.02 * (np.arange(12) - 5.5)
    assert np.allclose(s_next[:, STATE_GROUPS["joint_pos"]], q_expected, rtol=0, atol=1e-15)
    print(f"    hand built state: height {s_next[0, 0]:.4f} (upright, v_z = 0.5) and "
          f"{s_next[1, 0]:.4f} (tilted, -g.v = -0.2), joint angles match q + DT * dq")


def real_data_check():
    data = dict(np.load(DATASET))
    train = data["split"] == "train"
    states = data["states"][train]                   # (n_runs, 1000, 34)
    valid = data["valid"][train]                     # (n_runs, 999)
    actions = data["actions"][train][valid]          # (n_valid, 12)
    s_k = states[:, :-1][valid]                      # (n_valid, 34)
    s_k1 = states[:, 1:][valid]
    error = step(s_k, actions) - s_k1
    true_change = s_k1 - s_k
    print(f"(b) {DATASET[len(REPO_ROOT) + 1:]}: {train.sum()} train runs, {len(s_k)} valid transitions")

    rms_err = np.sqrt(np.mean(error ** 2, axis=0))
    rms_true = np.sqrt(np.mean(true_change ** 2, axis=0))
    print(f"    {'feature':32s} {'RMS error':>12s} {'RMS true change':>16s} {'ratio':>8s}")
    for i, name in enumerate(STATE_NAMES):
        print(f"    {name:32s} {rms_err[i]:12.4e} {rms_true[i]:16.4e} {rms_err[i] / rms_true[i]:8.4f}")
    print(f"    {'group':32s}")
    ratio = {}
    for group, cols in STATE_GROUPS.items():
        e = np.sqrt(np.mean(error[:, cols] ** 2))
        t = np.sqrt(np.mean(true_change[:, cols] ** 2))
        ratio[group] = e / t
        print(f"    {group:32s} {e:12.4e} {t:16.4e} {ratio[group]:8.4f}")

    for group in ["lin_vel", "ang_vel", "joint_vel"]:
        cols = STATE_GROUPS[group]
        diff = np.abs(error[:, cols] + true_change[:, cols]).max()
        assert diff < 1e-12, (group, diff)
    for group in ["height", "gravity", "joint_pos"]:
        assert ratio[group] < 1.0, (group, ratio[group])


def main():
    synthetic_checks()
    real_data_check()
    print("PASS test_constant_velocity")


if __name__ == "__main__":
    main()
