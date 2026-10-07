"""Test 4: right after a push, the pelvis horizontal velocity jumps by the
logged velocity change.

The pelvis velocity is measured independently of the logger's push code: from
the qpos after every physics step. MuJoCo's Euler step updates the free-joint
position as pos_new = pos + dt * vel_new, so (pos[i] - pos[i-1]) / dt is the
pelvis velocity after physics step i. We compare the velocity one physics step
before the push with the velocity one physics step after it. In between, only
2 ms of gravity, contact, and joint forces act besides the push, so the
difference should equal dv up to a small tolerance.

Run: python tests/test_push.py
"""

import os
import sys

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
import collect_data  # noqa: E402

TOLERANCE = 0.02  # m/s


def main():
    with open(os.path.join(REPO_ROOT, "configs", "collect_data.yaml")) as f:
        cfg = yaml.safe_load(f)
    sim = collect_data.load_sim(cfg)
    g1 = sim["g1"]
    dt, dec = g1["simulation_dt"], g1["control_decimation"]
    n_rows = cfg["episode_policy_steps"]
    rows_per_cmd = round(cfg["cmd_period_s"] / (dt * dec))

    # The schedule of run 0, plus the strongest allowed push in a fixed direction at row 300.
    rng = np.random.default_rng(cfg["base_seed"])
    cmds, pushes = collect_data.sample_schedule(rng, cfg, n_rows, dt * dec)
    pushes[300] = cfg["push_max_speed"] * np.array([np.cos(2.0), np.sin(2.0), 0.0])

    trace = []
    d = collect_data.run_episode(sim, cmds, pushes, n_rows, rows_per_cmd, cfg["contact_force_threshold"],
                                 qpos_trace=trace)
    trace = np.array(trace)  # trace[i] = qpos after physics step i + 1
    assert not d["fell"]

    worst = 0.0
    for j, k in enumerate(d["push_row"]):
        s = d["step"][k]                           # physics steps done when row k (the push) happens
        v_before = (trace[s - 1, :2] - trace[s - 2, :2]) / dt   # velocity of the state at row k, before the push
        v_after = (trace[s, :2] - trace[s - 1, :2]) / dt        # velocity after the first step after the push
        jump = v_after - v_before
        dv = d["push_dv"][j, :2]
        err = np.linalg.norm(jump - dv)
        worst = max(worst, err)
        print(f"push at row {k:3d} (t={d['push_t'][j]:.2f} s): dv = {dv.round(4)}, "
              f"measured jump = {jump.round(4)}, error {err:.4f} m/s")
        # The logged pre-push velocity is the velocity measured from qpos.
        assert np.allclose(d["push_vel_before"][j, :2], v_before, rtol=0, atol=1e-6)
    assert worst < TOLERANCE, f"push error {worst:.4f} m/s exceeds {TOLERANCE}"
    print(f"worst error {worst:.4f} m/s (tolerance {TOLERANCE})")
    print("PASS test_push")


if __name__ == "__main__":
    main()
