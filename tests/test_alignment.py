"""Test 2: the saved rows line up the way collect_data.py says they do.

Collects one run (seed = base_seed, with commands and pushes), saves it with
np.savez_compressed to a temporary folder, loads it back, and checks:
  - obs[k] can be rebuilt from saved qpos[k], qvel[k], cmd[k], phase[k], action[k-1]
  - replaying the saved obs through the (LSTM) policy from zero memory gives action[k]
  - target_q[k] = action[k] * action_scale + default_angles
  - 50 Hz and 500 Hz timestamps line up: row k is at physics step step[k] = 10(k+1),
    and 500 Hz row step[k] is computed from exactly row k's state
  - commands change only every 250 rows, pushes follow the schedule rules

Run: python tests/test_alignment.py
"""

import os
import sys
import tempfile

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
import collect_data  # noqa: E402


def rebuild_obs(g1, qpos, qvel, cmd, phase, prev_action):
    """Build the 47-dim observation from saved arrays (same layout as deploy_mujoco.py)."""
    n = g1["num_actions"]
    obs = np.zeros(g1["num_obs"], dtype=np.float32)
    obs[:3] = qvel[3:6] * g1["ang_vel_scale"]
    obs[3:6] = collect_data.get_gravity_orientation(qpos[3:7])
    obs[6:9] = cmd * g1["cmd_scale"]
    obs[9 : 9 + n] = (qpos[7:] - g1["default_angles"]) * g1["dof_pos_scale"]
    obs[9 + n : 9 + 2 * n] = qvel[6:] * g1["dof_vel_scale"]
    obs[9 + 2 * n : 9 + 3 * n] = prev_action
    obs[9 + 3 * n : 9 + 3 * n + 2] = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])
    return obs


def main():
    with open(os.path.join(REPO_ROOT, "configs", "collect_data.yaml")) as f:
        cfg = yaml.safe_load(f)
    sim = collect_data.load_sim(cfg)
    g1, m = sim["g1"], sim["m"]
    dt, dec = g1["simulation_dt"], g1["control_decimation"]

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "run_000.npz")
        np.savez_compressed(path, **collect_data.run_one(cfg, sim, 0))
        d = dict(np.load(path))
    n_rows, n_fine = len(d["t"]), len(d["t_fine"])
    print(f"run: {n_rows} rows, {n_fine} fine rows, fell={bool(d['fell'])}, pushes at rows {d['push_row'].tolist()}")
    assert n_rows == cfg["episode_policy_steps"] and not d["fell"], "pick a run that does not fall"

    # --- observation rebuild ---
    prev_action = np.zeros(g1["num_actions"], dtype=np.float32)
    for k in range(n_rows):
        obs = rebuild_obs(g1, d["qpos"][k], d["qvel"][k], d["cmd"][k], d["phase"][k], prev_action)
        assert np.array_equal(obs, d["obs"][k]), f"obs mismatch at row {k}"
        prev_action = d["action"][k]
    print(f"obs rebuilt exactly for all {n_rows} rows")

    # --- policy replay (the policy is an LSTM, so replay from zero memory in order) ---
    # Call it exactly like deploy_mujoco.py does (grad mode on, then .detach()).
    # Under torch.no_grad() the TorchScript LSTM gives results that differ by ~1e-6.
    policy = sim["policy"]
    policy.reset_memory()
    for k in range(n_rows):
        action = policy(torch.from_numpy(d["obs"][k]).unsqueeze(0)).detach().numpy().squeeze()
        assert np.array_equal(action, d["action"][k]), f"action mismatch at row {k}"
    # For contrast: one call on obs[500] with zero memory does NOT give action[500].
    policy.reset_memory()
    single = policy(torch.from_numpy(d["obs"][500]).unsqueeze(0)).detach().numpy().squeeze()
    print(f"policy replay reproduces all {n_rows} actions exactly "
          f"(single call without memory would be off by up to {np.abs(single - d['action'][500]).max():.3f})")

    assert np.array_equal(d["target_q"], d["action"] * g1["action_scale"] + g1["default_angles"])

    # --- phase and timestamps ---
    k = np.arange(n_rows)
    assert np.array_equal(d["step"], dec * (k + 1))
    assert np.array_equal(d["phase"], np.array([s * dt % 0.8 / 0.8 for s in d["step"]]))
    assert n_fine == d["step"][-1]  # 10000 physics steps; the last row's action is never executed
    assert np.allclose(d["t_fine"], np.arange(n_fine) * dt, rtol=0, atol=1e-9)
    assert np.allclose(d["t"], d["step"] * dt, rtol=0, atol=1e-9)
    has_fine = d["step"] < n_fine
    assert np.array_equal(d["t"][has_fine], d["t_fine"][d["step"][has_fine]])
    print(f"timestamps: t[k] == t_fine[step[k]] exactly; max |t_fine - i*dt| = "
          f"{np.abs(d['t_fine'] - np.arange(n_fine) * dt).max():.1e} s (float accumulation in d.time)")

    # 500 Hz row step[k] is computed from row k's state: its PD torque must match exactly.
    for k in np.flatnonzero(has_fine):
        tau = collect_data.pd_control(d["target_q"][k], d["qpos"][k, 7:], g1["kps"], np.zeros_like(g1["kds"]),
                                      d["qvel"][k, 6:], g1["kds"])
        assert np.array_equal(tau, d["tau_cmd"][d["step"][k]]), f"tau_cmd misaligned at row {k}"
    print("tau_cmd[step[k]] equals the PD torque from row k's state for every row")

    # Applied torque = PD torque clipped to the joint's actuatorfrcrange (gear is 1).
    limit = m.jnt_actfrcrange[1:, 1]
    assert np.array_equal(d["tau_applied"], np.clip(d["tau_cmd"], -limit, limit))
    clipped = np.abs(d["tau_cmd"]) > limit
    print(f"tau_applied == clip(tau_cmd); clamp active in {clipped.any(axis=1).mean() * 100:.2f}% of physics steps")

    assert np.array_equal(d["foot_contact"], d["foot_force"] > cfg["contact_force_threshold"])

    # --- commands: constant inside each 250-row block ---
    rows_per_cmd = round(cfg["cmd_period_s"] / (dt * dec))
    changes = np.flatnonzero(np.any(d["cmd"][1:] != d["cmd"][:-1], axis=1)) + 1
    assert set(changes) <= set(range(0, n_rows, rows_per_cmd))
    print(f"command changes at rows {changes.tolist()} (t = {d['t'][changes].round(2).tolist()} s)")

    # --- pushes: one per 250-row window, never before t = 1.0 s ---
    assert np.array_equal(d["push_row"] // rows_per_cmd, np.arange(len(d["push_row"])))
    assert np.all(d["t"][d["push_row"]] >= cfg["push_min_time_s"] - 1e-9)
    assert np.array_equal(d["push_t"], d["t"][d["push_row"]])
    assert np.array_equal(d["qvel"][d["push_row"], 0:3], d["push_vel_before"] + d["push_dv"])
    print(f"push times {d['push_t'].round(2).tolist()} s, |dv| = "
          f"{np.linalg.norm(d['push_dv'], axis=1).round(3).tolist()} m/s")
    print("PASS test_alignment")


if __name__ == "__main__":
    main()
