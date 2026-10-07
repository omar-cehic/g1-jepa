"""Test 1: the logger's simulation is exactly the deploy_mujoco.py loop.

With a fixed command [0.5, 0, 0] and no pushes, run 2 s with:
  (a) a plain headless transcription of deploy_mujoco.py (below), and
  (b) collect_data.run_episode,
and check qpos is identical (np.array_equal) after every physics step.

Run: python tests/test_equivalence.py
"""

import os
import sys

import mujoco
import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
import collect_data  # noqa: E402

DURATION_S = 2.0


# Own copies of the two helpers from deploy_mujoco.py (not imported from the
# logger), so a change in the logger's versions would make this test fail.
def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def deploy_transcription(unitree_dir, n_physics_steps):
    """deploy_mujoco.py with the viewer and sleep removed. Returns qpos after every mj_step."""
    LEGGED_GYM_ROOT_DIR = unitree_dir
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/g1.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]

        cmd = np.array(config["cmd_init"], dtype=np.float32)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    policy = torch.jit.load(policy_path)  # freshly loaded, so its LSTM memory is zero

    qpos_trace = []
    for _ in range(n_physics_steps):
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        qpos_trace.append(d.qpos.copy())

        counter += 1
        if counter % control_decimation == 0:
            qj = d.qpos[7:]
            dqj = d.qvel[6:]
            quat = d.qpos[3:7]
            omega = d.qvel[3:6]

            qj = (qj - default_angles) * dof_pos_scale
            dqj = dqj * dof_vel_scale
            gravity_orientation = get_gravity_orientation(quat)
            omega = omega * ang_vel_scale

            period = 0.8
            count = counter * simulation_dt
            phase = count % period / period
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)

            obs[:3] = omega
            obs[3:6] = gravity_orientation
            obs[6:9] = cmd * cmd_scale
            obs[9 : 9 + num_actions] = qj
            obs[9 + num_actions : 9 + 2 * num_actions] = dqj
            obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
            obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action = policy(obs_tensor).detach().numpy().squeeze()
            target_dof_pos = action * action_scale + default_angles
    return np.array(qpos_trace)


def main():
    with open(os.path.join(REPO_ROOT, "configs", "collect_data.yaml")) as f:
        cfg = yaml.safe_load(f)
    sim = collect_data.load_sim(cfg)
    g1 = sim["g1"]
    policy_dt = g1["simulation_dt"] * g1["control_decimation"]
    n_rows = round(DURATION_S / policy_dt)  # 100
    n_physics_steps = n_rows * g1["control_decimation"]  # 1000

    reference = deploy_transcription(sim["unitree_dir"], n_physics_steps)

    # Run a different episode first, so the test also catches LSTM memory leaking between runs.
    rng = np.random.default_rng(123)
    cmds, pushes = collect_data.sample_schedule(rng, cfg, n_rows, policy_dt)
    collect_data.run_episode(sim, cmds, pushes, n_rows, round(cfg["cmd_period_s"] / policy_dt),
                             cfg["contact_force_threshold"])

    cmds = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)
    assert np.array_equal(cmds[0], g1["cmd_init"])
    logger_trace = []
    data = collect_data.run_episode(sim, cmds, {}, n_rows, n_rows, cfg["contact_force_threshold"],
                                    qpos_trace=logger_trace)
    logger_trace = np.array(logger_trace)

    print(f"physics steps compared: {len(reference)} (transcription) vs {len(logger_trace)} (logger)")
    print(f"max |qpos difference|: {np.abs(reference - logger_trace).max():.3e}")
    print(f"pelvis moved from x=0 to x={reference[-1, 0]:.3f} m (robot is walking)")
    assert reference.shape == logger_trace.shape == (n_physics_steps, 19)
    assert np.array_equal(reference, logger_trace), "logger qpos differs from deploy transcription"
    # The saved 50 Hz rows are the states after every 10th physics step.
    assert np.array_equal(data["qpos"], reference[9::10])
    assert not data["fell"]
    print("PASS test_equivalence")


if __name__ == "__main__":
    main()
