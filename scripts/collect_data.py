"""Run the pretrained G1 walking policy headless and log raw simulator data.

The simulation and policy loop is a copy of
unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco.py (commit 276801e) with the
viewer and real-time sleep removed. The only additions are: commands that
change over time, small pushes, and recording.

Usage:
    python scripts/collect_data.py configs/collect_data.yaml

Output: data/raw/<YYYY-MM-DD_HHMMSS>/ with run_000.npz, run_001.npz, ...,
a copy of the config, and meta.json.

Timing of what is saved (dt = 0.002 s, policy every 10 physics steps):
- 50 Hz row k is recorded after physics step 10*(k+1), so at t = 0.02*(k+1).
  It holds the state the policy reads (after any push at this row), the
  observation built from it, and the action computed from that observation.
  That action is held for the next 10 physics steps.
- 500 Hz row i describes physics step i, which starts from the state at
  t_fine[i] = i * dt. Contact forces and qfrc_actuator read right after
  mj_step come from the forward pass at the START of the step, so they are
  stamped with that start time. So 500 Hz row step[k] is computed from
  exactly the state of 50 Hz row k.
"""

import argparse
import datetime
import json
import math
import os
import shutil
import subprocess
import time

import mujoco
import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- copied verbatim from deploy_mujoco.py ----
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
# ---- end of copied code ----


def read_g1_config(unitree_dir, g1_config):
    """Read g1.yaml with the same dtypes deploy_mujoco.py uses."""
    with open(os.path.join(unitree_dir, g1_config), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return {
        "policy_path": config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", unitree_dir),
        "xml_path": config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", unitree_dir),
        "simulation_dt": config["simulation_dt"],
        "control_decimation": config["control_decimation"],
        "kps": np.array(config["kps"], dtype=np.float32),
        "kds": np.array(config["kds"], dtype=np.float32),
        "default_angles": np.array(config["default_angles"], dtype=np.float32),
        "ang_vel_scale": config["ang_vel_scale"],
        "dof_pos_scale": config["dof_pos_scale"],
        "dof_vel_scale": config["dof_vel_scale"],
        "action_scale": config["action_scale"],
        "cmd_scale": np.array(config["cmd_scale"], dtype=np.float32),
        "num_actions": config["num_actions"],
        "num_obs": config["num_obs"],
        "cmd_init": np.array(config["cmd_init"], dtype=np.float32),
    }


def find_foot_geoms(m, foot_bodies):
    """For each foot body, return the ids of its colliding sphere geoms."""
    foot_geoms = []
    for name in foot_bodies:
        body_id = m.body(name).id
        geoms = [g for g in range(m.ngeom)
                 if m.geom_bodyid[g] == body_id
                 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                 and (m.geom_contype[g] or m.geom_conaffinity[g])]
        assert len(geoms) == 4, f"expected 4 contact spheres on {name}, found {geoms}"
        foot_geoms.append(geoms)
    return foot_geoms


def load_sim(cfg):
    """Load g1.yaml settings, the MuJoCo model, the foot geoms, and the policy."""
    unitree_dir = os.path.abspath(os.path.join(REPO_ROOT, cfg["unitree_rl_gym_dir"]))
    g1 = read_g1_config(unitree_dir, cfg["g1_config"])
    m = mujoco.MjModel.from_xml_path(g1["xml_path"])
    m.opt.timestep = g1["simulation_dt"]
    floor_id = m.geom("floor").id
    foot_geoms = find_foot_geoms(m, cfg["foot_bodies"])
    policy = torch.jit.load(g1["policy_path"])
    return {"unitree_dir": unitree_dir, "g1": g1, "m": m, "floor_id": floor_id,
            "foot_geoms": foot_geoms, "policy": policy}


def sample_schedule(rng, cfg, n_rows, policy_dt):
    """Draw the commands and pushes for one run.

    Returns
      cmds: (n_windows, 3) float32. Row k uses cmds[k // rows_per_cmd].
      pushes: dict {row k: dv (3,)}. dv is added to the pelvis velocity just
              before row k is recorded.
    Row k is at time t = (k + 1) * policy_dt.
    """
    rows_per_cmd = round(cfg["cmd_period_s"] / policy_dt)
    n_cmds = math.ceil(n_rows / rows_per_cmd)
    r = cfg["cmd_ranges"]
    low = [r["lin_vel_x"][0], r["lin_vel_y"][0], r["ang_vel_yaw"][0]]
    high = [r["lin_vel_x"][1], r["lin_vel_y"][1], r["ang_vel_yaw"][1]]
    cmds = rng.uniform(low, high, size=(n_cmds, 3)).astype(np.float32)

    rows_per_push = round(cfg["push_period_s"] / policy_dt)
    first_push_row = round(cfg["push_min_time_s"] / policy_dt) - 1  # first row with t >= push_min_time_s
    pushes = {}
    for start in range(0, n_rows, rows_per_push):
        allowed_rows = np.arange(max(start, first_push_row), min(start + rows_per_push, n_rows))
        k = int(rng.choice(allowed_rows))
        angle = rng.uniform(0.0, 2 * np.pi)
        speed = rng.uniform(0.0, cfg["push_max_speed"])
        # qvel[0:3] of a free joint is in the world frame, so dv is too.
        pushes[k] = np.array([speed * np.cos(angle), speed * np.sin(angle), 0.0])
    return cmds, pushes


def floor_contacts(m, d, floor_id, foot_geoms):
    """Sum the floor normal force on each foot's spheres.

    Call right after mj_step: d.contact and the contact forces then describe
    the state at the start of that step (before integration).
    Returns (force [left, right], fall_geom). fall_geom is a non-foot geom
    touching the floor, or -1 if there is none.
    """
    force = np.zeros(2)
    fall_geom = -1
    f6 = np.zeros(6)
    for i in range(d.ncon):
        a, b = d.contact.geom[i]
        if a == floor_id:
            other = b
        elif b == floor_id:
            other = a
        else:
            continue  # not a floor contact
        if other in foot_geoms[0]:
            mujoco.mj_contactForce(m, d, i, f6)  # f6[0] is the normal force
            force[0] += f6[0]
        elif other in foot_geoms[1]:
            mujoco.mj_contactForce(m, d, i, f6)
            force[1] += f6[0]
        else:
            fall_geom = other
    return force, fall_geom


def run_episode(sim, cmds, pushes, n_rows, rows_per_cmd, force_threshold, qpos_trace=None):
    """Run one episode and return all logged arrays in a dict.

    sim is the dict from load_sim. cmds and pushes are as returned by sample_schedule.

    If qpos_trace is a list, qpos after every physics step is appended to it
    (used only by the equivalence test).
    """
    m, policy, g1 = sim["m"], sim["policy"], sim["g1"]
    floor_id, foot_geoms = sim["floor_id"], sim["foot_geoms"]
    # Same names as deploy_mujoco.py, so the loop below can be compared line by line.
    simulation_dt = g1["simulation_dt"]
    control_decimation = g1["control_decimation"]
    kps, kds = g1["kps"], g1["kds"]
    default_angles = g1["default_angles"]
    num_actions = g1["num_actions"]

    # Fresh state, as deploy_mujoco.py does: default MuJoCo state, zero action.
    d = mujoco.MjData(m)
    policy.reset_memory()  # the policy is an LSTM: clear its memory from earlier runs
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(g1["num_obs"], dtype=np.float32)
    counter = 0

    rows = {key: [] for key in ["t", "step", "qpos", "qvel", "cmd", "phase", "obs", "action", "target_q"]}
    fine = {key: [] for key in ["t_fine", "foot_force", "foot_contact", "tau_cmd", "tau_applied"]}
    events = {key: [] for key in ["push_row", "push_t", "push_dv", "push_vel_before"]}
    fell, fall_time, fall_geom = False, np.nan, -1

    while counter < n_rows * control_decimation:
        t_start = d.time  # time of the state this physics step starts from
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        counter += 1
        if qpos_trace is not None:
            qpos_trace.append(d.qpos.copy())

        # 500 Hz row: all values describe the step that started at t_start.
        force, other_geom = floor_contacts(m, d, floor_id, foot_geoms)
        fine["t_fine"].append(t_start)
        fine["foot_force"].append(force)
        fine["foot_contact"].append(force > force_threshold)
        fine["tau_cmd"].append(tau.copy())
        fine["tau_applied"].append(d.qfrc_actuator[6:].copy())  # after the actuatorfrcrange clamp
        if other_geom >= 0:
            fell, fall_time, fall_geom = True, t_start, other_geom
            break

        if counter % control_decimation == 0:
            k = counter // control_decimation - 1  # 50 Hz row index
            cmd = cmds[k // rows_per_cmd]

            # Push first, so row k already shows the pushed velocity and the
            # transition k -> k+1 is plain physics. Transition k-1 -> k contains the push.
            if k in pushes:
                events["push_row"].append(k)
                events["push_t"].append(d.time)
                events["push_dv"].append(pushes[k])
                events["push_vel_before"].append(d.qvel[0:3].copy())
                d.qvel[0:3] += pushes[k]

            # Observation, exactly as in deploy_mujoco.py.
            qj = d.qpos[7:]
            dqj = d.qvel[6:]
            quat = d.qpos[3:7]
            omega = d.qvel[3:6]

            qj = (qj - default_angles) * g1["dof_pos_scale"]
            dqj = dqj * g1["dof_vel_scale"]
            gravity_orientation = get_gravity_orientation(quat)
            omega = omega * g1["ang_vel_scale"]

            period = 0.8
            count = counter * simulation_dt
            phase = count % period / period
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)

            obs[:3] = omega
            obs[3:6] = gravity_orientation
            obs[6:9] = cmd * g1["cmd_scale"]
            obs[9 : 9 + num_actions] = qj
            obs[9 + num_actions : 9 + 2 * num_actions] = dqj
            obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
            obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            # policy inference
            action = policy(obs_tensor).detach().numpy().squeeze()
            # transform action to target_dof_pos
            target_dof_pos = action * g1["action_scale"] + default_angles

            # 50 Hz row k: state the policy read, its observation, its action.
            rows["t"].append(d.time)
            rows["step"].append(counter)
            rows["qpos"].append(d.qpos.copy())
            rows["qvel"].append(d.qvel.copy())
            rows["cmd"].append(cmd.copy())
            rows["phase"].append(phase)
            rows["obs"].append(obs.copy())
            rows["action"].append(action.copy())
            rows["target_q"].append(target_dof_pos.copy())

    out = {key: np.array(val) for key, val in {**rows, **fine}.items()}
    out["push_row"] = np.array(events["push_row"], dtype=np.int64)
    out["push_t"] = np.array(events["push_t"], dtype=np.float64)
    out["push_dv"] = np.array(events["push_dv"], dtype=np.float64).reshape(-1, 3)
    out["push_vel_before"] = np.array(events["push_vel_before"], dtype=np.float64).reshape(-1, 3)
    out["fell"] = np.array(fell)
    out["fall_time"] = np.array(fall_time)
    out["fall_geom"] = np.array(fall_geom)
    out["fall_body"] = np.array(m.body(m.geom_bodyid[fall_geom]).name if fell else "")
    return out


def run_one(cfg, sim, run_index):
    """Seed everything for run run_index, sample its schedule, and run it."""
    g1 = sim["g1"]
    seed = cfg["base_seed"] + run_index
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)  # the policy is deterministic; set anyway (MuJoCo has no RNG here)
    policy_dt = g1["simulation_dt"] * g1["control_decimation"]
    n_rows = cfg["episode_policy_steps"]
    cmds, pushes = sample_schedule(rng, cfg, n_rows, policy_dt)
    data = run_episode(sim, cmds, pushes, n_rows, round(cfg["cmd_period_s"] / policy_dt),
                       cfg["contact_force_threshold"])
    data["seed"] = np.array(seed)
    data["run_index"] = np.array(run_index)
    return data


def write_meta(path, sim):
    m = sim["m"]
    commit = subprocess.run(["git", "-C", sim["unitree_dir"], "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "unitree_rl_gym_commit": commit,
        "mujoco_version": mujoco.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "joint_names": [m.joint(j).name for j in range(m.njnt)],
        "qpos_layout": "qpos[0:3] pelvis position (world), qpos[3:7] pelvis quaternion (w, x, y, z), "
                       "qpos[7:19] hinge joints in joint_names[1:] order",
        "foot_geom_ids": {"left": sim["foot_geoms"][0], "right": sim["foot_geoms"][1]},
        "floor_geom_id": sim["floor_id"],
        "robot_mass_kg": float(m.body_mass.sum()),
        "gravity": m.opt.gravity.tolist(),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to collect_data.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    sim = load_sim(cfg)
    for name, geoms in zip(cfg["foot_bodies"], sim["foot_geoms"]):
        print(f"foot geoms on {name}: {geoms}")

    out_dir = os.path.join(REPO_ROOT, cfg["out_dir"], datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    os.makedirs(out_dir)  # raises if the folder already exists: never overwrite
    shutil.copy(args.config, os.path.join(out_dir, "collect_data.yaml"))
    write_meta(os.path.join(out_dir, "meta.json"), sim)
    print(f"writing to {out_dir}")

    start = time.time()
    for i in range(cfg["n_runs"]):
        data = run_one(cfg, sim, i)
        np.savez_compressed(os.path.join(out_dir, f"run_{i:03d}.npz"), **data)
        status = f"FELL at t={float(data['fall_time']):.3f} s ({data['fall_body']})" if data["fell"] else "ok"
        print(f"run {i:3d} seed {int(data['seed'])}: {len(data['t'])} rows, {status}")
    elapsed = time.time() - start
    print(f"collection took {elapsed:.1f} s ({elapsed / cfg['n_runs']:.2f} s per run); "
          f"estimate for 300 runs: {elapsed / cfg['n_runs'] * 300 / 60:.1f} min")


if __name__ == "__main__":
    main()
