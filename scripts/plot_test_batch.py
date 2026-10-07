"""Plot one run of a collect_data.py folder, as a visual sanity check.

Saves to a new folder results/<YYYY-MM-DD_HHMMSS>_test_batch_plots/:
  pelvis_height.png      pelvis height, with push times and command changes
  foot_contact.png       500 Hz foot normal force and contact label, both feet
  foot_contact_zoom.png  the same over a 2 s window
  cmd_tracking.png       commanded vs actual forward, sideways, and yaw velocity

Actual velocities are in the frame the command uses in legged_gym:
forward/sideways = pelvis linear velocity rotated into the pelvis frame
(R^T v, R from the pelvis quaternion; legged_gym's base_lin_vel), and yaw rate
= qvel[5], the free joint's angular velocity about the pelvis z axis, which
MuJoCo already gives in the body frame (legged_gym's base_ang_vel[2]).

Usage: python scripts/plot_test_batch.py data/raw/<folder> [--run 0]
"""

import argparse
import datetime
import os

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZOOM_S = (6.0, 8.0)

BLUE, ORANGE = "#2a78d6", "#eb6834"   # series colors
INK, MUTED = "#222222", "#8a8a85"     # commanded values, event lines


def body_frame_velocity(qpos, qvel):
    """Rotate the world-frame pelvis velocity qvel[0:3] into the pelvis frame."""
    out = np.zeros((len(qpos), 3))
    R = np.zeros(9)
    for k in range(len(qpos)):
        mujoco.mju_quat2Mat(R, qpos[k, 3:7])
        out[k] = R.reshape(3, 3).T @ qvel[k, 0:3]
    return out


def mark_events(ax, push_t, cmd_change_t):
    for i, t in enumerate(push_t):
        ax.axvline(t, color=ORANGE, lw=1, ls="--", label="push" if i == 0 else None)
    for i, t in enumerate(cmd_change_t):
        ax.axvline(t, color=MUTED, lw=1, ls=":", label="command change" if i == 0 else None)


def style(ax):
    ax.grid(True, color="#e6e6e3", lw=0.6)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def plot_contact(d, title, path, t_range=None):
    t, force, contact = d["t_fine"], d["foot_force"], d["foot_contact"]
    sel = slice(None) if t_range is None else (t >= t_range[0]) & (t < t_range[1])
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for foot, (name, color) in enumerate([("left", BLUE), ("right", ORANGE)]):
        ax = axes[foot]
        top = force[sel, foot].max() * 1.05
        ax.fill_between(t[sel], 0, top, where=contact[sel, foot], step="post", color=color, alpha=0.15,
                        lw=0, label="contact label (force > 1 N)")
        ax.plot(t[sel], force[sel, foot], color=color, lw=1, label="normal force")
        ax.set_ylabel(f"{name} foot force [N]")
        ax.legend(loc="upper right", fontsize=8)
        style(ax)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--run", type=int, default=0)
    args = parser.parse_args()

    d = np.load(os.path.join(args.folder, f"run_{args.run:03d}.npz"))
    out_dir = os.path.join(REPO_ROOT, "results",
                           datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_test_batch_plots")
    os.makedirs(out_dir)  # raises if it exists: never overwrite
    title = f"{os.path.basename(os.path.normpath(args.folder))} run {args.run} (seed {int(d['seed'])})"

    t, qpos, qvel, cmd = d["t"], d["qpos"], d["qvel"], d["cmd"]
    cmd_change_t = t[1:][np.any(cmd[1:] != cmd[:-1], axis=1)]

    # (a) pelvis height
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, qpos[:, 2], color=BLUE, lw=1.5, label="pelvis height")
    mark_events(ax, d["push_t"], cmd_change_t)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("pelvis height [m]")
    ax.set_title(title, pad=24)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pelvis_height.png"), dpi=130)
    plt.close(fig)

    # (b) foot contact, full run and zoom
    plot_contact(d, title, os.path.join(out_dir, "foot_contact.png"))
    plot_contact(d, f"{title}, zoom {ZOOM_S[0]}-{ZOOM_S[1]} s", os.path.join(out_dir, "foot_contact_zoom.png"),
                 t_range=ZOOM_S)

    # (c) commanded vs actual velocity, in the pelvis frame
    v_body = body_frame_velocity(qpos, qvel)
    actual = [v_body[:, 0], v_body[:, 1], qvel[:, 5]]
    labels = ["forward velocity [m/s]", "sideways velocity [m/s]", "yaw rate [rad/s]"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, actual[i], color=BLUE, lw=1, label="actual (pelvis frame)")
        ax.plot(t, cmd[:, i], color=INK, lw=2, drawstyle="steps-post", label="commanded")
        mark_events(ax, d["push_t"], cmd_change_t)
        ax.set_ylabel(labels[i])
        style(ax)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=8, frameon=False)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cmd_tracking.png"), dpi=130)
    plt.close(fig)

    print(f"saved plots to {out_dir}")


if __name__ == "__main__":
    main()
