"""Plot rollout error against time for several models, from their rollout folders (scripts/eval_rollouts.py).

One figure per subset (all, contact, rest), 3 x 3 panels: one per state group, one for the foot contact force,
one diverged fraction panel and the legend.
  Group panels (log y): the median over windows of the per window group error (metrics.npz
    median_<subset>_<group>, raw units) against the time after the start state, t = j * dt for j = 1..H.
    One line per label: the mean over its seeds, with a band from the lowest to the highest seed.
    Labels whose folders have floor_median_<subset>_<group> also get that readout floor (mean over seeds),
    dashed, same color, no band (state groups: jepa; force: mlp, transformer and jepa).
    A train_mean label is one gray dashed line, no band.
  Force panel (log y): the same for the force group (N per foot), for the labels whose folders have it
    (folders made before the force group was added do not).
  Diverged panel (linear y): diverged_frac, mean over seeds and min to max band. metrics.npz has it for all
    windows only, so this panel is the same in the three figures. No readout floor here.

Checks (stops with an error otherwise):
  all folders use the same split and dataset, and the same H;
  per label, aggregate_seeds.check_seeds (training runs, same model, same config.yaml except the seed,
  different seeds) and at least 2 folders; all of a label's metrics.npz files have the same keys;
  a train_mean label (meta.json model train_mean) has exactly 1 folder.
dt is not stored in the dataset: it is read from the raw data the dataset was built from (the dataset's
meta.json source_raw_dir, column t of every raw run file). Stops if the t spacing is not the same in every
row of every run, or if a dataset run has a different number of rows than its raw run (the dataset must keep
every raw row, so that prediction j is j * dt after the start state).

Folder paths are relative to the root of this repo (or absolute).

Usage: python scripts/plot_rollouts.py --label <label> results/rollouts/<folder> ... [--label <label> <folder> ...]
Output: results/figures/<YYYY-MM-DD_HHMMSS>_rollouts/ with rollouts_<subset>.png and .pdf for each subset,
        and folders.txt (the folders used, per label, and dt).
"""

import argparse
import datetime
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from eval.rollout import SUBSETS  # noqa: E402
from features import STATE_GROUPS  # noqa: E402
from aggregate_seeds import check_seeds  # noqa: E402
from summarize_runs import full_path  # noqa: E402

# Slots 1 to 4 of the categorical palette of the dataviz skill's reference palette: blue, orange, aqua, yellow.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TRAIN_MEAN_COLOR = "0.45"  # gray
UNITS = {"height": "m", "gravity": "unit vector", "lin_vel": "m/s", "ang_vel": "rad/s",
         "joint_pos": "rad", "joint_vel": "rad/s", "force": "N per foot"}


def time_step(dataset):
    """Seconds between dataset rows, from the t column of the raw runs the dataset was built from.

    Stops with an error if a dataset run has a different number of rows than its raw run, or if the t spacing
    is not the same everywhere. Returns dt, the raw folder and the number of runs checked.
    """
    with open(os.path.join(os.path.dirname(full_path(dataset)), "meta.json"), "r") as f:
        raw_dir = full_path(json.load(f)["source_raw_dir"])
    data = np.load(full_path(dataset))
    run_ids = data["run_ids"]
    n_rows = data["states"].shape[1]  # states is (runs, rows, state dim)
    dts = []
    for run_id in run_ids:
        t = np.load(os.path.join(raw_dir, f"run_{int(run_id):03d}.npz"))["t"]
        if len(t) != n_rows:
            sys.exit(f"error: dataset run {run_id} has {n_rows} rows but its raw run has {len(t)}")
        spacing = np.diff(t)
        if not np.allclose(spacing, spacing[0], rtol=0, atol=1e-9):
            sys.exit(f"error: raw run {run_id} has uneven t spacing: {np.unique(spacing)}")
        dts.append(spacing[0])
    if not np.allclose(dts, dts[0], rtol=0, atol=1e-9):
        sys.exit(f"error: the raw runs have different t spacings: {np.unique(dts)}")
    return float(dts[0]), raw_dir, len(run_ids)


def load_label(label, folders):
    """Check and load one label's rollout folders. Returns a dict with the label's model, rollout metas,
    seeds (None for train_mean), run folders and metrics (one metrics.npz per folder)."""
    rollout_dirs = [full_path(r) for r in folders]
    metas = []
    for rollout_dir in rollout_dirs:
        with open(os.path.join(rollout_dir, "meta.json"), "r") as f:
            metas.append(json.load(f))
    if any(m["model"] == "train_mean" for m in metas):
        if len(metas) != 1:
            sys.exit(f"error: label {label} has a train_mean folder, so it must have exactly 1 folder")
        model, seeds, run_dirs = "train_mean", [None], [None]
    else:
        if len(metas) < 2:
            sys.exit(f"error: label {label} needs at least 2 rollout folders (one per seed)")
        metas, runs, split, model, all_keys, filled = check_seeds(rollout_dirs)
        seeds = [cfg["seed"] for cfg, log, meta in runs]
        run_dirs = [m["run_dir"] for m in metas]
    metrics = [np.load(os.path.join(r, "metrics.npz")) for r in rollout_dirs]
    if len({frozenset(m.files) for m in metrics}) != 1:
        sys.exit(f"error: label {label}: the metrics.npz files do not all have the same keys")
    keys = set(metrics[0].files)
    floor_groups = [g for g in list(STATE_GROUPS) + ["force"] if f"floor_median_all_{g}" in keys]
    return {"model": model, "metas": metas, "seeds": seeds, "run_dirs": run_dirs, "metrics": metrics,
            "rollout_dirs": [os.path.relpath(r, REPO_ROOT) for r in rollout_dirs], "keys": keys,
            "has_force": "median_all_force" in keys, "floor_groups": floor_groups}


def plot_label(ax, t, name, entry, color, key, floor):
    """One label in one panel: metrics key against t, mean over seeds with a min to max band (train_mean: one
    gray dashed line). With floor, also the mean of floor_<key> if the label has it, dashed."""
    values = np.stack([m[key] for m in entry["metrics"]])  # (seeds, H)
    if entry["model"] == "train_mean":
        ax.plot(t, values[0], color=TRAIN_MEAN_COLOR, linestyle="--", linewidth=1.5, label=name)
        return
    n = len(values)
    ax.fill_between(t, values.min(axis=0), values.max(axis=0), color=color, alpha=0.2, linewidth=0)
    ax.plot(t, values.mean(axis=0), color=color, linewidth=1.5, label=f"{name} (n = {n})")
    if floor and "floor_" + key in entry["keys"]:
        floor_mean = np.stack([m["floor_" + key] for m in entry["metrics"]]).mean(axis=0)
        ax.plot(t, floor_mean, color=color, linestyle="--", linewidth=1.2, label=f"{name} readout floor")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", nargs="+", action="append", required=True, metavar=("LABEL", "FOLDER"),
                        help="a label, then its rollout folders (one per seed); repeat for each model")
    args = parser.parse_args()
    names = [entry[0] for entry in args.label]
    if len(set(names)) != len(names):
        parser.error(f"a label is given twice: {names}")
    if any(len(entry) < 2 for entry in args.label):
        parser.error("each --label needs a label and at least one folder")

    labels = {entry[0]: load_label(entry[0], entry[1:]) for entry in args.label}
    if sum(e["model"] != "train_mean" for e in labels.values()) > len(COLORS):
        sys.exit(f"error: at most {len(COLORS)} labels besides train_mean (one color each)")
    all_metas = [m for e in labels.values() for m in e["metas"]]
    splits = {m["split"] for m in all_metas}
    if len(splits) != 1:
        sys.exit(f"error: the rollout folders use different splits: {sorted(splits)}")
    datasets = {os.path.normpath(m["dataset"]) for m in all_metas}
    if len(datasets) != 1:
        sys.exit(f"error: the rollout folders use different datasets: {sorted(datasets)}")
    Hs = {m["H"] for m in all_metas}
    if len(Hs) != 1:
        sys.exit(f"error: the rollout folders have different H: {sorted(Hs)}")
    split, dataset, H = splits.pop(), datasets.pop(), Hs.pop()
    dt, raw_dir, n_runs_checked = time_step(dataset)
    t = np.arange(1, H + 1) * dt  # prediction j (index j - 1) is j rows after the start state
    print(f"split {split}, dataset {dataset}, H = {H}, dt = {dt:.6g} s (raw t column of {n_runs_checked} runs "
          f"in {os.path.relpath(raw_dir, REPO_ROOT)}; each dataset run has as many rows as its raw run)")

    colors, i = {}, 0
    for name, entry in labels.items():
        if entry["model"] != "train_mean":
            colors[name] = COLORS[i]
            i += 1
        else:
            colors[name] = TRAIN_MEAN_COLOR

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "results", "figures", f"{stamp}_rollouts")
    os.makedirs(out_dir)  # raises if the folder already exists: never overwrite

    for subset in SUBSETS:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
        axes = axes.flat
        for ax, group in zip(axes, list(STATE_GROUPS) + ["force"]):
            for name, entry in labels.items():
                if group != "force" or entry["has_force"]:
                    plot_label(ax, t, name, entry, colors[name], f"median_{subset}_{group}", floor=True)
            ax.set_yscale("log")
            ax.set_title("foot contact force" if group == "force" else group)
            ax.set_ylabel(f"median error [{UNITS[group]}]")
        n_groups = len(STATE_GROUPS) + 1
        ax = axes[n_groups]
        for name, entry in labels.items():
            plot_label(ax, t, name, entry, colors[name], "diverged_frac", floor=False)
        ax.set_title("diverged fraction (all windows)")
        ax.set_ylabel("fraction of windows")
        for ax in axes[: n_groups + 1]:
            ax.grid(True, which="major", alpha=0.3)
        for ax in axes[6:]:
            ax.set_xlabel("time after start state [s]")
        legend_ax = axes[n_groups + 1]
        legend_ax.axis("off")
        handles = {}  # label text -> line, from every panel (the mlp and transformer floors are only in force)
        for ax in axes[:n_groups]:
            for handle, text in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(text, handle)
        legend_ax.legend(handles.values(), handles.keys(), loc="center", fontsize=9, frameon=False)
        fig.suptitle(f"Rollout error, subset {subset}, {split} split. Median over windows of the per window group "
                     f"error; line = mean over seeds, band = min to max over seeds, dashed = readout floor")
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(os.path.join(out_dir, f"rollouts_{subset}.{ext}"), dpi=150)
        plt.close(fig)

    lines = [f"command: python scripts/plot_rollouts.py " + " ".join(sys.argv[1:]),
             f"split {split}, dataset {dataset}, H {H}",
             f"dt {dt!r} s, from the t column of {n_runs_checked} raw runs in {os.path.relpath(raw_dir, REPO_ROOT)}",
             ""]
    for name, entry in labels.items():
        lines.append(f"label {name}: model {entry['model']}, n folders {len(entry['metrics'])}, "
                     f"force {entry['has_force']}, readout floor groups {entry['floor_groups']}, "
                     f"color {colors[name]}")
        for seed, rollout_dir, run_dir in zip(entry["seeds"], entry["rollout_dirs"], entry["run_dirs"]):
            lines.append(f"  seed {seed}  {rollout_dir}  (run {run_dir})")
        lines.append("")
    with open(os.path.join(out_dir, "folders.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
