"""Summarize training runs of src/train.py, and optionally their rollout evaluations.

One row per run (from config.yaml, log.csv, meta.json):
  width, depth, lr, seed, parameters, epochs, final train loss, final tuning loss,
  lowest tuning loss and its epoch, total clipped steps, total time (meta.json total_seconds)
With --rollouts (folders of scripts/eval_rollouts.py, one per run, in the same order):
  RMS error of every state group at j = 1, 10, 50, 100 (subsets all, contact, rest), from
  metrics.npz. Each rollout folder's meta.json run_dir must be its run folder.

Folder paths are relative to the root of this repo (or absolute).

Usage: python scripts/summarize_runs.py results/train/<run> ... [--rollouts results/rollouts/<folder> ...]
Output: results/summaries/<YYYY-MM-DD_HHMMSS>/ with table.txt (the printed table) and losses.png
        (train and tuning loss against epoch, log scale).
"""

import argparse
import csv
import datetime
import json
import os
import sys

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from eval.rollout import SUBSETS  # noqa: E402
from features import STATE_GROUPS  # noqa: E402

REPORT_STEPS = [1, 10, 50, 100]


def full_path(path):
    """Absolute path of a repo-relative (or absolute) path, without a trailing slash."""
    return os.path.realpath(os.path.join(REPO_ROOT, path))


def load_run(run_dir):
    """config, log (dict of column -> array, from log.csv) and meta of one training run."""
    with open(os.path.join(run_dir, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(run_dir, "log.csv"), "r") as f:
        rows = list(csv.DictReader(f))
    log = {column: np.array([float(row[column]) for row in rows]) for column in rows[0]}
    with open(os.path.join(run_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    return cfg, log, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="results/train/<run> folders")
    parser.add_argument("--rollouts", nargs="+", help="results/rollouts/<folder>, one per run, same order")
    args = parser.parse_args()
    if args.rollouts is not None and len(args.rollouts) != len(args.runs):
        parser.error(f"{len(args.runs)} runs but {len(args.rollouts)} rollout folders")

    runs = [full_path(r) for r in args.runs]
    names = [os.path.basename(r) for r in runs]
    loaded = [load_run(r) for r in runs]
    if args.rollouts is not None:
        rollout_dirs = [full_path(r) for r in args.rollouts]
        for run_dir, rollout_dir in zip(runs, rollout_dirs):
            with open(os.path.join(rollout_dir, "meta.json"), "r") as f:
                rollout_meta = json.load(f)
            if full_path(rollout_meta.get("run_dir", "")) != run_dir:
                sys.exit(f"error: {rollout_dir} has run_dir {rollout_meta.get('run_dir')}, expected {run_dir}")

    lines = []

    def out(line=""):
        print(line)
        lines.append(line)

    name_w = max(len(n) for n in names)
    out(f"{'run':{name_w}s} {'width':>5s} {'depth':>5s} {'lr':>7s} {'seed':>4s} {'params':>9s} {'epochs':>6s}"
        f" {'train end':>10s} {'tuning end':>10s} {'tuning min':>10s} {'at epoch':>8s} {'clipped':>7s} {'time s':>7s}")
    for name, (cfg, log, meta) in zip(names, loaded):
        best = np.argmin(log["tuning_loss"])
        out(f"{name:{name_w}s} {cfg['width']:5d} {cfg['depth']:5d} {cfg['lr']:7g} {cfg['seed']:4d}"
            f" {meta['n_parameters']:9d} {len(log['epoch']):6d}"
            f" {log['train_loss'][-1]:10.3e} {log['tuning_loss'][-1]:10.3e}"
            f" {log['tuning_loss'][best]:10.3e} {int(log['epoch'][best]):8d}"
            f" {int(log['n_clipped'].sum()):7d} {meta['total_seconds']:7.0f}")

    if args.rollouts is not None:
        metrics = [np.load(os.path.join(r, "metrics.npz")) for r in rollout_dirs]
        out()
        out("RMS rollout error in raw units, j = steps after the start state")
        out(f"{'group':10s} {'subset':8s} {'run':{name_w}s}" + "".join(f"{'j = ' + str(j):>12s}" for j in REPORT_STEPS))
        for subset in SUBSETS:
            for group in STATE_GROUPS:
                for name, m in zip(names, metrics):
                    rms = m[f"rms_{subset}_{group}"]
                    out(f"{group:10s} {subset:8s} {name:{name_w}s}" + "".join(f"{rms[j - 1]:12.4e}" for j in REPORT_STEPS))
                out()

    out_dir = os.path.join(REPO_ROOT, "results", "summaries", datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    os.makedirs(out_dir)  # raises if the folder already exists: never overwrite
    with open(os.path.join(out_dir, "table.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, (cfg, log, meta)) in enumerate(zip(names, loaded)):
        color = f"C{i}"
        ax.plot(log["epoch"], log["train_loss"], color=color, linestyle="--", label=f"{name} train")
        ax.plot(log["epoch"], log["tuning_loss"], color=color, linestyle="-", label=f"{name} tuning")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE on normalized delta")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "losses.png"), dpi=130)
    plt.close(fig)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
