"""Mean and spread across seeds of the rollout evaluations of one training setting.

Takes rollout folders of scripts/eval_rollouts.py (--model mlp), one per seed. Checks that all
folders used the same split and that their training runs have the same width, depth, lr and
epochs but different seeds; stops with an error otherwise. Prints each run's git commit.

Prints, over seeds: the mean, the sample standard deviation (ddof = 1) and the number of seeds of
  the final tuning loss (last row of log.csv),
  for each state group and subset (all, contact, rest): the RMS, median and p90 at j = 1, 10, 50, 100,
  the diverged fraction at j = 1, 10, 50, 100.

Folder paths are relative to the root of this repo (or absolute).

Usage: python scripts/aggregate_seeds.py <label> results/rollouts/<folder> ...
Output: results/summaries/<YYYY-MM-DD_HHMMSS>_<label>/ with table.txt (the printed table) and
        per_seed.npz (the per seed numbers behind every mean and std; see SAVED below).
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from eval.rollout import SUBSETS  # noqa: E402
from features import STATE_GROUPS  # noqa: E402
from summarize_runs import full_path, load_run  # noqa: E402

REPORT_STEPS = [1, 10, 50, 100]
STATS = ["rms", "median", "p90"]
SAME_KEYS = ["width", "depth", "lr", "epochs"]
# SAVED in per_seed.npz, one entry per seed (axis 0), in the order given on the command line:
#   seeds, run_dirs, rollout_dirs, git_commits, steps (= REPORT_STEPS, not per seed),
#   tuning_loss_final (n,), <stat>_<subset>_<group> (n, 4), diverged_frac (n, 4)


def mean_std(values):
    """Mean and sample std (ddof = 1) over axis 0."""
    return np.mean(values, axis=0), np.std(values, axis=0, ddof=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label", help="name of the setting, used in the output folder name")
    parser.add_argument("rollouts", nargs="+", help="results/rollouts/<folder>, one per seed")
    args = parser.parse_args()
    if len(args.rollouts) < 2:
        parser.error("need at least 2 rollout folders for a standard deviation")

    rollout_dirs = [full_path(r) for r in args.rollouts]
    rollout_metas, runs = [], []
    for rollout_dir in rollout_dirs:
        with open(os.path.join(rollout_dir, "meta.json"), "r") as f:
            rollout_meta = json.load(f)
        if "run_dir" not in rollout_meta:
            sys.exit(f"error: {rollout_dir} has no run_dir (not an mlp evaluation)")
        rollout_metas.append(rollout_meta)
        runs.append(load_run(full_path(rollout_meta["run_dir"])))  # (cfg, log, meta)

    splits = {m["split"] for m in rollout_metas}
    if len(splits) != 1:
        sys.exit(f"error: the rollout folders use different splits: {sorted(splits)}")
    for key in SAME_KEYS:
        values = {cfg[key] for cfg, log, meta in runs}
        if len(values) != 1:
            sys.exit(f"error: the runs have different {key}: {sorted(values)}")
    seeds = [cfg["seed"] for cfg, log, meta in runs]
    if len(set(seeds)) != len(seeds):
        sys.exit(f"error: some runs share a seed: {seeds}")
    split = splits.pop()
    cfg0 = runs[0][0]

    lines = []

    def out(line=""):
        print(line)
        lines.append(line)

    out(f"label {args.label}, split {split}, width {cfg0['width']}, depth {cfg0['depth']},"
        f" lr {cfg0['lr']:g}, epochs {cfg0['epochs']}, n seeds {len(seeds)}")
    out()
    out(f"{'seed':>4s}  {'git commit':40s}  {'uncommitted':11s}  run, rollout folder")
    for (cfg, log, meta), rollout_meta, rollout_dir in zip(runs, rollout_metas, rollout_dirs):
        out(f"{cfg['seed']:4d}  {meta['git_commit']:40s}  {str(meta['git_uncommitted_changes']):11s}"
            f"  {rollout_meta['run_dir']}, {os.path.relpath(rollout_dir, REPO_ROOT)}")

    per_seed = {
        "seeds": np.array(seeds),
        "run_dirs": np.array([m["run_dir"] for m in rollout_metas]),
        "rollout_dirs": np.array([os.path.relpath(r, REPO_ROOT) for r in rollout_dirs]),
        "git_commits": np.array([meta["git_commit"] for cfg, log, meta in runs]),
        "steps": np.array(REPORT_STEPS),
        "tuning_loss_final": np.array([log["tuning_loss"][-1] for cfg, log, meta in runs]),
    }
    metrics = [np.load(os.path.join(r, "metrics.npz")) for r in rollout_dirs]
    idx = np.array(REPORT_STEPS) - 1  # prediction j is at index j - 1
    for stat in STATS:
        for subset in SUBSETS:
            for group in STATE_GROUPS:
                key = f"{stat}_{subset}_{group}"
                per_seed[key] = np.stack([m[key][idx] for m in metrics])
    per_seed["diverged_frac"] = np.stack([m["diverged_frac"][idx] for m in metrics])

    n = len(seeds)
    out()
    mean, std = mean_std(per_seed["tuning_loss_final"])
    out(f"final tuning loss (MSE on normalized delta): mean {mean:.4e}  std {std:.4e}  n {n}")
    out()
    out(f"rollout errors in raw units, mean ± std over n = {n} seeds, j = steps after the start state")
    out(f"{'group':10s} {'subset':8s} {'stat':6s}" + "".join(f"{'j = ' + str(j):>24s}" for j in REPORT_STEPS))
    for group in STATE_GROUPS:
        for subset in SUBSETS:
            for stat in STATS:
                mean, std = mean_std(per_seed[f"{stat}_{subset}_{group}"])
                cells = "".join(f"{m:12.4e} ± {s:9.2e}" for m, s in zip(mean, std))
                out(f"{group:10s} {subset:8s} {stat:6s}{cells}")
        out()
    mean, std = mean_std(per_seed["diverged_frac"])
    out(f"{'diverged':10s} {'all':8s} {'frac':6s}" + "".join(f"{m:12.4f} ± {s:9.4f}" for m, s in zip(mean, std)))

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "results", "summaries", f"{stamp}_{args.label}")
    os.makedirs(out_dir)  # raises if the folder already exists: never overwrite
    with open(os.path.join(out_dir, "table.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    np.savez(os.path.join(out_dir, "per_seed.npz"), **per_seed)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
