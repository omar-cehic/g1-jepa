"""Mean and spread across seeds of the rollout evaluations of one training setting.

Takes rollout folders of scripts/eval_rollouts.py (--model mlp, transformer or jepa), one per seed.
Checks that all folders used the same split and the same model, and that their training runs have
the same config.yaml except the seed, and different seeds; stops with an error otherwise. A key that
an older config.yaml lacks gets the value from ADDED_LATER (the default the training scripts use when
the key is missing); the printed table lists which runs needed this. Prints each run's git commit.

Prints, over seeds: the mean, the sample standard deviation (ddof = 1) and the number of seeds of
  the final tuning loss (last row of log.csv: column tuning_loss for mlp and transformer,
  tuning_pred_loss for jepa),
  for each state group and subset (all, contact, rest): the RMS, median and p90 at j = 1, 10, 50, 100,
  the same for the force group (foot contact force, N per foot) if the metrics.npz files have it,
  the diverged fraction at j = 1, 10, 50, 100.
If the metrics.npz files have floor_ keys, prints the same table for them, titled "readout floor", with the
groups they have (jepa: the state groups, from the probe on the true embeddings, and the force group;
mlp and transformer: the force group only, from the force readout on the true states). The diverged
fraction row is printed only where the files have it. All folders must have the same groups.

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
# config.yaml keys added after some runs were trained, with the value train.py and train_jepa.py use
# when the key is missing (cfg.get(key, default)).
ADDED_LATER = {"model": "mlp", "history": 1, "predictor": "mlp", "stop_grad": False, "ema_tau": 0.0}
TUNING_LOSS_COLUMN = {"mlp": "tuning_loss", "transformer": "tuning_loss", "jepa": "tuning_pred_loss"}
# SAVED in per_seed.npz, one entry per seed (axis 0), in the order given on the command line:
#   seeds, run_dirs, rollout_dirs, git_commits, steps (= REPORT_STEPS, not per seed),
#   tuning_loss_column (the log.csv column, not per seed), tuning_loss_final (n,),
#   <stat>_<subset>_<group> (n, 4), diverged_frac (n, 4),
#   and if the metrics.npz files have them: floor_<stat>_<subset>_<group> (n, 4), floor_diverged_frac (n, 4)
#   (<group> is a state group or force)
GROUPS = list(STATE_GROUPS) + ["force"]


def mean_std(values):
    """Mean and sample std (ddof = 1) over axis 0."""
    return np.mean(values, axis=0), np.std(values, axis=0, ddof=1)


def metric_groups(metrics):
    """{prefix: groups} for prefix "" and "floor_": the groups (of GROUPS) that have <prefix>rms_all_<group>
    in every metrics.npz. Stops with an error if some files have a group and some do not."""
    groups = {}
    for prefix in ["", "floor_"]:
        for group in GROUPS:
            has = {f"{prefix}rms_all_{group}" in m.files for m in metrics}
            if len(has) != 1:
                sys.exit(f"error: some metrics.npz files have {prefix}rms_all_{group} and some do not")
            if has.pop():
                groups.setdefault(prefix, []).append(group)
    return groups


def check_seeds(rollout_dirs):
    """Load and check the rollout folders of one training setting, one per seed.

    Checks that all folders are evaluations of training runs (meta.json has run_dir), use the same split and
    the same model, and that their runs have the same config.yaml except the seed (keys missing from an older
    config.yaml are filled in from ADDED_LATER) and different seeds. Stops with an error otherwise.
    Returns rollout_metas (meta.json per folder), runs ((cfg, log, meta) per folder, cfg with the ADDED_LATER
    keys filled in), split, model, all_keys (sorted config keys except seed) and filled (per run, the keys
    filled in from ADDED_LATER).
    """
    rollout_metas, runs = [], []
    for rollout_dir in rollout_dirs:
        with open(os.path.join(rollout_dir, "meta.json"), "r") as f:
            rollout_meta = json.load(f)
        if "run_dir" not in rollout_meta:
            sys.exit(f"error: {rollout_dir} has no run_dir (not the evaluation of a training run)")
        rollout_metas.append(rollout_meta)
        runs.append(load_run(full_path(rollout_meta["run_dir"])))  # (cfg, log, meta)

    splits = {m["split"] for m in rollout_metas}
    if len(splits) != 1:
        sys.exit(f"error: the rollout folders use different splits: {sorted(splits)}")
    models = {m["model"] for m in rollout_metas}
    if len(models) != 1:
        sys.exit(f"error: the rollout folders have different models: {sorted(models)}")
    all_keys = sorted(set().union(*[cfg.keys() for cfg, log, meta in runs]) - {"seed"})
    filled = []  # per run, the keys filled in from ADDED_LATER
    for cfg, log, meta in runs:
        missing = [key for key in all_keys if key not in cfg]
        for key in missing:
            if key not in ADDED_LATER:
                sys.exit(f"error: a run with seed {cfg['seed']} has no {key} in config.yaml")
            cfg[key] = ADDED_LATER[key]
        filled.append(missing)
    for key in all_keys:
        values = [cfg[key] for cfg, log, meta in runs]
        if any(v != values[0] for v in values):
            sys.exit(f"error: the runs have different {key}: {values}")
    seeds = [cfg["seed"] for cfg, log, meta in runs]
    if len(set(seeds)) != len(seeds):
        sys.exit(f"error: some runs share a seed: {seeds}")
    return rollout_metas, runs, splits.pop(), models.pop(), all_keys, filled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label", help="name of the setting, used in the output folder name")
    parser.add_argument("rollouts", nargs="+", help="results/rollouts/<folder>, one per seed")
    args = parser.parse_args()
    if len(args.rollouts) < 2:
        parser.error("need at least 2 rollout folders for a standard deviation")

    rollout_dirs = [full_path(r) for r in args.rollouts]
    rollout_metas, runs, split, model, all_keys, filled = check_seeds(rollout_dirs)
    seeds = [cfg["seed"] for cfg, log, meta in runs]
    loss_column = TUNING_LOSS_COLUMN[model]
    cfg0 = runs[0][0]

    lines = []

    def out(line=""):
        print(line)
        lines.append(line)

    out(f"label {args.label}, split {split}, model {model}, n seeds {len(seeds)}")
    out("config (all except seed): " + ", ".join(f"{key} {cfg0[key]}" for key in all_keys))
    if any(filled):
        out("filled in from ADDED_LATER (missing from config.yaml): " + "; ".join(
            f"seed {cfg['seed']}: {', '.join(keys)}" for (cfg, log, meta), keys in zip(runs, filled) if keys))
    else:
        out("filled in from ADDED_LATER (missing from config.yaml): none")
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
        "tuning_loss_column": np.array(loss_column),
        "tuning_loss_final": np.array([log[loss_column][-1] for cfg, log, meta in runs]),
    }
    metrics = [np.load(os.path.join(r, "metrics.npz")) for r in rollout_dirs]
    groups = metric_groups(metrics)
    idx = np.array(REPORT_STEPS) - 1  # prediction j is at index j - 1
    for prefix in groups:
        for stat in STATS:
            for subset in SUBSETS:
                for group in groups[prefix]:
                    key = f"{prefix}{stat}_{subset}_{group}"
                    per_seed[key] = np.stack([m[key][idx] for m in metrics])
        if prefix + "diverged_frac" in metrics[0].files:
            per_seed[prefix + "diverged_frac"] = np.stack([m[prefix + "diverged_frac"][idx] for m in metrics])

    n = len(seeds)
    out()
    mean, std = mean_std(per_seed["tuning_loss_final"])
    out(f"final tuning loss (log.csv column {loss_column}): mean {mean:.4e}  std {std:.4e}  n {n}")

    def out_table(prefix):
        """The mean ± std table of the per_seed keys that start with prefix."""
        out(f"{'group':10s} {'subset':8s} {'stat':6s}" + "".join(f"{'j = ' + str(j):>24s}" for j in REPORT_STEPS))
        for group in groups[prefix]:
            for subset in SUBSETS:
                for stat in STATS:
                    mean, std = mean_std(per_seed[f"{prefix}{stat}_{subset}_{group}"])
                    cells = "".join(f"{m:12.4e} ± {s:9.2e}" for m, s in zip(mean, std))
                    out(f"{group:10s} {subset:8s} {stat:6s}{cells}")
            out()
        if prefix + "diverged_frac" not in per_seed:
            return
        mean, std = mean_std(per_seed[prefix + "diverged_frac"])
        out(f"{'diverged':10s} {'all':8s} {'frac':6s}" + "".join(f"{m:12.4f} ± {s:9.4f}" for m, s in zip(mean, std)))

    out()
    force_units = " (force: N per foot)" if "force" in groups[""] else ""
    out(f"rollout errors in raw units{force_units}, mean ± std over n = {n} seeds, j = steps after the start state")
    out_table("")
    if "floor_" in groups:
        what = []
        if groups["floor_"][0] != "force":
            what.append("state groups: the probe on the true embeddings")
        if "force" in groups["floor_"]:
            what.append("force: the force readout on the true x of row k+j-1")
        out()
        out(f"readout floor ({'; '.join(what)}), raw units, mean ± std over n = {n} seeds")
        out_table("floor_")

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "results", "summaries", f"{stamp}_{args.label}")
    os.makedirs(out_dir)  # raises if the folder already exists: never overwrite
    with open(os.path.join(out_dir, "table.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    np.savez(os.path.join(out_dir, "per_seed.npz"), **per_seed)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
