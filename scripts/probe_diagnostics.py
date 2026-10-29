"""Probe diagnostics for trained JEPA runs: where the probe error is, and how much the probe itself matters.

For each run folder (written by src/train_jepa.py), with the frozen JEPA and the train and tuning
pairs built exactly as in scripts/score_jepa.py (embeddings computed once, JEPA never updated,
probes fitted on train only). All errors are squared errors in normalized state units.
  (a) Per feature: fit the linear probe and the default MLP probe (seed 0) encode(s_k) -> s_k on
      train. On tuning, the present error of each state feature (mean over pairs) for both
      probes, and the mean over the features of each group. Features and groups follow the
      dataset's state_type: STATE_NAMES and STATE_GROUPS (base, 34), FULL_STATE_NAMES and
      FULL_STATE_GROUPS (full, 37) or FULL_NO_XY_STATE_NAMES and FULL_NO_XY_STATE_GROUPS
      (full_no_xy, 35). All runs of one call must use the same state_type.
  (b) Train vs tuning: present error (mean over pairs and features) of the default MLP probe from
      (a) on the train pairs and on the tuning pairs.
  (c) Probe capacity: a bigger MLP probe (width 1024, depth 3, 100 epochs, seed 0, everything else
      the same). Its present error on train and tuning, overall and per group.
  (d) Probe seed noise: the default MLP probe fitted with seeds 0, 1 and 2. For each seed
      mlp_score and mlp_present_mse on tuning (same definitions as score_jepa.py), plus their mean
      and max minus min.
Nothing is selected here.

Usage: python scripts/probe_diagnostics.py results/train/<run folder> [results/train/<run folder> ...]
Output: results/<YYYY-MM-DD_HHMMSS>_probe_diag/ with per_feature.csv, summary.csv and meta.json.
"""

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from build_dataset import git_commit  # noqa: E402
from eval.probe import fit_linear_probe, apply_probe, fit_mlp_probe, apply_mlp_probe  # noqa: E402
from features import (STATE_NAMES, STATE_GROUPS, FULL_STATE_NAMES, FULL_STATE_GROUPS,  # noqa: E402
                      FULL_NO_XY_STATE_NAMES, FULL_NO_XY_STATE_GROUPS)
from score_jepa import load_jepa, embed  # noqa: E402
from train import dataset_state_type  # noqa: E402
from train_jepa import pairs  # noqa: E402

BIG_WIDTH = 1024
BIG_DEPTH = 3
BIG_EPOCHS = 100
SEEDS = [0, 1, 2]


def per_feature_mse(pred, target):
    """Mean over pairs of the squared error, one number per feature (state_dim,)."""
    return np.mean((pred - target) ** 2, axis=0)


def group_means(feature_mse, groups):
    """Mean of the per feature errors within each group of groups (name -> slice), as a dict."""
    return {group: float(feature_mse[cols].mean()) for group, cols in groups.items()}


def run_state_type(run_dir):
    """state_type ("base", "full" or "full_no_xy") of the dataset named in run_dir's config.yaml."""
    with open(os.path.join(run_dir, "config.yaml"), "r") as f:
        return dataset_state_type(yaml.safe_load(f)["dataset"])


def diagnose_run(run_dir, device):
    """Per feature rows, one summary row (dict) and the run time in seconds of each part."""
    times = {}
    t = time.time()
    with open(os.path.join(run_dir, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    data = dict(np.load(os.path.join(REPO_ROOT, cfg["dataset"])))
    if dataset_state_type(cfg["dataset"]) == "full":
        names, groups = FULL_STATE_NAMES, FULL_STATE_GROUPS
    elif dataset_state_type(cfg["dataset"]) == "full_no_xy":
        names, groups = FULL_NO_XY_STATE_NAMES, FULL_NO_XY_STATE_GROUPS
    else:
        names, groups = STATE_NAMES, STATE_GROUPS
    model = load_jepa(run_dir, device)
    s_train, a_train, _ = pairs(data, "train")
    z_train, _ = embed(model, s_train, a_train, device)
    s, a, s_next = pairs(data, "tuning")
    z, z_hat = embed(model, s, a, device)
    s_train64, s64, s_next64 = (x.astype(np.float64) for x in (s_train, s, s_next))
    times["load_and_embed"] = time.time() - t

    # (a) linear and default MLP probe, per feature on tuning
    t = time.time()
    W, b = fit_linear_probe(z_train, s_train64)
    mlp_probe = fit_mlp_probe(z_train, s_train, device, seed=0)
    linear_feat = per_feature_mse(apply_probe(z, W, b), s64)
    mlp_feat = per_feature_mse(apply_mlp_probe(mlp_probe, z, device), s64)
    times["a"] = time.time() - t

    # (b) default MLP probe from (a): train vs tuning
    t = time.time()
    mlp_train = float(np.mean((apply_mlp_probe(mlp_probe, z_train, device) - s_train64) ** 2))
    mlp_tuning = float(mlp_feat.mean())
    times["b"] = time.time() - t

    # (c) bigger MLP probe
    t = time.time()
    big_probe = fit_mlp_probe(z_train, s_train, device, seed=0, epochs=BIG_EPOCHS,
                              width=BIG_WIDTH, depth=BIG_DEPTH)
    big_train_feat = per_feature_mse(apply_mlp_probe(big_probe, z_train, device), s_train64)
    big_tuning_feat = per_feature_mse(apply_mlp_probe(big_probe, z, device), s64)
    times["c"] = time.time() - t

    # (d) default MLP probe with several seeds
    t = time.time()
    seed_scores, seed_presents = [], []
    for seed in SEEDS:
        probe = fit_mlp_probe(z_train, s_train, device, seed=seed)
        seed_scores.append(float(np.mean((apply_mlp_probe(probe, z_hat, device) - s_next64) ** 2)))
        seed_presents.append(float(np.mean((apply_mlp_probe(probe, z, device) - s64) ** 2)))
    times["d"] = time.time() - t

    name = os.path.basename(os.path.normpath(run_dir))
    group_of = {i: group for group, cols in groups.items() for i in range(len(names))[cols]}
    feature_rows = [{"run": name, "feature": names[i], "group": group_of[i],
                     "linear_mse": float(linear_feat[i]), "mlp_mse": float(mlp_feat[i]),
                     "big_mlp_mse": float(big_tuning_feat[i])} for i in range(len(names))]

    summary = {"run": name, "mlp_train_present_mse": mlp_train, "mlp_tuning_present_mse": mlp_tuning,
               "big_mlp_train_present_mse": float(big_train_feat.mean()),
               "big_mlp_tuning_present_mse": float(big_tuning_feat.mean())}
    for group, value in group_means(big_train_feat, groups).items():
        summary[f"big_mlp_train_{group}"] = value
    for group, value in group_means(big_tuning_feat, groups).items():
        summary[f"big_mlp_tuning_{group}"] = value
    for label, values in [("mlp_score", seed_scores), ("mlp_present_mse", seed_presents)]:
        for seed, value in zip(SEEDS, values):
            summary[f"{label}_s{seed}"] = value
        summary[f"{label}_mean"] = float(np.mean(values))
        summary[f"{label}_max_minus_min"] = max(values) - min(values)

    # per group table and the (b), (c), (d) numbers
    linear_g, mlp_g = group_means(linear_feat, groups), group_means(mlp_feat, groups)
    big_train_g, big_tuning_g = group_means(big_train_feat, groups), group_means(big_tuning_feat, groups)
    print(f"  {'group':10s} {'linear':>10s} {'mlp':>10s} {'big_train':>10s} {'big_tuning':>10s}   (present error)")
    for group in groups:
        print(f"  {group:10s} {linear_g[group]:10.6f} {mlp_g[group]:10.6f} "
              f"{big_train_g[group]:10.6f} {big_tuning_g[group]:10.6f}")
    print(f"  {'all':10s} {linear_feat.mean():10.6f} {mlp_feat.mean():10.6f} "
          f"{big_train_feat.mean():10.6f} {big_tuning_feat.mean():10.6f}")
    print(f"  (b) default MLP probe present error: train {mlp_train:.6f}, tuning {mlp_tuning:.6f}")
    print(f"  (c) big MLP probe present error: train {big_train_feat.mean():.6f}, "
          f"tuning {big_tuning_feat.mean():.6f}")
    for label, values in [("mlp_score", seed_scores), ("mlp_present_mse", seed_presents)]:
        each = ", ".join(f"s{seed} {v:.6f}" for seed, v in zip(SEEDS, values))
        print(f"  (d) {label}: {each}; mean {np.mean(values):.6f}, max - min {max(values) - min(values):.6f}")
    print("  time: " + ", ".join(f"{part} {sec:.1f} s" for part, sec in times.items()))
    return feature_rows, summary


def write_csv(path, rows):
    """rows (list of dicts with the same keys) as a csv file, columns in key order."""
    with open(path, "w") as f:
        f.write(",".join(rows[0].keys()) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row.values()) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="results/train/<run folder> of JEPA runs")
    args = parser.parse_args()
    # summary.csv has one column per group, so all runs must have the same groups
    state_types = {run_state_type(os.path.join(REPO_ROOT, run_dir)) for run_dir in args.run_dirs}
    assert len(state_types) == 1, f"runs mix state types {sorted(state_types)}: score them in separate calls"

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "results", f"{stamp}_probe_diag")
    os.makedirs(out_dir)  # raises if the folder already exists: never overwrite

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))

    feature_rows, summaries = [], []
    for run_dir in args.run_dirs:
        print(f"\n{run_dir}")
        rows, summary = diagnose_run(os.path.join(REPO_ROOT, run_dir), device)
        feature_rows += rows
        summaries.append(summary)

    write_csv(os.path.join(out_dir, "per_feature.csv"), feature_rows)
    write_csv(os.path.join(out_dir, "summary.csv"), summaries)
    commit, dirty = git_commit()
    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_uncommitted_changes": dirty,
        "run_dirs": args.run_dirs,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
