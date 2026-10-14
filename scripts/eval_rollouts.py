"""Roll out a model on one split of the dataset and save the per step RMS error curves.

Uses every usable window (H = 100, stride 1) of every run in the split; see src/eval/rollout.py.
Also saves and prints robust statistics: the median and 90th percentile over windows of the per
window group error, and the fraction of windows that have left the train range (train_range).

Usage: python scripts/eval_rollouts.py --model constant_velocity [--split tuning]
       python scripts/eval_rollouts.py --model mlp --run_dir results/train/<run folder> [--split tuning]
Output: results/rollouts/<YYYY-MM-DD_HHMMSS>_<model>_<split>/ with metrics.npz and meta.json
        (for mlp: <YYYY-MM-DD_HHMMSS>_mlp_<run folder name>_<split>/).
"""

import argparse
import datetime
import json
import os
import resource
import sys

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from build_dataset import git_commit  # noqa: E402
from eval.rollout import evaluate, train_range, SUBSETS  # noqa: E402
from features import STATE_NAMES, STATE_GROUPS  # noqa: E402
from models import constant_velocity  # noqa: E402
from models.mlp import make_mlp, make_step  # noqa: E402

DATASET = os.path.join("data", "processed", "2026-10-07_182522", "dataset.npz")
MODELS = {"constant_velocity": constant_velocity.step}
H = 100
REPORT_STEPS = [1, 10, 50, 100]


def load_mlp_step(run_dir, data, device):
    """step(states, actions) of the MLP saved by src/train.py in run_dir, normalized with data's stats.

    Raises ValueError if the run was trained on a different dataset than DATASET.
    """
    with open(os.path.join(run_dir, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    if os.path.normpath(cfg["dataset"]) != os.path.normpath(DATASET):
        raise ValueError(f"{run_dir} was trained on {cfg['dataset']}, but this script evaluates {DATASET}")
    saved = torch.load(os.path.join(run_dir, "model.pt"), map_location=device)
    in_dim = len(data["state_mean"]) + len(data["action_mean"])  # 34 + 12
    model = make_mlp(in_dim, len(data["delta_mean"]), saved["width"], saved["depth"]).to(device)
    model.load_state_dict(saved["state_dict"])
    return make_step(model, data, device)


def report(metrics):
    print(f"windows: {metrics['n_windows']}")
    header = f"{'':12s} {'subset':8s}" + "".join(f"{'j = ' + str(j):>12s}" for j in REPORT_STEPS)
    print(header)
    for subset in SUBSETS:
        counts = "".join(f"{metrics['count_' + subset][j - 1]:12d}" for j in REPORT_STEPS)
        print(f"{'samples':12s} {subset:8s}{counts}")
    for group in STATE_GROUPS:
        for subset in SUBSETS:
            rms = metrics[f"rms_{subset}_{group}"]
            values = "".join(f"{rms[j - 1]:12.4e}" for j in REPORT_STEPS)
            print(f"{group if subset == 'all' else '':12s} {subset:8s}{values}")


def report_robust(metrics):
    print()
    print("per window group error (RMS over the group's features in one window), over windows: median and 90th pct")
    print(f"{'':12s} {'subset':8s} {'stat':6s}" + "".join(f"{'j = ' + str(j):>12s}" for j in REPORT_STEPS))
    for group in STATE_GROUPS:
        for subset in SUBSETS:
            for stat in ["median", "p90"]:
                values = "".join(f"{metrics[f'{stat}_{subset}_{group}'][j - 1]:12.4e}" for j in REPORT_STEPS)
                label = group if subset == "all" and stat == "median" else ""
                print(f"{label:12s} {subset if stat == 'median' else '':8s} {stat:6s}{values}")
    print()
    print("diverged fraction: windows where some predicted feature except gravity_z has left the train range"
          " (or is not finite) at a step <= j")
    print(f"{'diverged':12s} {'all':8s} {'frac':6s}" + "".join(f"{metrics['diverged_frac'][j - 1]:12.4f}" for j in REPORT_STEPS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODELS) + ["mlp"])
    parser.add_argument("--run_dir", help="results/train/<run folder>, required with --model mlp")
    parser.add_argument("--split", default="tuning", choices=["train", "tuning"])
    args = parser.parse_args()
    if args.model == "mlp" and args.run_dir is None:
        parser.error("--run_dir is required with --model mlp")

    data = dict(np.load(os.path.join(REPO_ROOT, DATASET)))
    if args.model == "mlp":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}")
        step = load_mlp_step(os.path.join(REPO_ROOT, args.run_dir), data, device)
        name = f"mlp_{os.path.basename(os.path.normpath(args.run_dir))}"
    else:
        step = MODELS[args.model]
        name = args.model
    metrics = evaluate(step, data, args.split, H, bounds=train_range(data))
    print(f"model {args.model}, split {args.split}, H = {H}, RMS error in raw units")
    report(metrics)
    report_robust(metrics)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "results", "rollouts", f"{stamp}_{name}_{args.split}")
    os.makedirs(out_dir)  # raises if the folder already exists: never overwrite
    np.savez(os.path.join(out_dir, "metrics.npz"), **metrics)
    commit, dirty = git_commit()
    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": DATASET,
        "git_commit": commit,
        "git_uncommitted_changes": dirty,
        "model": args.model,
        "split": args.split,
        "H": H,
        "n_windows": int(metrics["n_windows"]),
        "state_names": STATE_NAMES,  # columns of rms_<subset>
    }
    if args.model == "mlp":
        meta["run_dir"] = args.run_dir
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # ru_maxrss is in kB on Linux
    print(f"\nwrote {out_dir}")
    print(f"peak memory {peak_mb:.0f} MB")


if __name__ == "__main__":
    main()
