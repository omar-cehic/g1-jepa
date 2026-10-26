"""Two readout control checks. No world model is trained here.

All errors are squared errors in normalized state units (normalize(x, data, "state") with the train
state stats in dataset.npz): the mean over pairs and the 34 features ("all"), plus the mean over the
features of each STATE_GROUPS group.
  (a) Random encoder control. For embed_dim 64, 128 and 256: set_seed(0), build
      JEPA(embed_dim, 1024, 3) and do NOT train it. Encode the normalized s_k of the train and
      tuning pairs (pairs() from train_jepa.py). Untrained embeddings are very small, so each
      embedding number is standardized with its mean and std over the train embeddings (the std
      before standardizing is printed). Fit the linear probe and the default MLP probe (seed 0)
      encode(s_k) -> s_k on train; report the present error (probe(z) - s_k)^2 on train and tuning.
  (b) MLP world model in the same units. For each of the 5 main MLP runs (MLP_RUNS): predict
      s_{k+1} from the true s_k and a_k on the valid transitions with make_step (as in
      scripts/eval_rollouts.py), normalize the prediction and the true s_{k+1} with the train state
      stats, and report the one step error on train and tuning. Each seed, and the mean and the
      sample std (ddof = 1, as in aggregate_seeds.py) over the 5 seeds.
Nothing is selected here.

Usage: python scripts/readout_controls.py
Output: results/<YYYY-MM-DD_HHMMSS>_readout_controls/ with summary.csv and meta.json.
"""

import datetime
import json
import os
import sys

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from build_dataset import git_commit  # noqa: E402
from eval.probe import fit_linear_probe, apply_probe, fit_mlp_probe, apply_mlp_probe  # noqa: E402
from eval_rollouts import load_mlp_step, DATASET  # noqa: E402
from features import STATE_GROUPS  # noqa: E402
from models.jepa import JEPA  # noqa: E402
from models.mlp import normalize  # noqa: E402
from probe_diagnostics import per_feature_mse, group_means  # noqa: E402
from score_jepa import embed  # noqa: E402
from train import set_seed  # noqa: E402
from train_jepa import pairs  # noqa: E402

EMBED_DIMS = [64, 128, 256]
WIDTH = 1024
DEPTH = 3
# The 5 main MLP runs (width 1024, lr 0.001, 100 epochs), seed i at index i. Seed 0 is the run
# from the locked config batch; 2026-10-14_123253_mlp_w1024_lr0.001_s0 has identical weights.
MLP_RUNS = [
    "results/train/2026-10-14_143647_mlp_w1024_lr0.001_s0",
    "results/train/2026-10-14_144118_mlp_w1024_lr0.001_s1",
    "results/train/2026-10-14_144624_mlp_w1024_lr0.001_s2",
    "results/train/2026-10-14_145138_mlp_w1024_lr0.001_s3",
    "results/train/2026-10-14_145654_mlp_w1024_lr0.001_s4",
]
COLUMNS = ["part", "model", "probe", "split", "all"] + list(STATE_GROUPS)


def error_row(part, model, probe, split, feature_mse):
    """One row of summary.csv: the mean over all 34 features and the mean of each group."""
    return {"part": part, "model": model, "probe": probe, "split": split,
            "all": float(feature_mse.mean()), **group_means(feature_mse)}


def standardize(z_train, z_tune):
    """(z - mean) / std per embedding number, with the mean and std over z_train only, in float64."""
    mean, std = z_train.mean(axis=0), z_train.std(axis=0)
    return (z_train - mean) / std, (z_tune - mean) / std


def one_step_mse(step, data, split):
    """Per feature (34,) mean over the valid transitions of split of the one step squared error.

    step(s_k, a_k) gets raw states and actions (as in src/eval/rollout.py) and returns the
    predicted raw s_{k+1}; the prediction and the true s_{k+1} are normalized with the train state
    stats before the error is taken. All transitions go through step in one call.
    """
    runs = data["split"] == split
    states, valid = data["states"][runs], data["valid"][runs]
    s = states[:, :-1][valid]                  # s_k      (n, 34) raw
    s_next = states[:, 1:][valid]              # s_{k+1}  (n, 34) raw
    a = data["actions"][runs][valid]           # a_k      (n, 12) raw
    pred = step(s, a)
    return per_feature_mse(normalize(pred, data, "state"), normalize(s_next, data, "state"))


def random_encoder_rows(data, device):
    """(a) Rows for the untrained JEPA encoders, both probes, train and tuning."""
    s_train, a_train, _ = pairs(data, "train")
    s_tune, a_tune, _ = pairs(data, "tuning")
    s_train64, s_tune64 = s_train.astype(np.float64), s_tune.astype(np.float64)
    rows = []
    for embed_dim in EMBED_DIMS:
        set_seed(0)
        model = JEPA(embed_dim, WIDTH, DEPTH).to(device)
        model.eval()
        z_train, _ = embed(model, s_train, a_train, device)
        z_tune, _ = embed(model, s_tune, a_tune, device)
        std = z_train.std(axis=0)
        print(f"  d {embed_dim}: std over train states of each embedding number before standardizing: "
              f"mean {std.mean():.4f}, min {std.min():.4f}, max {std.max():.4f}")
        z_train, z_tune = standardize(z_train, z_tune)
        W, b = fit_linear_probe(z_train, s_train64)
        mlp_probe = fit_mlp_probe(z_train, s_train, device, seed=0)
        name = f"random_jepa_d{embed_dim}"
        for split, z, s in [("train", z_train, s_train64), ("tuning", z_tune, s_tune64)]:
            rows.append(error_row("a", name, "linear", split, per_feature_mse(apply_probe(z, W, b), s)))
            rows.append(error_row("a", name, "mlp", split,
                                  per_feature_mse(apply_mlp_probe(mlp_probe, z, device), s)))
    return rows


def mlp_world_model_rows(data, device):
    """(b) Rows for each MLP run, then the mean and std over the seeds, train and tuning."""
    rows = []
    for seed, run_dir in enumerate(MLP_RUNS):
        with open(os.path.join(REPO_ROOT, run_dir, "config.yaml"), "r") as f:
            cfg = yaml.safe_load(f)
        assert (cfg["width"], cfg["depth"], cfg["lr"], cfg["epochs"], cfg["seed"]) == (WIDTH, DEPTH, 0.001, 100, seed), \
            (run_dir, cfg)
        step = load_mlp_step(os.path.join(REPO_ROOT, run_dir), data, device)  # checks the dataset
        for split in ["train", "tuning"]:
            rows.append(error_row("b", os.path.basename(run_dir), "none", split, one_step_mse(step, data, split)))
    for split in ["train", "tuning"]:
        seed_rows = [r for r in rows if r["split"] == split]
        mean_row = {"part": "b", "model": "mlp_mean_over_seeds", "probe": "none", "split": split}
        std_row = {"part": "b", "model": "mlp_std_over_seeds", "probe": "none", "split": split}
        for c in COLUMNS[4:]:
            values = [r[c] for r in seed_rows]
            mean_row[c] = float(np.mean(values))
            std_row[c] = float(np.std(values, ddof=1))
        rows += [mean_row, std_row]
    return rows


def print_table(rows):
    print(f"{'part':4s} {'model':42s} {'probe':6s} {'split':6s}" + "".join(f"{c:>11s}" for c in COLUMNS[4:]))
    for r in rows:
        print(f"{r['part']:4s} {r['model']:42s} {r['probe']:6s} {r['split']:6s}"
              + "".join(f"{r[c]:11.4e}" for c in COLUMNS[4:]))


def main():
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "results", f"{stamp}_readout_controls")
    os.makedirs(out_dir)  # raises if the folder already exists: never overwrite

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))
    print("MLP world model runs for (b):")
    for run_dir in MLP_RUNS:
        print(f"  {run_dir}")

    data = dict(np.load(os.path.join(REPO_ROOT, DATASET)))
    print("\n(a) random encoder control")
    rows = random_encoder_rows(data, device)
    print("(b) MLP world model one step error")
    rows += mlp_world_model_rows(data, device)

    with open(os.path.join(out_dir, "summary.csv"), "w") as f:
        f.write(",".join(COLUMNS) + "\n")
        for row in rows:
            f.write(",".join(str(row[c]) for c in COLUMNS) + "\n")
    commit, dirty = git_commit()
    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_uncommitted_changes": dirty,
        "dataset": DATASET,
        "run_dirs": MLP_RUNS,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nsquared error in normalized state units, mean over pairs and features (all) and per group")
    print_table(rows)
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
