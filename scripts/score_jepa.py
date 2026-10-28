"""Score trained JEPA runs with a linear probe and an MLP probe from embeddings back to states.

For each run folder (written by src/train_jepa.py):
  1. Rebuild the JEPA from model.pt and load the dataset named in config.yaml.
  2. Build the train and tuning pairs (normalized s_k, a_k, s_{k+1}) with pairs() from train_jepa.py.
  3. Fit a linear probe and an MLP probe encode(s_k) -> s_k on the train pairs (src/eval/probe.py).
     The embeddings are computed once by the frozen JEPA; the JEPA is never updated.
     The tuning split is never used for fitting.
  4. On the tuning pairs, each a mean over pairs and the state features (34 base, 37 full),
     in normalized state units:
       score           = (probe(predict(encode(s_k), a_k)) - s_{k+1})^2
       present_mse     = (probe(encode(s_k)) - s_k)^2
       mlp_score       = (mlp_probe(predict(encode(s_k), a_k)) - s_{k+1})^2
       mlp_present_mse = (mlp_probe(encode(s_k)) - s_k)^2
       no_change_mse   = (s_k - s_{k+1})^2      (the same for every run; context only)
  5. From the last row of log.csv: tuning_sigreg, tuning_emb_std, n_clipped; and the number of epochs.
The network runs in eval mode with no gradients, in float32 batches on the GPU (if there is one);
the linear probe and the scores are computed in float64. The MLP probe is trained in float32 on
the same device with seed 0. Nothing is selected here.

Usage: python scripts/score_jepa.py results/train/<run folder> [results/train/<run folder> ...]
Output: results/<YYYY-MM-DD_HHMMSS>_score_jepa/ with scores.csv and meta.json.
"""

import argparse
import csv
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
from models.jepa import JEPA  # noqa: E402
from train_jepa import pairs  # noqa: E402

BATCH_SIZE = 16384   # pairs per forward pass; only affects speed and GPU memory
COLUMNS = ["folder", "embed_dim", "sigreg_weight", "lr", "seed", "epochs", "score", "present_mse",
           "mlp_score", "mlp_present_mse", "no_change_mse", "tuning_sigreg", "tuning_emb_std",
           "n_clipped"]


def load_jepa(run_dir, device):
    """The JEPA saved by src/train_jepa.py in run_dir, in eval mode on device.

    Checkpoints saved before state_dim was stored are base state runs (34).
    """
    saved = torch.load(os.path.join(run_dir, "model.pt"), map_location=device)
    model = JEPA(saved["embed_dim"], saved["width"], saved["depth"],
                 state_dim=saved.get("state_dim", 34)).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model


def embed(model, s, a, device):
    """z = encode(s) and z_hat = predict(z, a) as float64 numpy arrays, computed in batches."""
    z_parts, z_hat_parts = [], []
    with torch.no_grad():
        for start in range(0, len(s), BATCH_SIZE):
            s_batch = torch.from_numpy(s[start:start + BATCH_SIZE]).to(device)
            a_batch = torch.from_numpy(a[start:start + BATCH_SIZE]).to(device)
            z = model.encode(s_batch)
            z_hat = model.predict(z, a_batch)
            z_parts.append(z.cpu().numpy().astype(np.float64))
            z_hat_parts.append(z_hat.cpu().numpy().astype(np.float64))
    return np.concatenate(z_parts), np.concatenate(z_hat_parts)


def read_log(run_dir):
    """Rows of log.csv as a list of dicts of strings."""
    with open(os.path.join(run_dir, "log.csv"), "r") as f:
        return list(csv.DictReader(f))


def score_run(run_dir, device):
    """One row of scores.csv (a dict with COLUMNS as keys) for the run in run_dir."""
    with open(os.path.join(run_dir, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    data = dict(np.load(os.path.join(REPO_ROOT, cfg["dataset"])))
    model = load_jepa(run_dir, device)

    s_train, a_train, _ = pairs(data, "train")
    z_train, _ = embed(model, s_train, a_train, device)
    W, b = fit_linear_probe(z_train, s_train.astype(np.float64))
    mlp_probe = fit_mlp_probe(z_train, s_train, device, seed=0)
    del z_train  # free memory before the tuning pass

    s, a, s_next = pairs(data, "tuning")
    z, z_hat = embed(model, s, a, device)
    s, s_next = s.astype(np.float64), s_next.astype(np.float64)
    log = read_log(run_dir)
    return {
        "folder": os.path.basename(os.path.normpath(run_dir)),
        "embed_dim": cfg["embed_dim"],
        "sigreg_weight": cfg["sigreg_weight"],
        "lr": cfg["lr"],
        "seed": cfg["seed"],
        "epochs": len(log),
        "score": float(np.mean((apply_probe(z_hat, W, b) - s_next) ** 2)),
        "present_mse": float(np.mean((apply_probe(z, W, b) - s) ** 2)),
        "mlp_score": float(np.mean((apply_mlp_probe(mlp_probe, z_hat, device) - s_next) ** 2)),
        "mlp_present_mse": float(np.mean((apply_mlp_probe(mlp_probe, z, device) - s) ** 2)),
        "no_change_mse": float(np.mean((s - s_next) ** 2)),
        "tuning_sigreg": float(log[-1]["tuning_sigreg"]),
        "tuning_emb_std": float(log[-1]["tuning_emb_std"]),
        "n_clipped": int(log[-1]["n_clipped"]),
    }


def print_table(rows):
    """The rows sorted by mlp_score (lowest first)."""
    print(f"{'folder':50s} {'d':>4s} {'sw':>7s} {'ep':>4s} {'mlp_score':>10s} {'mlp_pres':>10s} "
          f"{'score':>10s} {'present':>10s} {'no_change':>10s} {'sigreg':>8s} {'emb_std':>8s} "
          f"{'clipped':>7s}")
    for r in sorted(rows, key=lambda r: r["mlp_score"]):
        print(f"{r['folder']:50s} {r['embed_dim']:4d} {r['sigreg_weight']:7g} {r['epochs']:4d} "
              f"{r['mlp_score']:10.6f} {r['mlp_present_mse']:10.6f} "
              f"{r['score']:10.6f} {r['present_mse']:10.6f} {r['no_change_mse']:10.6f} "
              f"{r['tuning_sigreg']:8.3f} {r['tuning_emb_std']:8.3f} {r['n_clipped']:7d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="results/train/<run folder> of JEPA runs")
    args = parser.parse_args()

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "results", f"{stamp}_score_jepa")
    os.makedirs(out_dir)  # raises if the folder already exists: never overwrite

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))

    rows = []
    for run_dir in args.run_dirs:
        print(f"scoring {run_dir}")
        rows.append(score_run(os.path.join(REPO_ROOT, run_dir), device))

    with open(os.path.join(out_dir, "scores.csv"), "w") as f:
        f.write(",".join(COLUMNS) + "\n")
        for row in rows:
            f.write(",".join(str(row[c]) for c in COLUMNS) + "\n")
    commit, dirty = git_commit()
    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_uncommitted_changes": dirty,
        "run_dirs": args.run_dirs,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print_table(rows)
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
