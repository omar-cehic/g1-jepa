"""Train the MLP world model on the train runs and report the loss on the tuning runs.

Training pairs: every valid transition k -> k+1 of the train runs.
  input   model_input(s_k, a_k)                     (src/models/mlp.py)
  target  normalize(s_{k+1} - s_k, "delta")
Loss: mean squared error on the normalized delta, averaged over batch and features.
The test split is never used.

AdamW, lr: linear warmup from 0 over warmup_epochs, then cosine decay to 0 at the end,
updated every step. Gradient norm clipped at grad_clip every step.
The saved model is the one after the last epoch (no best checkpoint).

Usage: python src/train.py configs/train_mlp.yaml [--seed S] [--width W] [--lr LR] [--epochs E]
Output: results/train/<YYYY-MM-DD_HHMMSS>_mlp_w<width>_lr<lr>_s<seed>/ with
        config.yaml, model.pt, log.csv, meta.json.
        On a full state dataset (state_type "full" in its meta.json) the folder is
        <YYYY-MM-DD_HHMMSS>_mlp_full_w<width>_lr<lr>_s<seed>/.
"""

import argparse
import datetime
import json
import math
import os
import sys
import time

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from build_dataset import git_commit  # noqa: E402
from models.mlp import make_mlp, normalize, model_input  # noqa: E402

INT_KEYS = ["width", "depth", "batch_size", "epochs", "warmup_epochs", "seed"]
FLOAT_KEYS = ["lr", "weight_decay", "grad_clip"]
LOG_COLUMNS = ["epoch", "train_loss", "tuning_loss", "lr_end", "n_clipped", "seconds"]


def check_config(cfg):
    """Fail early if a value has the wrong type (e.g. 1e-3 read as a string)."""
    assert isinstance(cfg["dataset"], str), cfg["dataset"]
    for key in INT_KEYS:
        assert type(cfg[key]) is int, f"{key} = {cfg[key]!r} is not an int"
    for key in FLOAT_KEYS:
        assert type(cfg[key]) is float, f"{key} = {cfg[key]!r} is not a float"
    assert cfg["epochs"] > cfg["warmup_epochs"], "need at least one epoch after the warmup"


def dataset_state_type(dataset):
    """"base" or "full", from the meta.json next to dataset (a dataset.npz path relative to the repo).

    Datasets built before state_type existed have no such key: they are "base".
    """
    with open(os.path.join(REPO_ROOT, os.path.dirname(dataset), "meta.json"), "r") as f:
        return json.load(f).get("state_type", "base")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def transitions(data, split):
    """Inputs (n, state_dim + 12) and targets (n, state_dim), float32, from the valid transitions of one split."""
    runs = data["split"] == split
    states, valid = data["states"][runs], data["valid"][runs]
    s = states[:, :-1][valid]                          # s_k        (n, state_dim) float64
    delta = (states[:, 1:] - states[:, :-1])[valid]    # s_{k+1} - s_k
    a = data["actions"][runs][valid]                   # a_k        (n, 12)
    x = model_input(s, a, data)
    y = normalize(delta, data, "delta")
    return x.astype(np.float32), y.astype(np.float32)


def lr_factor(step, warmup_steps, total_steps):
    """Multiplier of lr at step: 0 -> 1 linearly over warmup_steps, then cosine 1 -> 0 at total_steps."""
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_step(model, optimizer, x, y, grad_clip):
    """One AdamW step on one batch. Returns the loss before the step and whether clipping was active."""
    loss = torch.nn.functional.mse_loss(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # norm before clipping
    optimizer.step()
    return loss.item(), grad_norm.item() > grad_clip


def train(cfg, x_train, y_train, x_tune, y_tune, device):
    """Train a fresh MLP. Tensors must already be on device. Returns the final model and the epoch log."""
    set_seed(cfg["seed"])
    model = make_mlp(x_train.shape[1], y_train.shape[1], cfg["width"], cfg["depth"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n, batch_size = len(x_train), cfg["batch_size"]
    steps_per_epoch = math.ceil(n / batch_size)       # the last partial batch is kept
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch
    total_steps = cfg["epochs"] * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, warmup_steps, total_steps))
    generator = torch.Generator().manual_seed(cfg["seed"])  # on the CPU: same order on any device

    log = []
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        order = torch.randperm(n, generator=generator).to(device)
        loss_sum, n_clipped = 0.0, 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            loss, clipped = train_step(model, optimizer, x_train[idx], y_train[idx], cfg["grad_clip"])
            scheduler.step()
            loss_sum += loss * len(idx)                # weight by batch size: the last batch is smaller
            n_clipped += clipped
        with torch.no_grad():
            tuning_loss = torch.nn.functional.mse_loss(model(x_tune), y_tune).item()
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / n,
            "tuning_loss": tuning_loss,
            "lr_end": scheduler.get_last_lr()[0],      # lr the next step would use
            "n_clipped": n_clipped,
            "seconds": time.time() - t0,
        }
        log.append(row)
        print(f"epoch {epoch:3d}/{cfg['epochs']}  train {row['train_loss']:.5f}  "
              f"tuning {row['tuning_loss']:.5f}  lr_end {row['lr_end']:.2e}  "
              f"clipped {n_clipped}/{steps_per_epoch}  {row['seconds']:.1f} s")
    return model, log


def main():
    t_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to train_mlp.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    for key in ["seed", "width", "lr", "epochs"]:
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    check_config(cfg)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    full = "_full" if dataset_state_type(cfg["dataset"]) == "full" else ""
    name = f"{stamp}_mlp{full}_w{cfg['width']}_lr{cfg['lr']:g}_s{cfg['seed']}"
    out_dir = os.path.join(REPO_ROOT, "results", "train", name)
    os.makedirs(out_dir)  # raises if the folder already exists: never overwrite
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))

    data = dict(np.load(os.path.join(REPO_ROOT, cfg["dataset"])))
    x_train, y_train = transitions(data, "train")
    x_tune, y_tune = transitions(data, "tuning")
    print(f"train pairs {len(x_train)}, tuning pairs {len(x_tune)}")
    x_train, y_train, x_tune, y_tune = [torch.from_numpy(a).to(device) for a in [x_train, y_train, x_tune, y_tune]]

    model, log = train(cfg, x_train, y_train, x_tune, y_tune, device)

    n_params = sum(p.numel() for p in model.parameters())
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state_dict, "width": cfg["width"], "depth": cfg["depth"],
                "state_dim": y_train.shape[1]},
               os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "log.csv"), "w") as f:
        f.write(",".join(LOG_COLUMNS) + "\n")
        for row in log:
            f.write(",".join(str(row[c]) for c in LOG_COLUMNS) + "\n")
    commit, dirty = git_commit()
    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_uncommitted_changes": dirty,
        "n_parameters": n_params,
        "device": str(device),
        "torch_version": torch.__version__,
        "total_seconds": time.time() - t_start,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nparameters {n_params}, total {meta['total_seconds']:.1f} s")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
