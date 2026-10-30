"""Train the JEPA world model on the train runs and report losses on the tuning runs.

Training pairs: every valid transition k -> k+1 of the train runs, as
  normalized s_k (state_dim), normalized a_k (12), normalized s_{k+1} (state_dim)
with the train stats in dataset.npz (state_dim 34 for the base state, 37 for the full state,
35 for the full state without pelvis x and y).
The test split is never used.

One step on a batch of B pairs:
  z = encode(s_k), z_next = encode(s_{k+1}), z_hat = predict(z, a_k)
  pred_loss = mean over batch and embedding numbers of (z_hat - z_next)^2
  sigreg    = SIGReg()(stack([z, z_next]))     input shape (2, B, embed_dim)
  loss      = pred_loss + sigreg_weight * sigreg
Gradients flow through both z and z_next: nothing is detached, there is no EMA target encoder.
With --stop_grad (stop_grad: true in the config; off by default, and off when the key is missing)
the target z_next in pred_loss is detached, so pred_loss sends no gradient through encode(s_{k+1}).
SIGReg still gets the undetached z and z_next, and the loss value is the same as without it.
SIGReg multiplies its value by the batch size B, so on the last partial batch of each epoch
(226,099 = 441 * 512 + 307 pairs) it is multiplied by 307 instead of 512. This is kept on purpose.

Everything else is the same as src/train.py (MLP): AdamW on all parameters, lr warmup + cosine
per step, gradient norm clipped at grad_clip every step, batch_size with the last partial batch
kept, shuffling with a seeded CPU generator. The saved model is the one after the last epoch.

Usage: python src/train_jepa.py configs/train_jepa.yaml [--seed S] [--embed_dim D]
                                [--sigreg_weight W] [--lr LR] [--epochs E] [--stop_grad]
Output: results/train/<YYYY-MM-DD_HHMMSS>_jepa_d<embed_dim>_sw<sigreg_weight>_lr<lr>_s<seed>/
        with config.yaml, model.pt, log.csv, meta.json.
        With stop_grad on, "_sg" goes before "_s<seed>" (..._lr0.001_sg_s0).
        On a full state dataset (state_type "full" in its meta.json) "jepa" becomes "jepa_full",
        and "jepa_fullnoxy" for state_type "full_no_xy".
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
from models.mlp import normalize  # noqa: E402
from models.jepa import JEPA  # noqa: E402
from models.sigreg import SIGReg  # noqa: E402
from train import check_config, set_seed, lr_factor, dataset_state_type  # noqa: E402

LOG_COLUMNS = ["epoch", "train_pred_loss", "train_sigreg", "train_loss", "tuning_pred_loss",
               "tuning_sigreg", "tuning_emb_std", "lr_end", "n_clipped", "seconds"]


def check_jepa_config(cfg):
    """check_config from train.py, plus the three JEPA values."""
    check_config(cfg)
    assert type(cfg["embed_dim"]) is int, f"embed_dim = {cfg['embed_dim']!r} is not an int"
    assert type(cfg["sigreg_weight"]) is float, f"sigreg_weight = {cfg['sigreg_weight']!r} is not a float"
    assert type(cfg["stop_grad"]) is bool, f"stop_grad = {cfg['stop_grad']!r} is not true or false"


def pairs(data, split):
    """Normalized s_k (n, state_dim), a_k (n, 12), s_{k+1} (n, state_dim), float32, of the valid transitions of one split."""
    runs = data["split"] == split
    states, valid = data["states"][runs], data["valid"][runs]
    s = normalize(states[:, :-1][valid], data, "state")
    s_next = normalize(states[:, 1:][valid], data, "state")
    a = normalize(data["actions"][runs][valid], data, "action")
    return s.astype(np.float32), a.astype(np.float32), s_next.astype(np.float32)


def jepa_losses(model, sigreg, s, a, s_next, stop_grad=False):
    """pred_loss and SIGReg value (both tensors with gradients) on one batch.

    With stop_grad, the target z_next is detached in pred_loss only; SIGReg gets the undetached z_next.
    """
    z = model.encode(s)
    z_next = model.encode(s_next)
    z_hat = model.predict(z, a)
    if stop_grad:
        pred_loss = (z_hat - z_next.detach()).square().mean()
    else:
        pred_loss = (z_hat - z_next).square().mean()
    return pred_loss, sigreg(torch.stack([z, z_next]))


def train_step(model, sigreg, optimizer, s, a, s_next, sigreg_weight, grad_clip, stop_grad=False):
    """One AdamW step on one batch.

    Returns pred_loss, sigreg and total loss before the step, and whether clipping was active.
    """
    pred_loss, sig = jepa_losses(model, sigreg, s, a, s_next, stop_grad)
    loss = pred_loss + sigreg_weight * sig
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # norm before clipping
    optimizer.step()
    return pred_loss.item(), sig.item(), loss.item(), grad_norm.item() > grad_clip


def evaluate(model, sigreg, s, a, s_next, batch_size, tuning_order):
    """tuning_pred_loss, tuning_sigreg and tuning_emb_std on the full tuning set (no gradients).

    tuning_sigreg is the mean over the full batches of batch_size taken in tuning_order (a fixed
    permutation, the same every epoch); the last partial batch is skipped so every value is
    multiplied by the same batch size and is comparable to the ~1.05 of a Gaussian batch.
    tuning_emb_std is the std over all tuning states of each embedding number, then the mean.
    """
    with torch.no_grad():
        z = model.encode(s)
        pred_loss = (model.predict(z, a) - model.encode(s_next)).square().mean().item()
        emb_std = z.std(dim=0).mean().item()
        values = []
        for start in range(0, len(s) - batch_size + 1, batch_size):
            idx = tuning_order[start:start + batch_size]
            values.append(sigreg(torch.stack([model.encode(s[idx]), model.encode(s_next[idx])])).item())
    return pred_loss, sum(values) / len(values), emb_std


def train(cfg, train_pairs, tuning_pairs, device):
    """Train a fresh JEPA. Pairs are (s, a, s_next) tensors already on device.

    A cfg without a stop_grad key trains with stop_grad off. Returns the final model and the epoch log.
    """
    s_train, a_train, n_train = train_pairs
    set_seed(cfg["seed"])
    model = JEPA(cfg["embed_dim"], cfg["width"], cfg["depth"], state_dim=s_train.shape[1]).to(device)
    sigreg = SIGReg().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n, batch_size = len(s_train), cfg["batch_size"]
    steps_per_epoch = math.ceil(n / batch_size)       # the last partial batch is kept
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch
    total_steps = cfg["epochs"] * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, warmup_steps, total_steps))
    generator = torch.Generator().manual_seed(cfg["seed"])  # on the CPU: same order on any device
    # fixed shuffled order of the tuning pairs for tuning_sigreg: the pairs are stored run by run,
    # and 512 consecutive steps of one run are not a sample of the whole tuning set
    tuning_order = torch.randperm(len(tuning_pairs[0]), generator=torch.Generator().manual_seed(0)).to(device)

    log = []
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        order = torch.randperm(n, generator=generator).to(device)
        pred_sum, sig_sum, loss_sum, n_clipped = 0.0, 0.0, 0.0, 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            pred_loss, sig, loss, clipped = train_step(
                model, sigreg, optimizer, s_train[idx], a_train[idx], n_train[idx],
                cfg["sigreg_weight"], cfg["grad_clip"], cfg.get("stop_grad", False))
            scheduler.step()
            pred_sum += pred_loss * len(idx)           # weight by batch size: the last batch is smaller
            sig_sum += sig * len(idx)
            loss_sum += loss * len(idx)
            n_clipped += clipped
        tuning_pred_loss, tuning_sigreg, tuning_emb_std = evaluate(
            model, sigreg, *tuning_pairs, batch_size, tuning_order)
        row = {
            "epoch": epoch,
            "train_pred_loss": pred_sum / n,
            "train_sigreg": sig_sum / n,
            "train_loss": loss_sum / n,
            "tuning_pred_loss": tuning_pred_loss,
            "tuning_sigreg": tuning_sigreg,
            "tuning_emb_std": tuning_emb_std,
            "lr_end": scheduler.get_last_lr()[0],      # lr the next step would use
            "n_clipped": n_clipped,
            "seconds": time.time() - t0,
        }
        log.append(row)
        print(f"epoch {epoch:3d}/{cfg['epochs']}  train pred {row['train_pred_loss']:.5f} "
              f"sigreg {row['train_sigreg']:.3f} loss {row['train_loss']:.5f}  "
              f"tuning pred {tuning_pred_loss:.5f} sigreg {tuning_sigreg:.3f} emb_std {tuning_emb_std:.3f}  "
              f"lr_end {row['lr_end']:.2e}  clipped {n_clipped}/{steps_per_epoch}  {row['seconds']:.1f} s")
    return model, log


def main():
    t_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to train_jepa.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--embed_dim", type=int)
    parser.add_argument("--sigreg_weight", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--stop_grad", action="store_true", default=None)  # None when not given
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("stop_grad", False)  # configs without the key: off (saved in config.yaml)
    for key in ["seed", "embed_dim", "sigreg_weight", "lr", "epochs", "stop_grad"]:
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    check_jepa_config(cfg)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    full = {"base": "", "full": "_full", "full_no_xy": "_fullnoxy"}[dataset_state_type(cfg["dataset"])]
    sg = "_sg" if cfg["stop_grad"] else ""
    name = (f"{stamp}_jepa{full}_d{cfg['embed_dim']}_sw{cfg['sigreg_weight']:g}"
            f"_lr{cfg['lr']:g}{sg}_s{cfg['seed']}")
    out_dir = os.path.join(REPO_ROOT, "results", "train", name)
    os.makedirs(out_dir)  # raises if the folder already exists: never overwrite
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))

    data = dict(np.load(os.path.join(REPO_ROOT, cfg["dataset"])))
    train_pairs = [torch.from_numpy(x).to(device) for x in pairs(data, "train")]
    tuning_pairs = [torch.from_numpy(x).to(device) for x in pairs(data, "tuning")]
    print(f"train pairs {len(train_pairs[0])}, tuning pairs {len(tuning_pairs[0])}")

    model, log = train(cfg, train_pairs, tuning_pairs, device)

    n_params = sum(p.numel() for p in model.parameters())
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state_dict, "embed_dim": cfg["embed_dim"],
                "width": cfg["width"], "depth": cfg["depth"], "state_dim": train_pairs[0].shape[1]},
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
        "n_encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
        "n_predictor_parameters": sum(p.numel() for p in model.predictor.parameters()),
        "device": str(device),
        "torch_version": torch.__version__,
        "total_seconds": time.time() - t_start,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nparameters {n_params} (encoder {meta['n_encoder_parameters']}, "
          f"predictor {meta['n_predictor_parameters']}), total {meta['total_seconds']:.1f} s")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
