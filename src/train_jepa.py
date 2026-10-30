"""Train the JEPA world model on the train runs and report losses on the tuning runs.

Training pairs: every valid transition k -> k+1 of the train runs, as
  normalized s_k (state_dim), normalized a_k (12), normalized s_{k+1} (state_dim)
with the train stats in dataset.npz (state_dim 34 for the base state, 37 for the full state,
35 for the full state without pelvis x and y).
The test split is never used.

State history (history n, config key history, --history; 1 by default and when the key is missing):
the predictor sees the window of the last n embeddings z_{k-n+1}, ..., z_k (oldest first) and a_k.
The window states s_{k-n+1}, ..., s_{k-1} are the stored states of the same run, whether or not the
transitions between them are valid. The pairs are the same for every n (only the valid transitions;
the first is at k = 49, so n is at most 50). The windows are not stored: window_pairs() keeps the
normalized states of the split's runs and the row of each s_k, and gather_windows() builds the
windows of one batch by index. With history 1 the window is s_k alone and everything below is
exactly the computation without history.

One step on a batch of B pairs:
  z = encode(s_k), z_past = encode each of s_{k-n+1}, ..., s_{k-1} (none for history 1)
  z_next = encode(s_{k+1}), z_hat = predict(z, a_k, z_past) = z + predictor([z_past, z, a_k])
  pred_loss = mean over batch and embedding numbers of (z_hat - z_next)^2
  sigreg    = SIGReg()(stack([z, z_next]))     input shape (2, B, embed_dim); z_past is not in it
  loss      = pred_loss + sigreg_weight * sigreg
Gradients flow through every embedding of the window and z_next: nothing is detached, there is no EMA
target encoder.
With --stop_grad (stop_grad: true in the config; off by default, and off when the key is missing)
the target z_next in pred_loss is detached, so pred_loss sends no gradient through encode(s_{k+1}).
SIGReg still gets the undetached z and z_next, and the loss value is the same as without it.
SIGReg multiplies its value by the batch size B, so on the last partial batch of each epoch
(226,099 = 441 * 512 + 307 pairs) it is multiplied by 307 instead of 512. This is kept on purpose.

Everything else is the same as src/train.py (MLP): AdamW on all parameters, lr warmup + cosine
per step, gradient norm clipped at grad_clip every step, batch_size with the last partial batch
kept, shuffling with a seeded CPU generator. The saved model is the one after the last epoch.

Usage: python src/train_jepa.py configs/train_jepa.yaml [--seed S] [--embed_dim D]
                                [--sigreg_weight W] [--lr LR] [--epochs E] [--stop_grad] [--history N]
Output: results/train/<YYYY-MM-DD_HHMMSS>_jepa_d<embed_dim>_sw<sigreg_weight>_lr<lr>_s<seed>/
        with config.yaml, model.pt, log.csv, meta.json.
        With stop_grad on, "_sg" goes before "_s<seed>" (..._lr0.001_sg_s0).
        With history above 1, "_h<history>" goes right before "_s<seed>" (..._lr0.001_h10_s0).
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

MAX_HISTORY = 50     # the first valid pair of a run is at k = 49, so a window of 50 reaches back to s_0
EVAL_STATES = 32768  # states per encoder pass for tuning_pred_loss; one pass at history 1

LOG_COLUMNS = ["epoch", "train_pred_loss", "train_sigreg", "train_loss", "tuning_pred_loss",
               "tuning_sigreg", "tuning_emb_std", "lr_end", "n_clipped", "seconds"]


def check_jepa_config(cfg):
    """check_config from train.py, plus the JEPA values."""
    check_config(cfg)
    assert type(cfg["embed_dim"]) is int, f"embed_dim = {cfg['embed_dim']!r} is not an int"
    assert type(cfg["sigreg_weight"]) is float, f"sigreg_weight = {cfg['sigreg_weight']!r} is not a float"
    assert type(cfg["stop_grad"]) is bool, f"stop_grad = {cfg['stop_grad']!r} is not true or false"
    assert type(cfg["history"]) is int, f"history = {cfg['history']!r} is not an int"
    assert 1 <= cfg["history"] <= MAX_HISTORY, (
        f"history = {cfg['history']} is not between 1 and {MAX_HISTORY}: the first valid pair of a run "
        f"is at k = 49, so a longer window would reach before the start of the run")


def pairs(data, split):
    """Normalized s_k (n, state_dim), a_k (n, 12), s_{k+1} (n, state_dim), float32, of the valid transitions of one split."""
    runs = data["split"] == split
    states, valid = data["states"][runs], data["valid"][runs]
    s = normalize(states[:, :-1][valid], data, "state")
    s_next = normalize(states[:, 1:][valid], data, "state")
    a = normalize(data["actions"][runs][valid], data, "action")
    return s.astype(np.float32), a.astype(np.float32), s_next.astype(np.float32)


def window_pairs(data, split, history):
    """The valid transitions of one split, with what is needed to gather state windows by index.

    Returns, float32 / int64:
      states (n_runs * steps, state_dim)  normalized states of the split's runs, run after run
      rows   (n,)                         row of s_k in states for each valid pair, in the order of pairs()
      a      (n, 12)                      normalized a_k
    s_k is states[rows], s_{k+1} is states[rows + 1], and the window s_{k-history+1}, ..., s_k is
    gather_windows(states, rows, history). Raises if a window would start before the first step of its run.
    """
    runs = data["split"] == split
    states, valid = data["states"][runs], data["valid"][runs]
    n_runs, steps = states.shape[0], states.shape[1]
    run, k = np.nonzero(valid)             # run by run, k ascending: the same order as pairs()
    assert np.all(k - history + 1 >= 0), (
        f"history {history}: the window of the pair at k = {k.min()} would start before step 0 of its run")
    rows = run * steps + k
    states = normalize(states.reshape(n_runs * steps, states.shape[2]), data, "state")
    a = normalize(data["actions"][runs][valid], data, "action")
    return states.astype(np.float32), rows.astype(np.int64), a.astype(np.float32)


def gather_windows(states, rows, history):
    """State windows (B, history, state_dim): row j holds states[rows[j] - history + 1], ..., states[rows[j]].

    states and rows are tensors on the same device (from window_pairs). Oldest state first, s_k last.
    """
    offsets = torch.arange(1 - history, 1, device=rows.device)   # -history+1, ..., 0
    return states[rows[:, None] + offsets]


def encode_past(model, s):
    """Embeddings of the older states s_{k-n+1}, ..., s_{k-1} of state windows s (B, n, state_dim):
    (B, n - 1, embed_dim), oldest first, or None when the window is s_k alone (history 1)."""
    return model.encode_window(s[:, :-1]) if s.shape[1] > 1 else None


def jepa_losses(model, sigreg, s, a, s_next, stop_grad=False):
    """pred_loss and SIGReg value (both tensors with gradients) on one batch.

    s is the state window of each pair (B, history, state_dim), oldest first and s_k last; with
    history 1 it can also be s_k alone (B, state_dim). Every window state is encoded and gets
    gradients; SIGReg gets only z_k and z_{k+1}.
    z_k is encoded on its own (not as part of the window), so with history 1 the computation,
    including the order in which autograd adds up gradients, is exactly the one without history.
    With stop_grad, the target z_next is detached in pred_loss only; SIGReg gets the undetached z_next.
    """
    if s.dim() == 2:                      # (B, state_dim): a window of one state
        s = s.unsqueeze(1)
    z = model.encode(s[:, -1])            # z_k
    z_past = encode_past(model, s)
    z_next = model.encode(s_next)
    z_hat = model.predict(z, a, z_past)
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


def evaluate(model, sigreg, states, rows, a, batch_size, tuning_order):
    """tuning_pred_loss, tuning_sigreg and tuning_emb_std on the full tuning set (no gradients).

    states, rows, a are from window_pairs() (tensors on the device).
    tuning_pred_loss is computed in chunks of EVAL_STATES // history pairs, so the windows of one
    chunk fit in memory; with history 1 the whole tuning set is one chunk.
    tuning_sigreg is the mean over the full batches of batch_size taken in tuning_order (a fixed
    permutation, the same every epoch); the last partial batch is skipped so every value is
    multiplied by the same batch size and is comparable to the ~1.05 of a Gaussian batch.
    tuning_emb_std is the std over all tuning states s_k of each embedding number, then the mean.
    """
    history = model.history
    chunk = EVAL_STATES // history
    with torch.no_grad():
        zs, squared_errors = [], []
        for start in range(0, len(rows), chunk):
            r = rows[start:start + chunk]
            z = model.encode(states[r])
            z_past = encode_past(model, gather_windows(states, r, history))
            z_hat = model.predict(z, a[start:start + chunk], z_past)
            zs.append(z)
            squared_errors.append((z_hat - model.encode(states[r + 1])).square())
        pred_loss = torch.cat(squared_errors).mean().item()
        emb_std = torch.cat(zs).std(dim=0).mean().item()
        values = []
        for start in range(0, len(rows) - batch_size + 1, batch_size):
            r = rows[tuning_order[start:start + batch_size]]
            values.append(sigreg(torch.stack([model.encode(states[r]), model.encode(states[r + 1])])).item())
    return pred_loss, sum(values) / len(values), emb_std


def train(cfg, train_data, tuning_data, device):
    """Train a fresh JEPA. train_data and tuning_data are (states, rows, a) from window_pairs(), as
    tensors already on device.

    A cfg without a stop_grad key trains with stop_grad off, and without a history key with history 1.
    Returns the final model and the epoch log.
    """
    states, rows, a_train = train_data
    history = cfg.get("history", 1)
    set_seed(cfg["seed"])
    model = JEPA(cfg["embed_dim"], cfg["width"], cfg["depth"], state_dim=states.shape[1],
                 history=history).to(device)
    sigreg = SIGReg().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n, batch_size = len(rows), cfg["batch_size"]
    steps_per_epoch = math.ceil(n / batch_size)       # the last partial batch is kept
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch
    total_steps = cfg["epochs"] * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, warmup_steps, total_steps))
    generator = torch.Generator().manual_seed(cfg["seed"])  # on the CPU: same order on any device
    # fixed shuffled order of the tuning pairs for tuning_sigreg: the pairs are stored run by run,
    # and 512 consecutive steps of one run are not a sample of the whole tuning set
    tuning_order = torch.randperm(len(tuning_data[1]), generator=torch.Generator().manual_seed(0)).to(device)

    log = []
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        order = torch.randperm(n, generator=generator).to(device)
        pred_sum, sig_sum, loss_sum, n_clipped = 0.0, 0.0, 0.0, 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            r = rows[idx]
            pred_loss, sig, loss, clipped = train_step(
                model, sigreg, optimizer, gather_windows(states, r, history), a_train[idx], states[r + 1],
                cfg["sigreg_weight"], cfg["grad_clip"], cfg.get("stop_grad", False))
            scheduler.step()
            pred_sum += pred_loss * len(idx)           # weight by batch size: the last batch is smaller
            sig_sum += sig * len(idx)
            loss_sum += loss * len(idx)
            n_clipped += clipped
        tuning_pred_loss, tuning_sigreg, tuning_emb_std = evaluate(
            model, sigreg, *tuning_data, batch_size, tuning_order)
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
    parser.add_argument("--history", type=int)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("stop_grad", False)  # configs without the key: off (saved in config.yaml)
    cfg.setdefault("history", 1)        # configs without the key: 1 (saved in config.yaml)
    for key in ["seed", "embed_dim", "sigreg_weight", "lr", "epochs", "stop_grad", "history"]:
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    check_jepa_config(cfg)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    full = {"base": "", "full": "_full", "full_no_xy": "_fullnoxy"}[dataset_state_type(cfg["dataset"])]
    sg = "_sg" if cfg["stop_grad"] else ""
    h = f"_h{cfg['history']}" if cfg["history"] > 1 else ""
    name = (f"{stamp}_jepa{full}_d{cfg['embed_dim']}_sw{cfg['sigreg_weight']:g}"
            f"_lr{cfg['lr']:g}{sg}{h}_s{cfg['seed']}")
    out_dir = os.path.join(REPO_ROOT, "results", "train", name)
    os.makedirs(out_dir)  # raises if the folder already exists: never overwrite
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))

    data = dict(np.load(os.path.join(REPO_ROOT, cfg["dataset"])))
    train_data = [torch.from_numpy(x).to(device) for x in window_pairs(data, "train", cfg["history"])]
    tuning_data = [torch.from_numpy(x).to(device) for x in window_pairs(data, "tuning", cfg["history"])]
    print(f"train pairs {len(train_data[1])}, tuning pairs {len(tuning_data[1])}, history {cfg['history']}")

    model, log = train(cfg, train_data, tuning_data, device)

    n_params = sum(p.numel() for p in model.parameters())
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state_dict, "embed_dim": cfg["embed_dim"],
                "width": cfg["width"], "depth": cfg["depth"], "state_dim": train_data[0].shape[1],
                "history": cfg["history"]},
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
