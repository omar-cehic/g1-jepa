"""Train the MLP world model on the train runs and report the loss on the tuning runs.

Training pairs: every valid transition k -> k+1 of the train runs.
  input   model_input(s_k, a_k)                     (src/models/mlp.py)
  target  normalize(s_{k+1} - s_k, "delta")
Loss: mean squared error on the normalized delta, averaged over batch and features.
The test split is never used.

Transformer world model (model: transformer in the config, --model transformer; "mlp" by default and
when the key is missing), with history n (history, --history; 1 to 50, 1 by default and when the key
is missing; above 1 only with the transformer). The network is TransformerPredictor from
src/models/jepa.py on states instead of embeddings: for each step j = k-n+1, ..., k of the window
  token_j = Linear([s_j, a_j]) + p_j        (state_dim + 12 -> 256, p_j a learned position vector)
then depth pre-LayerNorm transformer blocks (d_model 256, nhead 4, feedforward width, ReLU, no
dropout, no attention mask), then a LayerNorm and a Linear (256 -> state_dim) on the last token.
s_j and a_j are normalized as for the MLP. The output, the target and the loss are exactly the MLP's
(normalized s_{k+1} - s_k). The pairs are the same valid transitions in the same order as for the
MLP; the windows are built by index with window_pairs() and gather_windows() from train_jepa.py.
The tuning loss of the transformer is computed in chunks (EVAL_TOKENS // history pairs at a time).
With model "mlp" everything is exactly the computation without this option.

AdamW, lr: linear warmup from 0 over warmup_epochs, then cosine decay to 0 at the end,
updated every step. Gradient norm clipped at grad_clip every step.
The saved model is the one after the last epoch (no best checkpoint).

Usage: python src/train.py configs/train_mlp.yaml [--seed S] [--width W] [--lr LR] [--epochs E]
                          [--model mlp|transformer] [--history N]
Output: results/train/<YYYY-MM-DD_HHMMSS>_mlp_w<width>_lr<lr>_s<seed>/ with
        config.yaml, model.pt, log.csv, meta.json.
        On a full state dataset (state_type "full" in its meta.json) the folder is
        <YYYY-MM-DD_HHMMSS>_mlp_full_w<width>_lr<lr>_s<seed>/ ("_fullnoxy" for state_type "full_no_xy").
        With the transformer, "_tf_h<history>" goes right before "_s<seed>" (..._lr0.001_tf_h10_s0).
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
from models.jepa import TransformerPredictor, ACTION_DIM  # noqa: E402

INT_KEYS = ["width", "depth", "batch_size", "epochs", "warmup_epochs", "seed"]
FLOAT_KEYS = ["lr", "weight_decay", "grad_clip"]
LOG_COLUMNS = ["epoch", "train_loss", "tuning_loss", "lr_end", "n_clipped", "seconds"]
D_MODEL = 256         # token size of the transformer world model
EVAL_TOKENS = 32768   # window states per transformer pass for tuning_loss


def check_config(cfg):
    """Fail early if a value has the wrong type (e.g. 1e-3 read as a string)."""
    assert isinstance(cfg["dataset"], str), cfg["dataset"]
    for key in INT_KEYS:
        assert type(cfg[key]) is int, f"{key} = {cfg[key]!r} is not an int"
    for key in FLOAT_KEYS:
        assert type(cfg[key]) is float, f"{key} = {cfg[key]!r} is not a float"
    assert cfg["epochs"] > cfg["warmup_epochs"], "need at least one epoch after the warmup"


def check_world_model_config(cfg):
    """check_config, plus model and history (check_config is also used by train_jepa.py)."""
    from train_jepa import MAX_HISTORY  # here, not at the top: train_jepa.py imports this file
    check_config(cfg)
    assert cfg["model"] in ["mlp", "transformer"], f"model = {cfg['model']!r} is not mlp or transformer"
    assert type(cfg["history"]) is int, f"history = {cfg['history']!r} is not an int"
    assert 1 <= cfg["history"] <= MAX_HISTORY, (
        f"history = {cfg['history']} is not between 1 and {MAX_HISTORY}: the first valid pair of a run "
        f"is at k = 49, so a longer window would reach before the start of the run")
    assert cfg["model"] == "transformer" or cfg["history"] == 1, "the MLP sees s_k alone: history must be 1"


def dataset_state_type(dataset):
    """"base", "full" or "full_no_xy", from the meta.json next to dataset (a dataset.npz path relative to the repo).

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


def window_transitions(data, split, history):
    """For the transformer: inputs (states, actions, rows) and targets (n, state_dim) of the valid
    transitions of one split: the same pairs, in the same order, and the same targets as transitions().

    states, actions and rows come from window_pairs() in train_jepa.py: the normalized states and
    actions of the split's runs, and the row of each s_k. batch() gathers the windows by index.
    """
    from train_jepa import window_pairs  # here, not at the top: train_jepa.py imports this file
    states, rows, _, actions = window_pairs(data, split, history)
    _, y = transitions(data, split)
    return (states, actions, rows), y


def make_model(model, state_dim, width, depth, history):
    """The world model network for model "mlp" (make_mlp on [s_k, a_k]) or "transformer"
    (TransformerPredictor on the state and action windows). Both output the normalized delta (state_dim)."""
    if model == "mlp":
        return make_mlp(state_dim + ACTION_DIM, state_dim, width, depth)
    assert model == "transformer", f"unknown model {model!r}"
    return TransformerPredictor(state_dim, D_MODEL, state_dim, width, depth, history)


def batch(model, x, idx):
    """The network input of the pairs idx (a tensor of indices into the pairs).

    MLP: x is the input of transitions() and this is x[idx].
    Transformer: x is (states, actions, rows) from window_transitions(); this is the pair
    (state windows (B, history, state_dim), action windows (B, history, 12)), oldest first and step k
    last, gathered by index with gather_windows() from train_jepa.py.
    """
    if not isinstance(model, TransformerPredictor):
        return x[idx]
    from train_jepa import gather_windows  # here, not at the top: train_jepa.py imports this file
    states, actions, rows = x
    r = rows[idx]
    return gather_windows(states, r, model.history), gather_windows(actions, r, model.history)


def forward(model, inputs):
    """Network output (B, state_dim) on inputs from batch(): model(x) for the MLP,
    model(state windows, action windows) for the transformer."""
    if isinstance(model, TransformerPredictor):
        return model(*inputs)
    return model(inputs)


def outputs(model, x):
    """Network outputs (n, state_dim) on all n pairs of x. Call it under torch.no_grad().

    MLP: one pass, model(x). Transformer: chunks of EVAL_TOKENS // history pairs, so the windows of
    one chunk fit in GPU memory.
    """
    if not isinstance(model, TransformerPredictor):
        return model(x)
    rows = x[2]
    idx = torch.arange(len(rows), device=rows.device)
    return torch.cat([forward(model, batch(model, x, i)) for i in idx.split(EVAL_TOKENS // model.history)])


def lr_factor(step, warmup_steps, total_steps):
    """Multiplier of lr at step: 0 -> 1 linearly over warmup_steps, then cosine 1 -> 0 at total_steps."""
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_step(model, optimizer, x, y, grad_clip):
    """One AdamW step on one batch (x from batch()). Returns the loss before the step and whether clipping
    was active."""
    loss = torch.nn.functional.mse_loss(forward(model, x), y)
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # norm before clipping
    optimizer.step()
    return loss.item(), grad_norm.item() > grad_clip


def train(cfg, x_train, y_train, x_tune, y_tune, device):
    """Train a fresh world model. Tensors must already be on device. Returns the final model and the epoch log.

    x is the input of transitions() for the MLP, (states, actions, rows) of window_transitions() for the
    transformer. A cfg without a model key trains the MLP, without a history key with history 1.
    """
    set_seed(cfg["seed"])
    model = make_model(cfg.get("model", "mlp"), y_train.shape[1], cfg["width"], cfg["depth"],
                       cfg.get("history", 1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n, batch_size = len(y_train), cfg["batch_size"]
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
            loss, clipped = train_step(model, optimizer, batch(model, x_train, idx), y_train[idx], cfg["grad_clip"])
            scheduler.step()
            loss_sum += loss * len(idx)                # weight by batch size: the last batch is smaller
            n_clipped += clipped
        with torch.no_grad():
            tuning_loss = torch.nn.functional.mse_loss(outputs(model, x_tune), y_tune).item()
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
    parser.add_argument("--model", choices=["mlp", "transformer"])
    parser.add_argument("--history", type=int)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("model", "mlp")   # configs without the key: mlp (saved in config.yaml)
    cfg.setdefault("history", 1)     # configs without the key: 1 (saved in config.yaml)
    for key in ["seed", "width", "lr", "epochs", "model", "history"]:
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    check_world_model_config(cfg)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    full = {"base": "", "full": "_full", "full_no_xy": "_fullnoxy"}[dataset_state_type(cfg["dataset"])]
    tf = f"_tf_h{cfg['history']}" if cfg["model"] == "transformer" else ""
    name = f"{stamp}_mlp{full}_w{cfg['width']}_lr{cfg['lr']:g}{tf}_s{cfg['seed']}"
    out_dir = os.path.join(REPO_ROOT, "results", "train", name)
    os.makedirs(out_dir)  # raises if the folder already exists: never overwrite
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))

    data = dict(np.load(os.path.join(REPO_ROOT, cfg["dataset"])))
    if cfg["model"] == "mlp":
        x_train, y_train = transitions(data, "train")
        x_tune, y_tune = transitions(data, "tuning")
        x_train, x_tune = torch.from_numpy(x_train).to(device), torch.from_numpy(x_tune).to(device)
    else:
        x_train, y_train = window_transitions(data, "train", cfg["history"])
        x_tune, y_tune = window_transitions(data, "tuning", cfg["history"])
        x_train = tuple(torch.from_numpy(v).to(device) for v in x_train)
        x_tune = tuple(torch.from_numpy(v).to(device) for v in x_tune)
    y_train, y_tune = torch.from_numpy(y_train).to(device), torch.from_numpy(y_tune).to(device)
    print(f"train pairs {len(y_train)}, tuning pairs {len(y_tune)}, model {cfg['model']}, history {cfg['history']}")

    model, log = train(cfg, x_train, y_train, x_tune, y_tune, device)

    n_params = sum(p.numel() for p in model.parameters())
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state_dict, "width": cfg["width"], "depth": cfg["depth"],
                "state_dim": y_train.shape[1], "model": cfg["model"], "history": cfg["history"]},
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
