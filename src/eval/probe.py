"""Probes: maps fit from embeddings back to states. A linear probe and a small MLP probe.

Linear probe
Given embeddings Z (N, d) and targets Y (N, m), find W (d, m) and b (m,) that make
  Y ≈ Z W + b
as close as possible in squared error (ordinary least squares with a bias). It is solved with
np.linalg.lstsq on [Z, ones] in float64.

lstsq also works when the columns of Z are not independent (rank deficient Z, for example a
collapsed encoder that gives the same embedding for every state). Then many W, b are equally
good; lstsq returns the one with the smallest norm, and its outputs Z W + b are still the best
possible least squares fit. If every row of Z is the same, that best fit is the mean of Y.

MLP probe
A small network (2 hidden layers of 256, ReLU) trained with gradient descent to map Z to Y.
It can also find curved (nonlinear) maps that the linear probe cannot. See fit_mlp_probe.
"""

import math

import numpy as np
import torch

from models.mlp import make_mlp
from train import set_seed, lr_factor, train_step

MLP_PROBE_WIDTH = 256
MLP_PROBE_DEPTH = 2
MLP_PROBE_LR = 0.001
MLP_PROBE_WEIGHT_DECAY = 0.001
MLP_PROBE_BATCH_SIZE = 512
MLP_PROBE_WARMUP_EPOCHS = 1
MLP_PROBE_GRAD_CLIP = 1.0
APPLY_BATCH_SIZE = 16384   # rows per forward pass in apply_mlp_probe; only affects speed and memory


def fit_linear_probe(Z, Y):
    """Least squares W (d, m), b (m,) with Y ≈ Z W + b, in float64."""
    Z = np.asarray(Z, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = np.concatenate([Z, np.ones((len(Z), 1))], axis=1)   # last column multiplies the bias
    coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return coef[:-1], coef[-1]


def apply_probe(Z, W, b):
    """Probe output Z W + b (N, m), in float64."""
    return np.asarray(Z, dtype=np.float64) @ W + b


def fit_mlp_probe(Z, Y, device, seed=0, epochs=20, width=MLP_PROBE_WIDTH, depth=MLP_PROBE_DEPTH):
    """Train a fresh small MLP that maps embeddings Z (N, d) to targets Y (N, m). Returns the network.

    Z and Y are fixed arrays: when Z comes from a JEPA, the embeddings were computed once by the
    frozen JEPA before this call, so only the probe's own weights are trained here.
    Network: make_mlp(d, m, width, depth), by default 2 hidden layers of 256 with ReLU and a final Linear.
    Training is the same recipe as src/train.py: mean squared error, AdamW (lr 0.001, weight decay
    0.001), batches of 512 with the last partial batch kept, lr warmup over 1 epoch then cosine to 0,
    updated every step, gradient norm clipped at 1.0, shuffled with a seeded CPU generator.
    Runs in float32 on device. epochs is 20 for real use; the tests use fewer to run fast.
    """
    Z = torch.from_numpy(np.asarray(Z, dtype=np.float32)).to(device)
    Y = torch.from_numpy(np.asarray(Y, dtype=np.float32)).to(device)
    set_seed(seed)
    model = make_mlp(Z.shape[1], Y.shape[1], width, depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_PROBE_LR, weight_decay=MLP_PROBE_WEIGHT_DECAY)
    n = len(Z)
    steps_per_epoch = math.ceil(n / MLP_PROBE_BATCH_SIZE)
    warmup_steps = MLP_PROBE_WARMUP_EPOCHS * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, warmup_steps, total_steps))
    generator = torch.Generator().manual_seed(seed)  # on the CPU: same order on any device

    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator).to(device)
        for start in range(0, n, MLP_PROBE_BATCH_SIZE):
            idx = order[start:start + MLP_PROBE_BATCH_SIZE]
            train_step(model, optimizer, Z[idx], Y[idx], MLP_PROBE_GRAD_CLIP)
            scheduler.step()
    model.eval()
    return model


def apply_mlp_probe(model, Z, device):
    """Probe output (N, m) of the network from fit_mlp_probe on Z (N, d), as float64 numpy.

    Runs in float32 batches with no gradients.
    """
    Z = np.asarray(Z, dtype=np.float32)
    parts = []
    with torch.no_grad():
        for start in range(0, len(Z), APPLY_BATCH_SIZE):
            z = torch.from_numpy(Z[start:start + APPLY_BATCH_SIZE]).to(device)
            parts.append(model(z).cpu().numpy().astype(np.float64))
    return np.concatenate(parts)
