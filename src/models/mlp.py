"""MLP world model: predicts the one step change of the state from (state, action).

  input   [normalized s_k (34), normalized a_k (12)]   (46)
  output  normalized delta_k (34),  delta_k = s_{k+1} - s_k

Normalization uses the train stats stored in dataset.npz (state_, action_, delta_ mean and std).
It is done in float64 in numpy; only the network runs in float32.
"""

import numpy as np
import torch
from torch import nn


def make_mlp(in_dim, out_dim, width, depth):
    """depth hidden layers (Linear + ReLU) of size width, then a final Linear with no activation."""
    layers = []
    d = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def normalize(x, stats, name):
    """(x - mean) / std with the stats of name ("state", "action" or "delta"), in float64."""
    return (x.astype(np.float64) - stats[name + "_mean"]) / stats[name + "_std"]


def denormalize(x, stats, name):
    """Inverse of normalize, in float64."""
    return x.astype(np.float64) * stats[name + "_std"] + stats[name + "_mean"]


def model_input(states, actions, stats):
    """Network input (N, 46) float64 from raw states (N, 34) and actions (N, 12)."""
    return np.concatenate([normalize(states, stats, "state"), normalize(actions, stats, "action")], axis=1)


def make_step(model, stats, device):
    """step(states, actions) -> next states, for src/eval/rollout.py.

    states (N, 34) raw float64, actions (N, 12) raw. Returns states + predicted delta in float64.
    """
    def step(states, actions):
        x = torch.from_numpy(model_input(states, actions, stats).astype(np.float32)).to(device)
        with torch.no_grad():
            out = model(x).cpu().numpy()
        return states + denormalize(out, stats, "delta")
    return step
