"""JEPA world model (following LeWorldModel): an encoder and an action conditioned predictor.

  encode(s)        s  normalized state s_k (N, state_dim)                -> z_k (N, embed_dim)
  encode_window(s) s  windows of normalized states (N, n, state_dim)     -> (N, n, embed_dim),
                      each state encoded alone
  predict(z, a, z_past)
                   z  embedding z_k (N, embed_dim), a normalized a_k (N, 12),
                   z_past  the older embeddings z_{k-history+1}, ..., z_{k-1} (N, history - 1, embed_dim),
                           oldest first; None for history 1
                   -> z_hat_{k+1} = z + predictor([z_{k-history+1}, ..., z_{k-1}, z_k, a])   (N, embed_dim)

The predictor outputs the CHANGE of the embedding over one step (residual), the same way
the MLP world model outputs the change of the state. Its input has history * embed_dim + 12
numbers: the window's embeddings one after the other (oldest first, z_k last), then the action.
With history 1 it is [z_k, a], the model without history.
Both parts are make_mlp networks (Linear + ReLU hidden layers, final Linear), with the same
width and depth as the MLP. No dropout, normalization layers, projector, EMA or stop gradient.
Inputs are normalized with the train stats in dataset.npz (normalize in src/models/mlp.py).
state_dim is the number of state features in the dataset: 34 (base state) or 37 (full state).
"""

import torch
from torch import nn

from models.mlp import make_mlp

ACTION_DIM = 12


class JEPA(nn.Module):
    def __init__(self, embed_dim, width, depth, state_dim=34, history=1):
        super().__init__()
        self.history = history
        self.encoder = make_mlp(state_dim, embed_dim, width, depth)
        self.predictor = make_mlp(history * embed_dim + ACTION_DIM, embed_dim, width, depth)

    def encode(self, s):
        """Normalized states (N, state_dim) -> embeddings (N, embed_dim)."""
        return self.encoder(s)

    def encode_window(self, s):
        """Windows of normalized states (N, n, state_dim) -> embeddings (N, n, embed_dim)."""
        n_windows, n = s.shape[0], s.shape[1]
        z = self.encoder(s.reshape(n_windows * n, s.shape[2]))   # every state on its own
        return z.reshape(n_windows, n, z.shape[1])

    def predict(self, z, a, z_past=None):
        """Embeddings z_k (N, embed_dim), normalized actions (N, 12) and, for history > 1, the older
        embeddings (N, history - 1, embed_dim) -> predicted next embeddings (N, embed_dim)."""
        parts = [z, a] if z_past is None else [z_past.flatten(1), z, a]
        return z + self.predictor(torch.cat(parts, dim=1))
