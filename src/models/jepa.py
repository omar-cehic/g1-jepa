"""JEPA world model (following LeWorldModel): an encoder and an action conditioned predictor.

  encode(s)     s  normalized state s_k (N, 34)                     -> z_k (N, embed_dim)
  predict(z, a) z  embedding z_k (N, embed_dim), a normalized a_k (N, 12)
                -> z_hat_{k+1} = z + predictor([z, a])               (N, embed_dim)

The predictor outputs the CHANGE of the embedding over one step (residual), the same way
the MLP world model outputs the change of the state.
Both parts are make_mlp networks (Linear + ReLU hidden layers, final Linear), with the same
width and depth as the MLP. No dropout, normalization layers, projector, EMA or stop gradient.
Inputs are normalized with the train stats in dataset.npz (normalize in src/models/mlp.py).
"""

import torch
from torch import nn

from models.mlp import make_mlp

STATE_DIM = 34
ACTION_DIM = 12


class JEPA(nn.Module):
    def __init__(self, embed_dim, width, depth):
        super().__init__()
        self.encoder = make_mlp(STATE_DIM, embed_dim, width, depth)
        self.predictor = make_mlp(embed_dim + ACTION_DIM, embed_dim, width, depth)

    def encode(self, s):
        """Normalized states (N, 34) -> embeddings (N, embed_dim)."""
        return self.encoder(s)

    def predict(self, z, a):
        """Embeddings (N, embed_dim) and normalized actions (N, 12) -> predicted next embeddings."""
        return z + self.predictor(torch.cat([z, a], dim=1))
