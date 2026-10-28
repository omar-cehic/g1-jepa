"""JEPA world model (following LeWorldModel): an encoder and an action conditioned predictor.

  encode(s)     s  normalized state s_k (N, state_dim)              -> z_k (N, embed_dim)
  predict(z, a) z  embedding z_k (N, embed_dim), a normalized a_k (N, 12)
                -> z_hat_{k+1} = z + predictor([z, a])               (N, embed_dim)

The predictor outputs the CHANGE of the embedding over one step (residual), the same way
the MLP world model outputs the change of the state.
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
    def __init__(self, embed_dim, width, depth, state_dim=34):
        super().__init__()
        self.encoder = make_mlp(state_dim, embed_dim, width, depth)
        self.predictor = make_mlp(embed_dim + ACTION_DIM, embed_dim, width, depth)

    def encode(self, s):
        """Normalized states (N, state_dim) -> embeddings (N, embed_dim)."""
        return self.encoder(s)

    def predict(self, z, a):
        """Embeddings (N, embed_dim) and normalized actions (N, 12) -> predicted next embeddings."""
        return z + self.predictor(torch.cat([z, a], dim=1))
