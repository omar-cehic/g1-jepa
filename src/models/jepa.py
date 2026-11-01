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

With predictor "transformer" the encoder is the same, and predict(z, a, z_past) takes the action
window a = a_{k-history+1}, ..., a_k (N, history, 12), oldest first, instead of a_k alone.
The predictor is TransformerPredictor: for each slot j of the window
  token_j = Linear([z_j, a_j]) + p_j        (embed_dim + 12 -> embed_dim, p_j a learned position vector)
then depth pre-LayerNorm transformer blocks (nhead 4, feedforward width, ReLU, no dropout, no
attention mask: every token sees the whole window, which ends at step k), then a LayerNorm and a
Linear (embed_dim -> embed_dim) on the last token only give the change of the embedding:
  z_hat_{k+1} = z_k + Linear(LayerNorm(last token))
Inputs are normalized with the train stats in dataset.npz (normalize in src/models/mlp.py).
state_dim is the number of state features in the dataset: 34 (base state) or 37 (full state).
"""

import torch
from torch import nn

from models.mlp import make_mlp

ACTION_DIM = 12


class TransformerPredictor(nn.Module):
    """Windows of inputs (N, n, in_dim) and actions (N, n, 12), oldest first -> one output vector
    (N, out_dim) from the last token. n must be history (one position vector per window slot).

    In the JEPA the inputs are embeddings: in_dim = d_model = out_dim = embed_dim, and the output is
    the change of the last embedding. The transformer world model in src/train.py uses it on states:
    in_dim = out_dim = state_dim, d_model 256, and the output is the normalized change of the state.
    """

    def __init__(self, in_dim, d_model, out_dim, width, depth, history):
        super().__init__()
        self.history = history
        self.token = nn.Linear(in_dim + ACTION_DIM, d_model)
        self.position = nn.Parameter(torch.empty(history, d_model))
        nn.init.normal_(self.position, std=0.02)
        # depth layers built one by one, so each starts from its own random weights
        # (nn.TransformerEncoder deep copies one layer, so all layers would start the same)
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=width, dropout=0.0,
                                       activation="relu", batch_first=True, norm_first=True)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, out_dim)

    def forward(self, z, a):
        x = self.token(torch.cat([z, a], dim=2)) + self.position
        x = self.blocks(x)
        return self.out(self.norm(x[:, -1]))     # last token only: slot k


class JEPA(nn.Module):
    def __init__(self, embed_dim, width, depth, state_dim=34, history=1, predictor="mlp"):
        super().__init__()
        self.history = history
        self.predictor_type = predictor
        self.encoder = make_mlp(state_dim, embed_dim, width, depth)
        if predictor == "mlp":
            self.predictor = make_mlp(history * embed_dim + ACTION_DIM, embed_dim, width, depth)
        else:
            assert predictor == "transformer", f"unknown predictor {predictor!r}"
            self.predictor = TransformerPredictor(embed_dim, embed_dim, embed_dim, width, depth, history)

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
        embeddings (N, history - 1, embed_dim) -> predicted next embeddings (N, embed_dim).
        For the transformer predictor, a is the action window (N, history, 12), a_k last."""
        if self.predictor_type == "transformer":
            z_window = z[:, None] if z_past is None else torch.cat([z_past, z[:, None]], dim=1)
            return z + self.predictor(z_window, a)
        parts = [z, a] if z_past is None else [z_past.flatten(1), z, a]
        return z + self.predictor(torch.cat(parts, dim=1))
