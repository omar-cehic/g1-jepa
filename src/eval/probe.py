"""Linear probe: a straight line (plane) fit from embeddings back to states.

Given embeddings Z (N, d) and targets Y (N, m), find W (d, m) and b (m,) that make
  Y ≈ Z W + b
as close as possible in squared error (ordinary least squares with a bias). It is solved with
np.linalg.lstsq on [Z, ones] in float64.

lstsq also works when the columns of Z are not independent (rank deficient Z, for example a
collapsed encoder that gives the same embedding for every state). Then many W, b are equally
good; lstsq returns the one with the smallest norm, and its outputs Z W + b are still the best
possible least squares fit. If every row of Z is the same, that best fit is the mean of Y.
"""

import numpy as np


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
