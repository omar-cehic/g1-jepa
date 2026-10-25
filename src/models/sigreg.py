# Copied from LeWorldModel module.py at commit 8edfeb3
# (github.com/lucas-maes/le-wm), MIT License:
#
# Copyright (c) 2026 Lucas Maes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""SIGReg: pushes a batch of embeddings toward a standard Gaussian N(0, I).

Input proj has shape (T, B, D): T time steps, a batch of B embeddings, D numbers each.
Each call draws num_proj new random unit directions in D (resampled on every call) and
projects every embedding onto each one, giving num_proj sets of B scalars per time step.

For each set it compares the empirical characteristic function of those B scalars,
mean_b exp(i t x_b) = mean cos(t x) + i mean sin(t x), with the standard Gaussian's,
exp(-t^2 / 2). The squared gap is integrated over t in [0, 3] (17 knots, trapezoid
rule weighted by exp(-t^2 / 2), doubled because the gap is symmetric in t, so this
covers [-3, 3]). This is the Epps-Pulley test statistic.

The value is multiplied by the batch size B, then averaged over the projections and T.
Because of the factor B it does not shrink with batch size for a Gaussian batch
(it stays near 1.05), but grows about linearly in B for a non Gaussian batch.
The class is kept exactly as in LeWorldModel; the projections are float32.
"""

import torch


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
