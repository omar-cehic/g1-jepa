"""Tests for src/models/sigreg.py (SIGReg copied from LeWorldModel).

Synthetic data only, float32 inputs. The reference t, integration weights and phi are rebuilt
here in numpy (float64), not read from the class buffers.

(a) All zero embeddings (1, 512, 64): every projection is 0, so the value is exactly
    512 * sum_k weights_k * (1 - phi_k)^2, about 205.85 (relative tolerance 1e-4).
(b) Gaussian embeddings: the mean over 200 draws of (1, 512, 64) from N(0, I) is within 0.05 of
    the exact expectation sum_k weights_k * (1 - phi_k^2), about 1.0525.
(c) Non Gaussian embeddings (mean of 5 draws, D = 64): N(0, I) times 2, and only the first 2 of
    64 numbers N(0, 1) with the rest zero. Both are above 50 at batch 512, and
    value(512) / value(128) is between 3 and 5 (the factor B makes it grow linearly).
(d) Gradient descent: 512 trainable points in D = 8, started uniform on [0, 3], minimized with
    Adam (lr 0.01) for 2000 steps on CPU. Start above 20; then the mean of 20 fresh calls is
    below 1.5, every |coordinate mean| < 0.1 and max |covariance - I| < 0.2.
(e) If CUDA is available: a (2, 512, 64) input on cuda gives a finite value and finite
    gradients. Otherwise SKIPPED (not counted as a pass).

Nothing is written. Run: python tests/test_sigreg.py
"""

import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from models.sigreg import SIGReg  # noqa: E402

B, D = 512, 64


def reference_weights_phi(knots=17):
    """Trapezoid weights on [0, 3], doubled for the symmetric half of [-3, 3], times phi."""
    t = np.linspace(0, 3, knots)
    dt = 3 / (knots - 1)
    trapezoid = np.full(knots, dt)
    trapezoid[[0, -1]] = dt / 2
    phi = np.exp(-t**2 / 2)                  # characteristic function of N(0, 1)
    return 2 * trapezoid * phi, phi


def test_zeros(sigreg):
    w, phi = reference_weights_phi()
    expected = B * np.sum(w * (1 - phi)**2)
    value = sigreg(torch.zeros(1, B, D)).item()
    rel = abs(value - expected) / expected
    assert abs(expected - 205.85) / 205.85 < 1e-4, expected
    assert rel < 1e-4, (value, expected)
    print(f"(a) zeros (1, {B}, {D}): value {value:.4f}, closed form {expected:.4f}, "
          f"relative error {rel:.1e}")


def test_gaussian(sigreg):
    w, phi = reference_weights_phi()
    expected = np.sum(w * (1 - phi**2))
    assert abs(expected - 1.0525) < 1e-4, expected
    values = np.array([sigreg(torch.randn(1, B, D)).item() for _ in range(200)])
    mean, se = values.mean(), values.std(ddof=1) / np.sqrt(len(values))
    assert abs(mean - expected) < 0.05, (mean, expected)
    print(f"(b) N(0, I) (1, {B}, {D}), 200 draws: mean {mean:.4f} +- {se:.4f} (standard error), "
          f"expected {expected:.4f}, |difference| {abs(mean - expected):.4f}")


def scaled(batch):
    return 2 * torch.randn(1, batch, D)


def two_dims(batch):
    x = torch.zeros(1, batch, D)
    x[..., :2] = torch.randn(1, batch, 2)
    return x


def test_bad_cases(sigreg):
    for name, make in [("N(0, I) times 2", scaled), ("first 2 of 64 dims N(0, 1)", two_dims)]:
        v = {batch: np.mean([sigreg(make(batch)).item() for _ in range(5)]) for batch in [512, 128]}
        ratio = v[512] / v[128]
        assert v[512] > 50, (name, v[512])
        assert 3 < ratio < 5, (name, ratio)
        print(f"(c) {name}: value(512) {v[512]:.2f}, value(128) {v[128]:.2f}, ratio {ratio:.2f}")


def test_gradient_descent():
    sigreg = SIGReg()
    x = (3 * torch.rand(B, 8)).requires_grad_()
    optimizer = torch.optim.Adam([x], lr=0.01)
    for step in range(2000):
        loss = sigreg(x.unsqueeze(0))
        if step == 0:
            start = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = np.mean([sigreg(x.unsqueeze(0)).item() for _ in range(20)])
    points = x.detach().numpy().astype(np.float64)
    mean_err = np.abs(points.mean(axis=0)).max()
    cov_err = np.abs(np.cov(points, rowvar=False) - np.eye(8)).max()
    print(f"(d) uniform [0, 3] in D = 8, 2000 Adam steps: start {start:.2f}, final (mean of 20) "
          f"{final:.4f}, max |coordinate mean| {mean_err:.4f}, max |covariance - I| {cov_err:.4f}")
    assert start > 20, start
    assert final < 1.5, final
    assert mean_err < 0.1, mean_err
    assert cov_err < 0.2, cov_err


def test_cuda():
    if not torch.cuda.is_available():
        print("(e) SKIPPED (no CUDA)")
        return False
    sigreg = SIGReg().to("cuda")
    x = torch.randn(2, B, D, device="cuda", requires_grad=True)
    value = sigreg(x)
    value.backward()
    assert torch.isfinite(value).item()
    assert torch.isfinite(x.grad).all().item()
    print(f"(e) cuda ({torch.cuda.get_device_name()}) (2, {B}, {D}): value {value.item():.4f}, "
          f"all {x.grad.numel()} gradient entries finite, max |gradient| {x.grad.abs().max().item():.2e}")
    return True


def timed(test, *args):
    start = time.perf_counter()
    result = test(*args)
    print(f"    {time.perf_counter() - start:.1f} s")
    return result


def main():
    torch.manual_seed(0)
    sigreg = SIGReg()
    timed(test_zeros, sigreg)
    timed(test_gaussian, sigreg)
    timed(test_bad_cases, sigreg)
    timed(test_gradient_descent)
    if timed(test_cuda):
        print("PASS test_sigreg")
    else:
        print("PASS test_sigreg (a)-(d); (e) SKIPPED (no CUDA)")


if __name__ == "__main__":
    main()
