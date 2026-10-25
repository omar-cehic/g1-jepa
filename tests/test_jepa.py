"""Tests for src/models/jepa.py and src/train_jepa.py.

Uses data/processed/2026-10-07_182522.

(a) Parameter counts at width 1024, depth 3 (encoder / predictor / total):
    embed_dim  64: 2,200,640 / 2,243,648 / 4,444,288
    embed_dim 128: 2,266,240 / 2,374,784 / 4,641,024
    embed_dim 256: 2,397,440 / 2,637,056 / 5,034,496
    Shapes: encode (N, 34) -> (N, d); predict ((N, d), (N, 12)) -> (N, d).
(b) Residual: with the predictor's last Linear weight and bias set to zero, predict(z, a) == z exactly.
(c) Collapse: encoder's last Linear weight 0 and bias a fixed nonzero vector, predictor's last
    Linear weight and bias 0. On 512 real train pairs pred_loss is exactly 0, so the loss at
    sigreg_weight 0 is exactly 0 (the global minimum). SIGReg of this constant embedding is printed
    and must be above 10 (far above the ~1.05 of a Gaussian batch).
(d) No stop gradient: on one real batch, encoder gradients of the loss as written (jepa_losses)
    differ from those with z_next detached, and the z_next branch alone gives nonzero gradient.
(e) pairs(): 226,099 train and 28,381 tuning pairs, float32; on the first 3 train runs,
    s_k and s_{k+1} of each pair equal the hand normalized states at rows k and k+1 of its run.
(f) On CPU, a fresh JEPA (embed_dim 64) trained with train_step on one fixed batch of 512 train
    pairs for 300 steps (sigreg_weight 0.0225, AdamW lr 0.001): the final total loss is below the
    initial one and the final SIGReg is below half the initial SIGReg.
(g) Determinism: on CPU, train() twice with the same seed (20 steps on 640 train pairs,
    embed_dim 64) gives identical weights; a different seed gives different weights.

Nothing is written. Run: python tests/test_jepa.py
"""

import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from models.jepa import JEPA  # noqa: E402
from models.sigreg import SIGReg  # noqa: E402
import train_jepa  # noqa: E402

DATASET = os.path.join(REPO_ROOT, "data", "processed", "2026-10-07_182522", "dataset.npz")
CPU = torch.device("cpu")
CFG = {"width": 1024, "depth": 3, "lr": 0.001, "weight_decay": 0.001, "grad_clip": 1.0,
       "sigreg_weight": 0.0225}


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def train_batch(data, n):
    """The first n train pairs as CPU tensors (s, a, s_next)."""
    return [torch.from_numpy(x[:n]) for x in train_jepa.pairs(data, "train")]


def test_shapes():
    expected = {64: (2200640, 2243648, 4444288),
                128: (2266240, 2374784, 4641024),
                256: (2397440, 2637056, 5034496)}
    for d, counts in expected.items():
        model = JEPA(d, 1024, 3)
        got = (n_params(model.encoder), n_params(model.predictor), n_params(model))
        assert got == counts, (d, got)
        z = model.encode(torch.zeros(5, 34))
        z_hat = model.predict(z, torch.zeros(5, 12))
        assert z.shape == (5, d) and z_hat.shape == (5, d), (z.shape, z_hat.shape)
        print(f"(a) embed_dim {d}: encoder {got[0]:,}, predictor {got[1]:,}, total {got[2]:,}; "
              f"encode (5, 34) -> {tuple(z.shape)}, predict -> {tuple(z_hat.shape)}")


def test_residual():
    torch.manual_seed(0)
    model = JEPA(64, 1024, 3)
    with torch.no_grad():
        model.predictor[-1].weight.zero_()
        model.predictor[-1].bias.zero_()
    z, a = torch.randn(100, 64), torch.randn(100, 12)
    z_hat = model.predict(z, a)
    assert torch.equal(z_hat, z)
    print(f"(b) predictor's last layer zeroed: predict(z, a) == z exactly "
          f"(max |diff| {(z_hat - z).abs().max().item()})")


def test_collapse(data):
    s, a, s_next = train_batch(data, 512)
    torch.manual_seed(0)
    model = JEPA(64, 1024, 3)
    sigreg = SIGReg()
    constant = torch.linspace(-1.0, 1.0, 64)
    with torch.no_grad():
        model.encoder[-1].weight.zero_()
        model.encoder[-1].bias.copy_(constant)
        model.predictor[-1].weight.zero_()
        model.predictor[-1].bias.zero_()
        pred_loss, sig = train_jepa.jepa_losses(model, sigreg, s, a, s_next)
    loss = pred_loss + 0.0 * sig
    assert pred_loss.item() == 0.0 and loss.item() == 0.0, (pred_loss.item(), loss.item())
    assert sig.item() > 10, sig.item()
    print(f"(c) constant embedding on 512 train pairs: pred_loss {pred_loss.item()}, "
          f"loss at sigreg_weight 0 = {loss.item()} (global minimum); SIGReg {sig.item():.2f} "
          f"(Gaussian batch ~1.05)")


def test_no_stop_gradient(data):
    s, a, s_next = train_batch(data, 512)
    torch.manual_seed(0)
    model = JEPA(64, 1024, 3)
    sigreg = SIGReg()
    w = CFG["sigreg_weight"]

    def encoder_grads(loss):
        model.zero_grad()
        loss.backward()
        return torch.cat([p.grad.flatten() for p in model.encoder.parameters()])

    torch.manual_seed(1)                                # same SIGReg projections in both
    pred_loss, sig = train_jepa.jepa_losses(model, sigreg, s, a, s_next)
    g_full = encoder_grads(pred_loss + w * sig)

    torch.manual_seed(1)
    z = model.encode(s)
    z_next = model.encode(s_next).detach()
    pred_detached = (model.predict(z, a) - z_next).square().mean()
    sig_detached = sigreg(torch.stack([z, z_next]))
    assert torch.equal(pred_detached, pred_loss) and torch.equal(sig_detached, sig)  # same forward values
    g_detached = encoder_grads(pred_detached + w * sig_detached)

    g_branch = g_full - g_detached                      # gradient that comes through z_next only
    assert not torch.equal(g_full, g_detached)
    assert g_branch.norm().item() > 0
    print(f"(d) encoder grad norm: loss as written {g_full.norm().item():.4e}, z_next detached "
          f"{g_detached.norm().item():.4e}, through z_next alone {g_branch.norm().item():.4e}")


def test_pairs(data):
    for split, n_expected in [("train", 226099), ("tuning", 28381)]:
        s, a, s_next = train_jepa.pairs(data, split)
        assert s.dtype == a.dtype == s_next.dtype == np.float32
        assert s.shape == (n_expected, 34) and a.shape == (n_expected, 12) and s_next.shape == (n_expected, 34)
    s, a, s_next = train_jepa.pairs(data, "train")
    runs = np.flatnonzero(data["split"] == "train")[:3]
    offset, n_checked = 0, 0
    for run in runs:
        ks = np.flatnonzero(data["valid"][run])          # pairs are stored run by run, k ascending
        rows = data["states"][run]
        hand_s = ((rows[ks] - data["state_mean"]) / data["state_std"]).astype(np.float32)
        hand_next = ((rows[ks + 1] - data["state_mean"]) / data["state_std"]).astype(np.float32)
        assert np.array_equal(s[offset:offset + len(ks)], hand_s)
        assert np.array_equal(s_next[offset:offset + len(ks)], hand_next)
        offset += len(ks)
        n_checked += len(ks)
    print(f"(e) 226099 train pairs, 28381 tuning pairs; s_k and s_{{k+1}} equal hand normalized rows "
          f"k and k+1 for all {n_checked} pairs of the first {len(runs)} train runs")


def summary(model, sigreg, s, a, s_next, w):
    with torch.no_grad():
        pred_loss, sig = train_jepa.jepa_losses(model, sigreg, s, a, s_next)
        emb_std = model.encode(s).std(dim=0).mean().item()
    return pred_loss.item(), sig.item(), pred_loss.item() + w * sig.item(), emb_std


def test_fixed_batch(data):
    s, a, s_next = train_batch(data, 512)
    w = CFG["sigreg_weight"]
    torch.manual_seed(0)
    model = JEPA(64, CFG["width"], CFG["depth"])
    sigreg = SIGReg()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    t0 = time.time()
    first = summary(model, sigreg, s, a, s_next, w)
    for _ in range(300):
        train_jepa.train_step(model, sigreg, optimizer, s, a, s_next, w, CFG["grad_clip"])
    last = summary(model, sigreg, s, a, s_next, w)
    print(f"(f) one fixed batch of 512, 300 steps ({time.time() - t0:.0f} s):")
    for name, i in [("pred_loss", 0), ("SIGReg", 1), ("total loss", 2), ("embedding std", 3)]:
        print(f"      {name:14s} {first[i]:.5f} -> {last[i]:.5f}")
    assert last[2] < first[2], (first[2], last[2])
    assert last[1] < first[1] / 2, (first[1], last[1])
    print("    final total loss < initial, final SIGReg < initial / 2")


def test_determinism(data):
    s, a, s_next = train_batch(data, 640)
    cfg = dict(CFG, embed_dim=64, batch_size=64, epochs=2, warmup_epochs=1)   # 2 x 10 = 20 steps
    print("    determinism runs:")
    weights = []
    for seed in [0, 0, 1]:
        model, _ = train_jepa.train(dict(cfg, seed=seed), (s, a, s_next),
                                    (s[:128], a[:128], s_next[:128]), CPU)
        weights.append(model.state_dict())
    same = all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])
    different = not any(torch.equal(weights[0][k], weights[2][k]) for k in weights[0])
    assert same and different
    print("(g) 20 steps on CPU: seed 0 twice gives identical weights; seed 1 gives different weights")


def main():
    data = dict(np.load(DATASET))
    test_shapes()
    test_residual()
    test_collapse(data)
    test_no_stop_gradient(data)
    test_pairs(data)
    test_fixed_batch(data)
    test_determinism(data)
    print("PASS test_jepa")


if __name__ == "__main__":
    main()
