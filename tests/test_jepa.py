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
(d) No stop gradient: on one real batch, encoder gradients of the loss as written (jepa_losses
    with stop_grad off) differ from those with z_next detached, and the z_next branch alone gives
    nonzero gradient.
(e) pairs(): 226,099 train and 28,381 tuning pairs, float32; on the first 3 train runs,
    s_k and s_{k+1} of each pair equal the hand normalized states at rows k and k+1 of its run.
(f) On CPU, a fresh JEPA (embed_dim 64) trained with train_step on one fixed batch of 512 train
    pairs for 300 steps (sigreg_weight 0.0225, AdamW lr 0.001): the final total loss is below the
    initial one and the final SIGReg is below half the initial SIGReg.
(g) Determinism: on CPU, train() twice with the same seed (20 steps on 640 train pairs,
    embed_dim 64) gives identical weights; a different seed gives different weights.
(h) Stop gradient option, on one real batch of 512 (the global torch RNG is seeded before every
    loss, so SIGReg draws the same projections each time):
    (h1) the total loss value is identical with stop_grad on and off;
    (h2) with it on, the encoder gradient equals the one where the target is a constant from a
         separate torch.no_grad() forward pass;
    (h3) the encoder gradient of the SIGReg term alone is identical with it on and off;
    (h4) the encoder gradient of the total loss differs between on and off.
State history:
(i) window_pairs() gives 226,099 train and 28,381 tuning pairs for history 1, 3, 10 and 40, the same
    pairs as pairs() (same rows for every history, states[rows] == s_k and states[rows + 1] == s_{k+1}
    of pairs()). History 51 is refused by window_pairs() and by check_jepa_config(), history 0 by
    check_jepa_config().
(j) Windows: at history 10 and 40, for the first, middle and last pair of each of the first 3 train
    runs and the first pair after every gap in its valid steps, the window from gather_windows()
    equals the hand normalized rows k-n+1, ..., k of that run, and its last state is s_k of pairs().
(k) Residual at history 10: predict(z, a, z_past) equals z + predictor([z_past[:, 0], ..., z_past[:, 8], z, a])
    built by hand (oldest first, z_k last, then a_k), exactly; with the predictor's last Linear zeroed it is z.
(l) Loss at history 10 on one real batch of 512: the SIGReg value is exactly the one of z_k and z_{k+1}
    encoded alone, pred_loss matches a hand computation (each window state encoded alone), the gradient of
    pred_loss reaches the oldest window state, and the gradient of SIGReg alone on the older window
    states is exactly zero.
(m) Predictor weight counts at embed_dim 256 for history 1, 3, 10 and 40 (printed; history 1 is the
    2,637,056 of (a)).

Nothing is written. Run: python tests/test_jepa.py
"""

import os
import sys
import time

import numpy as np
import torch
import yaml

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
    pred_loss, sig = train_jepa.jepa_losses(model, sigreg, s, a, s_next, stop_grad=False)
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
    states, rows, a = (torch.from_numpy(x) for x in train_jepa.window_pairs(data, "train", 1))
    cfg = dict(CFG, embed_dim=64, batch_size=64, epochs=2, warmup_epochs=1)   # 2 x 10 = 20 steps
    print("    determinism runs:")
    weights = []
    for seed in [0, 0, 1]:
        model, _ = train_jepa.train(dict(cfg, seed=seed), (states, rows[:640], a[:640]),
                                    (states, rows[:128], a[:128]), CPU)
        weights.append(model.state_dict())
    same = all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])
    different = not any(torch.equal(weights[0][k], weights[2][k]) for k in weights[0])
    assert same and different
    print("(g) 20 steps on CPU: seed 0 twice gives identical weights; seed 1 gives different weights")


def test_stop_grad(data):
    s, a, s_next = train_batch(data, 512)
    torch.manual_seed(0)
    model = JEPA(64, 1024, 3)
    sigreg = SIGReg()
    w = CFG["sigreg_weight"]

    def encoder_grads(loss):
        model.zero_grad()
        loss.backward()
        return torch.cat([p.grad.flatten() for p in model.encoder.parameters()])

    torch.manual_seed(1)
    pred_off, sig_off = train_jepa.jepa_losses(model, sigreg, s, a, s_next, stop_grad=False)
    loss_off = pred_off + w * sig_off
    torch.manual_seed(1)
    pred_on, sig_on = train_jepa.jepa_losses(model, sigreg, s, a, s_next, stop_grad=True)
    loss_on = pred_on + w * sig_on
    assert torch.equal(loss_on, loss_off), (loss_on.item(), loss_off.item())
    print(f"(h1) total loss: stop_grad off {loss_off.item():.6f}, on {loss_on.item():.6f} (identical)")

    g_off = encoder_grads(loss_off)
    g_on = encoder_grads(loss_on)

    # reference: the target is computed in its own no_grad pass, so it is a constant
    with torch.no_grad():
        target = model.encode(s_next)
    torch.manual_seed(1)
    z = model.encode(s)
    z_next = model.encode(s_next)
    pred_ref = (model.predict(z, a) - target).square().mean()
    sig_ref = sigreg(torch.stack([z, z_next]))
    g_ref = encoder_grads(pred_ref + w * sig_ref)
    assert torch.equal(g_on, g_ref), (g_on - g_ref).abs().max().item()
    print(f"(h2) encoder grad with stop_grad on == with a no_grad constant target "
          f"(norm {g_on.norm().item():.4e}, max |diff| {(g_on - g_ref).abs().max().item()})")

    torch.manual_seed(1)
    _, sig_off = train_jepa.jepa_losses(model, sigreg, s, a, s_next, stop_grad=False)
    g_sig_off = encoder_grads(sig_off)
    torch.manual_seed(1)
    _, sig_on = train_jepa.jepa_losses(model, sigreg, s, a, s_next, stop_grad=True)
    g_sig_on = encoder_grads(sig_on)
    assert torch.equal(g_sig_on, g_sig_off)
    assert g_sig_on.norm().item() > 0
    print(f"(h3) encoder grad of SIGReg alone identical with stop_grad on and off "
          f"(norm {g_sig_on.norm().item():.4e})")

    assert not torch.equal(g_on, g_off)
    print(f"(h4) encoder grad of the total loss differs: off {g_off.norm().item():.4e}, "
          f"on {g_on.norm().item():.4e}, max |diff| {(g_on - g_off).abs().max().item():.4e}")


def test_window_pairs(data):
    with open(os.path.join(REPO_ROOT, "configs", "train_jepa.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    for split, n_expected in [("train", 226099), ("tuning", 28381)]:
        s, a, s_next = train_jepa.pairs(data, split)
        rows_1 = None
        for history in [1, 3, 10, 40]:
            states, rows, a_w = train_jepa.window_pairs(data, split, history)
            assert states.dtype == a_w.dtype == np.float32 and rows.dtype == np.int64
            assert len(rows) == n_expected and a_w.shape == (n_expected, 12), (split, history, len(rows))
            if rows_1 is None:
                rows_1 = rows
            assert np.array_equal(rows, rows_1)
            assert np.array_equal(states[rows], s) and np.array_equal(states[rows + 1], s_next)
            assert np.array_equal(a_w, a)
        print(f"(i) {split}: {n_expected} pairs at history 1, 3, 10 and 40, the same pairs as pairs()")
    for bad in [lambda: train_jepa.window_pairs(data, "train", 51),
                lambda: train_jepa.check_jepa_config(dict(cfg, stop_grad=False, history=51)),
                lambda: train_jepa.check_jepa_config(dict(cfg, stop_grad=False, history=0))]:
        try:
            bad()
        except AssertionError as e:
            print(f"    refused: {e}")
        else:
            raise AssertionError("history 51 or 0 was not refused")
    train_jepa.check_jepa_config(dict(cfg, stop_grad=False, history=50))


def test_windows(data):
    s_pairs, _, _ = train_jepa.pairs(data, "train")
    runs = np.flatnonzero(data["split"] == "train")[:3]
    for history in [10, 40]:
        states, rows, _ = train_jepa.window_pairs(data, "train", history)
        windows = train_jepa.gather_windows(torch.from_numpy(states), torch.from_numpy(rows), history).numpy()
        offset, n_checked, n_with_invalid = 0, 0, 0
        for run in runs:
            ks = np.flatnonzero(data["valid"][run])          # pairs are stored run by run, k ascending
            after_gap = np.flatnonzero(np.diff(ks) > 1) + 1  # first pair after a run of invalid steps
            for i in sorted({0, len(ks) // 2, len(ks) - 1, *after_gap.tolist()}):
                k = ks[i]
                hand = data["states"][run][k - history + 1:k + 1]      # rows k-n+1, ..., k of this run
                hand = ((hand - data["state_mean"]) / data["state_std"]).astype(np.float32)
                window = windows[offset + i]
                assert window.shape == (history, 34)
                assert np.array_equal(window, hand), (history, run, k)
                assert np.array_equal(window[-1], s_pairs[offset + i])
                n_checked += 1
                n_with_invalid += not data["valid"][run][k - history + 1:k].all()
            offset += len(ks)
        print(f"(j) history {history}: {n_checked} windows in the first {len(runs)} train runs equal the hand "
              f"normalized rows k-{history - 1}..k, last state = s_k "
              f"({n_with_invalid} of them span invalid steps)")


def test_residual_history():
    torch.manual_seed(0)
    model = JEPA(64, 1024, 3, history=10)
    assert model.predictor[0].in_features == 10 * 64 + 12
    z, a, z_past = torch.randn(100, 64), torch.randn(100, 12), torch.randn(100, 9, 64)
    by_hand = z + model.predictor(torch.cat([z_past[:, j] for j in range(9)] + [z, a], dim=1))
    z_hat = model.predict(z, a, z_past)
    assert torch.equal(z_hat, by_hand)
    with torch.no_grad():
        model.predictor[-1].weight.zero_()
        model.predictor[-1].bias.zero_()
    assert torch.equal(model.predict(z, a, z_past), z)
    print("(k) history 10: predict(z, a, z_past) == z + predictor([z_past oldest first, z, a]) exactly; "
          "with the last layer zeroed predict == z")


def test_loss_history(data):
    states, rows, a = (torch.from_numpy(x) for x in train_jepa.window_pairs(data, "train", 10))
    r = rows[:512]
    s = train_jepa.gather_windows(states, r, 10).requires_grad_()
    s_next, a = states[r + 1], a[:512]
    torch.manual_seed(0)
    model = JEPA(64, 1024, 3, history=10)
    sigreg = SIGReg()
    torch.manual_seed(1)
    pred_loss, sig = train_jepa.jepa_losses(model, sigreg, s, a, s_next)
    with torch.no_grad():
        z, z_next = model.encode(s[:, -1]), model.encode(s_next)
        z_past = torch.stack([model.encode(s[:, j]) for j in range(9)], dim=1)
        hand_pred = (model.predict(z, a, z_past) - z_next).square().mean()
        torch.manual_seed(1)
        hand_sig = sigreg(torch.stack([z, z_next]))
    assert torch.equal(sig, hand_sig), (sig.item(), hand_sig.item())
    assert torch.allclose(pred_loss, hand_pred, rtol=1e-5, atol=0), (pred_loss.item(), hand_pred.item())
    g_pred = torch.autograd.grad(pred_loss, s, retain_graph=True)[0]
    g_sig = torch.autograd.grad(sig, s)[0]
    assert g_pred[:, 0].norm().item() > 0
    assert torch.equal(g_sig[:, :-1], torch.zeros_like(g_sig[:, :-1])) and g_sig[:, -1].norm().item() > 0
    print(f"(l) history 10, 512 real pairs: SIGReg {sig.item():.4f} == SIGReg of z_k, z_k+1 alone; pred_loss "
          f"{pred_loss.item():.6f} vs hand {hand_pred.item():.6f}; grad of pred_loss on the oldest state "
          f"{g_pred[:, 0].norm().item():.3e}, grad of SIGReg on the older states exactly 0")


def test_history_weight_counts():
    counts = {}
    for history in [1, 3, 10, 40]:
        counts[history] = n_params(JEPA(256, 1024, 3, history=history).predictor)
        expected = (history * 256 + 12) * 1024 + 1024 + 2 * (1024 * 1024 + 1024) + 1024 * 256 + 256
        assert counts[history] == expected, (history, counts[history], expected)
    assert counts[1] == 2637056
    print("(m) predictor weights at embed_dim 256: "
          + ", ".join(f"history {h} {c:,}" for h, c in counts.items()))


def main():
    data = dict(np.load(DATASET))
    test_shapes()
    test_residual()
    test_collapse(data)
    test_no_stop_gradient(data)
    test_pairs(data)
    test_fixed_batch(data)
    test_determinism(data)
    test_stop_grad(data)
    test_window_pairs(data)
    test_windows(data)
    test_residual_history()
    test_loss_history(data)
    test_history_weight_counts()
    print("PASS test_jepa")


if __name__ == "__main__":
    main()
