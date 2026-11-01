"""Tests for the transformer world model (model "transformer" in src/train.py) and its one step error in
scripts/readout_controls.py.

Uses data/processed/2026-10-07_182522 (base state, state_dim 34).

(a) Weight count at width 1024, depth 3, history 10: equals the hand count (token Linear, position
    vectors, 3 blocks of attention + feedforward + 2 LayerNorms, final LayerNorm, final Linear 256 -> 34),
    printed next to the MLP's (make_mlp(46, 34, 1024, 3)).
(b) Residual form: with the final Linear zeroed, the network output (the normalized delta) is exactly
    0 on random windows, so the predicted next state s_k + denormalize(0, "delta") is exactly
    s_k + delta_mean, the same as the MLP with a zero output (test_mlp.py (c)). In normalized delta
    units, output 0 means "the average change"; delta_mean itself is printed relative to the state std.
(c) Swapping window steps 0 and 1 (both s_j and a_j) changes the output; with the position vectors
    zeroed the change is only float rounding.
(d) Pairs: at history 1, 3 and 10, window_transitions() gives the same pairs as transitions():
    states[rows] and actions[rows] equal the s_k and a_k parts of the MLP input exactly, and the
    targets are identical. On the first 3 tuning runs, every gathered window equals the hand
    normalized stored states and actions at steps k-n+1, ..., k of the same run.
(e) One step error: on the first 3 tuning runs, transformer_one_step_mse() of a random transformer
    (history 10) equals a hand computation (windows sliced run by run from the raw arrays, s_k +
    denormalized output, normalized error) to 1e-6 relative; with the final Linear zeroed it equals
    the error of predicting s_k + delta_mean, to 1e-12 relative.
(f) check_world_model_config() refuses history 3 with the MLP, history 0 and 51, and model "gru";
    it accepts mlp at history 1 and transformer at history 1, 10 and 50.

Nothing is written. Run: python tests/test_transformer_world_model.py
"""

import os
import sys

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from models.mlp import make_mlp, normalize, denormalize  # noqa: E402
from readout_controls import transformer_one_step_mse  # noqa: E402
import train  # noqa: E402
from train_jepa import gather_windows  # noqa: E402

DATASET = os.path.join(REPO_ROOT, "data", "processed", "2026-10-07_182522", "dataset.npz")
CPU = torch.device("cpu")
STATE_DIM = 34


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def first_tuning_runs(data, n_runs=3):
    """dataset dict with only the first n_runs tuning runs (and the stats)."""
    runs = np.flatnonzero(data["split"] == "tuning")[:n_runs]
    small = dict(data)
    for key in ["states", "actions", "valid", "split"]:
        small[key] = data[key][runs]
    return small


def test_weight_counts():
    d, w = train.D_MODEL, 1024
    block = (3 * d * d + 3 * d) + (d * d + d) + (d * w + w) + (w * d + d) + 4 * d   # attention, ff, 2 LayerNorms
    expected = ((STATE_DIM + 12) * d + d) + 10 * d + 3 * block + 2 * d + (d * STATE_DIM + STATE_DIM)
    count = n_params(train.make_model("transformer", STATE_DIM, w, 3, 10))
    mlp = n_params(train.make_model("mlp", STATE_DIM, w, 3, 1))
    assert count == expected, (count, expected)
    assert mlp == n_params(make_mlp(STATE_DIM + 12, STATE_DIM, w, 3))
    print(f"(a) weights at width 1024, depth 3, state_dim 34: transformer (history 10) {count:,}, "
          f"MLP {mlp:,}")


def test_residual(data):
    torch.manual_seed(0)
    model = train.make_model("transformer", STATE_DIM, 1024, 3, 10)
    with torch.no_grad():
        model.out.weight.zero_()
        model.out.bias.zero_()
        out = model(torch.randn(100, 10, STATE_DIM), torch.randn(100, 10, 12))
    assert torch.equal(out, torch.zeros(100, STATE_DIM))
    s = data["states"][0, 50:150]                                   # 100 raw states
    pred = s + denormalize(out.numpy(), data, "delta")
    assert np.array_equal(pred, s + data["delta_mean"])
    rel = np.abs(data["delta_mean"]) / data["state_std"]
    print(f"(b) final Linear zeroed: output exactly 0, predicted next state exactly s_k + delta_mean "
          f"(max |delta_mean| / state_std {rel.max():.1e}, feature {int(rel.argmax())})")


def test_positions():
    torch.manual_seed(0)
    model = train.make_model("transformer", STATE_DIM, 1024, 3, 10)
    s, a = torch.randn(100, 10, STATE_DIM), torch.randn(100, 10, 12)
    swap = [1, 0] + list(range(2, 10))                    # swap window steps 0 and 1 (s_j and a_j together)
    with torch.no_grad():
        diff = (model(s, a) - model(s[:, swap], a[:, swap])).abs().max().item()
        model.position.zero_()
        control = (model(s, a) - model(s[:, swap], a[:, swap])).abs().max().item()
    assert diff > 1e-3, diff
    assert control < 1e-5, control
    print(f"(c) swapping window steps 0 and 1: max |change| {diff:.3e}; with the position vectors zeroed "
          f"{control:.1e} (float rounding)")


def test_pairs(data):
    for split in ["train", "tuning"]:
        x_mlp, y_mlp = train.transitions(data, split)
        for history in [1, 3, 10]:
            (states, actions, rows), y = train.window_transitions(data, split, history)
            assert np.array_equal(states[rows], x_mlp[:, :STATE_DIM])
            assert np.array_equal(actions[rows], x_mlp[:, STATE_DIM:])
            assert np.array_equal(y, y_mlp)
    small = first_tuning_runs(data)
    history = 10
    (states, actions, rows), _ = train.window_transitions(small, "tuning", history)
    s_win = gather_windows(torch.from_numpy(states), torch.from_numpy(rows), history).numpy()
    a_win = gather_windows(torch.from_numpy(actions), torch.from_numpy(rows), history).numpy()
    run, k = np.nonzero(small["valid"])
    for j in range(len(rows)):
        steps = slice(k[j] - history + 1, k[j] + 1)
        hand_s = normalize(small["states"][run[j], steps], small, "state").astype(np.float32)
        hand_a = normalize(small["actions"][run[j], steps], small, "action").astype(np.float32)
        assert np.array_equal(s_win[j], hand_s) and np.array_equal(a_win[j], hand_a), j
    print(f"(d) history 1, 3, 10: window_transitions() pairs, s_k, a_k and targets == transitions() exactly "
          f"(train and tuning); {len(rows)} history 10 windows of the first 3 tuning runs == hand slices")


def hand_one_step_mse(model, data, history):
    """The one step error of transformer_one_step_mse, computed run by run with windows sliced by hand."""
    pairs = []
    for run in range(len(data["states"])):
        for k in np.flatnonzero(data["valid"][run]):
            steps = slice(k - history + 1, k + 1)
            s_win = normalize(data["states"][run, steps], data, "state").astype(np.float32)
            a_win = normalize(data["actions"][run, steps], data, "action").astype(np.float32)
            pairs.append((run, k, s_win, a_win))
    s_win = torch.from_numpy(np.stack([e[2] for e in pairs]))
    a_win = torch.from_numpy(np.stack([e[3] for e in pairs]))
    with torch.no_grad():
        out = model(s_win, a_win).numpy()
    s = np.stack([data["states"][e[0], e[1]] for e in pairs])
    s_next = np.stack([data["states"][e[0], e[1] + 1] for e in pairs])
    pred = s + denormalize(out, data, "delta")
    return np.mean((normalize(pred, data, "state") - normalize(s_next, data, "state")) ** 2, axis=0)


def test_one_step(data):
    small = first_tuning_runs(data)
    torch.manual_seed(0)
    model = train.make_model("transformer", STATE_DIM, 1024, 3, 10).eval()
    got = transformer_one_step_mse(model, small, "tuning", CPU)
    hand = hand_one_step_mse(model, small, 10)
    rel = np.abs(got - hand) / hand
    assert got.shape == (STATE_DIM,) and rel.max() < 1e-6, rel.max()
    with torch.no_grad():
        model.out.weight.zero_()
        model.out.bias.zero_()
    got_zero = transformer_one_step_mse(model, small, "tuning", CPU)
    valid = small["valid"]
    s, s_next = small["states"][:, :-1][valid], small["states"][:, 1:][valid]
    expected = np.mean((normalize(s + small["delta_mean"], small, "state") - normalize(s_next, small, "state")) ** 2,
                       axis=0)
    rel_zero = np.abs(got_zero - expected) / expected
    assert rel_zero.max() < 1e-12, rel_zero.max()
    print(f"(e) one step error on {valid.sum()} transitions of the first 3 tuning runs: random transformer vs "
          f"hand windows max relative diff {rel.max():.1e} (mean error {got.mean():.3f}); zeroed final Linear "
          f"vs s_k + delta_mean {rel_zero.max():.1e}")


def test_config_checks():
    with open(os.path.join(REPO_ROOT, "configs", "train_mlp.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    for model, history in [("mlp", 1), ("transformer", 1), ("transformer", 10), ("transformer", 50)]:
        train.check_world_model_config(dict(cfg, model=model, history=history))
    refused = 0
    for model, history in [("mlp", 3), ("transformer", 0), ("transformer", 51), ("gru", 1)]:
        try:
            train.check_world_model_config(dict(cfg, model=model, history=history))
        except AssertionError:
            refused += 1
    assert refused == 4, refused
    print("(f) check_world_model_config refuses mlp/3, transformer/0, transformer/51, gru/1; "
          "accepts mlp/1, transformer/1, 10, 50")


def main():
    data = dict(np.load(DATASET))
    test_weight_counts()
    test_residual(data)
    test_positions()
    test_pairs(data)
    test_one_step(data)
    test_config_checks()
    print("PASS test_transformer_world_model")


if __name__ == "__main__":
    main()
