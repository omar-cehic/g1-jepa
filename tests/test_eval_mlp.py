"""Tests for the MLP path of scripts/eval_rollouts.py (load_mlp_step).

Uses data/processed/2026-10-07_182522. The saved runs go to temporary folders (tempfile),
never to data/ or results/.

(a) A random MLP (make_mlp, seed 0, width 32, depth 2: not the config's 256 x 3, so the
    saved width and depth must really be used) is saved the way src/train.py saves it:
    config.yaml (configs/train_mlp.yaml with width and depth changed, yaml.safe_dump) and
    model.pt ({"state_dict" on the CPU, "width", "depth"}).
(b) load_mlp_step on that folder gives exactly the same outputs as make_step on the
    in-memory model, on the valid transitions of the first 3 tuning runs, on the CPU
    (and also on cuda, if available).
(c) On the first 3 tuning runs, evaluate()'s RMS at j = 1 (all / contact / rest, per feature
    and per group) equals the RMS of step(s_k, a_k) - s_{k+1} computed directly over the same
    usable starts, to 1e-12 relative.
(d) The same folder with config.yaml pointing to another dataset is rejected (ValueError).

Nothing is written outside the temporary folders. Run: python tests/test_eval_mlp.py
"""

import copy
import os
import sys
import tempfile

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from eval_rollouts import load_mlp_step, DATASET  # noqa: E402
from eval.rollout import usable_starts, evaluate  # noqa: E402
from features import STATE_GROUPS  # noqa: E402
from models.mlp import make_mlp, make_step  # noqa: E402

H = 100
N_RUNS = 3
WIDTH, DEPTH = 32, 2
OTHER_DATASET = "data/processed/2026-10-07_174249/dataset.npz"


def save_run(run_dir, model, dataset):
    """Save model and a config in the format of src/train.py."""
    with open(os.path.join(REPO_ROOT, "configs", "train_mlp.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    cfg.update(dataset=dataset, width=WIDTH, depth=DEPTH)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state_dict, "width": WIDTH, "depth": DEPTH}, os.path.join(run_dir, "model.pt"))


def subset(data, runs):
    """dataset dict with only the given run indices (and the stats)."""
    keys = ["states", "actions", "valid", "contact_step", "split"]
    out = {key: data[key][runs] for key in keys}
    for name in ["state", "action", "delta"]:
        out[name + "_mean"], out[name + "_std"] = data[name + "_mean"], data[name + "_std"]
    return out


def test_same_outputs(run_dir, model, data, devices):
    valid = data["valid"]
    s, a = data["states"][:, :-1][valid], data["actions"][valid]
    for device in devices:
        loaded = load_mlp_step(run_dir, data, device)(s, a)
        in_memory = make_step(copy.deepcopy(model).to(device), data, device)(s, a)
        assert loaded.dtype == np.float64
        assert np.array_equal(loaded, in_memory), np.abs(loaded - in_memory).max()
    print(f"(b) load_mlp_step == in-memory model exactly on {len(s)} tuning transitions, "
          f"devices {[str(d) for d in devices]}")


def test_one_step_rms(run_dir, data, device):
    step = load_mlp_step(run_dir, data, device)
    err1, contact1 = [], []
    for run in range(N_RUNS):
        # Same batches as evaluate (one per run), so the network sees exactly the same inputs.
        starts = usable_starts(data["valid"][run], H)
        err1.append(step(data["states"][run, starts], data["actions"][run, starts]) - data["states"][run, starts + 1])
        contact1.append(data["contact_step"][run, starts])
    err1, contact1 = np.concatenate(err1), np.concatenate(contact1)

    m = evaluate(step, data, "tuning", H)
    worst = 0.0
    for subset_name, mask in [("all", np.ones_like(contact1)), ("contact", contact1), ("rest", ~contact1)]:
        assert m["count_" + subset_name][0] == mask.sum()
        direct_rms = np.sqrt(np.mean(err1[mask] ** 2, axis=0))
        rel = np.abs(m["rms_" + subset_name][0] - direct_rms) / direct_rms
        worst = max(worst, rel.max())
        for group, cols in STATE_GROUPS.items():
            direct_group = np.sqrt(np.mean(err1[mask][:, cols] ** 2))
            got = m[f"rms_{subset_name}_{group}"][0]
            worst = max(worst, abs(got - direct_group) / direct_group)
    assert worst < 1e-12, worst
    print(f"(c) j = 1 on {device}: evaluate RMS vs direct step(s_k, a_k) - s_(k+1): "
          f"max relative diff {worst:.1e} over {m['count_all'][0]} windows")


def test_wrong_dataset(model, data):
    with tempfile.TemporaryDirectory() as run_dir:
        save_run(run_dir, model, OTHER_DATASET)
        try:
            load_mlp_step(run_dir, data, torch.device("cpu"))
        except ValueError as e:
            print(f"(d) config.yaml with dataset {OTHER_DATASET} rejected: {e}")
            return
    raise AssertionError("a run trained on another dataset was not rejected")


def main():
    full = dict(np.load(os.path.join(REPO_ROOT, DATASET)))
    tuning_runs = np.flatnonzero(full["split"] == "tuning")[:N_RUNS]
    data = subset(full, tuning_runs)
    print(f"first {N_RUNS} tuning runs: run ids {full['run_ids'][tuning_runs].tolist()}")
    del full

    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    torch.manual_seed(0)
    model = make_mlp(46, 34, WIDTH, DEPTH)
    with tempfile.TemporaryDirectory() as run_dir:
        save_run(run_dir, model, DATASET)
        print(f"(a) random MLP {WIDTH} x {DEPTH} saved to {run_dir}: {sorted(os.listdir(run_dir))}")
        test_same_outputs(run_dir, model, data, devices)
        test_one_step_rms(run_dir, data, devices[-1])
    test_wrong_dataset(model, data)
    print("PASS test_eval_mlp")


if __name__ == "__main__":
    main()
