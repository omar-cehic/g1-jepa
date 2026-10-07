"""Build the training dataset (states, actions, masks, split, normalization) from one raw batch.

Per run (50 Hz rows k = 0 .. 999, transitions k -> k+1 for k = 0 .. 998):
  states       (1000, 34) make_state(qpos, qvel), float64
  actions      (999, 12)  target_q[k]: the joint targets computed from state k and held
                          during transition k -> k+1 (raw float32 from the logger)
  contact_step (999,)     contact_steps(step, landings of both feet, 40 ms window)
  valid        (999,)     False if k < first_valid_k (the first 1 s), or if row k+1 is a push row
                          (the push is applied just before row k+1 is recorded, so transition
                          k -> k+1 contains it); True otherwise

Split: shuffle all run ids with split_seed, take n_train / n_tuning / n_test, then drop
drop_runs (the fall) from whichever split it landed in.

Normalization: mean and std (population std, ddof = 0) of s_k, a_k and
delta_k = s_{k+1} - s_k over the valid transitions of training runs only.
The stored states and actions are raw, not normalized.

Usage: python scripts/build_dataset.py configs/build_dataset.yaml
Output: data/processed/<YYYY-MM-DD_HHMMSS>/ with dataset.npz, build_dataset.yaml, meta.json.
"""

import argparse
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from contact import detect_landings, contact_steps  # noqa: E402
from features import make_state, JOINT_NAMES, STATE_NAMES  # noqa: E402

ACTION_NAMES = ["target_q_" + j for j in JOINT_NAMES]
SPLITS = ["train", "tuning", "test"]


def removal_reasons(d, cfg):
    """Why transitions k -> k+1 of one raw run are removed, as two boolean arrays (n_rows - 1,).

    early: k < first_valid_k (the first 1 s)
    push:  row k+1 is a push row
    """
    n_transitions = len(d["t"]) - 1
    early = np.zeros(n_transitions, dtype=bool)
    early[: cfg["first_valid_k"]] = True
    push = np.zeros(n_transitions, dtype=bool)
    for k in d["push_row"]:
        if k > 0:
            push[k - 1] = True  # push at row k is inside transition k-1 -> k
    return early, push


def load_run(path, cfg):
    """Return states, actions, valid, contact_step for one raw run file."""
    d = np.load(path)
    n_rows = len(d["t"])
    assert n_rows == 1000 and not d["fell"], f"{path}: {n_rows} rows, fell = {d['fell']}"

    states, _ = make_state(d["qpos"], d["qvel"])
    actions = d["target_q"][:-1]  # transition k uses target_q[k]; the last row has no next state

    early, push = removal_reasons(d, cfg)
    valid = ~(early | push)

    t_fine, step = d["t_fine"], d["step"]
    dt = t_fine[1] - t_fine[0]
    landings = detect_landings(d["foot_contact"], t_fine)  # one array per foot
    window = round(cfg["contact_window_s"] / dt)
    contact = contact_steps(step, np.concatenate(landings), window)
    return states, actions, valid, contact


def split_runs(run_ids, cfg):
    """Split label per run id, as a dict {run id: "train" / "tuning" / "test"}."""
    rng = np.random.default_rng(cfg["split_seed"])
    shuffled = rng.permutation(run_ids)
    n_train, n_tuning = cfg["n_train"], cfg["n_tuning"]
    assert n_train + n_tuning + cfg["n_test"] == len(run_ids)
    split = {}
    for i, run in enumerate(shuffled):
        if i < n_train:
            split[int(run)] = "train"
        elif i < n_train + n_tuning:
            split[int(run)] = "tuning"
        else:
            split[int(run)] = "test"
    return split


def norm_stats(states, actions, valid):
    """Mean and std of s_k, a_k, delta_k over the transitions where valid is True."""
    s = states[:, :-1][valid]                       # (n_valid, 34)
    a = actions[valid]                              # (n_valid, 12)
    delta = (states[:, 1:] - states[:, :-1])[valid]  # (n_valid, 34)
    stats = {}
    for name, x in [("state", s), ("action", a), ("delta", delta)]:
        x = x.astype(np.float64)
        stats[name + "_mean"] = x.mean(axis=0)
        stats[name + "_std"] = x.std(axis=0)
    return stats


def build(cfg):
    """Load the raw runs, split them, and compute the stats. Returns a dict of arrays."""
    raw_dir = os.path.join(REPO_ROOT, cfg["raw_dir"])
    paths = sorted(glob.glob(os.path.join(raw_dir, "run_*.npz")))
    all_ids = [int(os.path.basename(p)[4:7]) for p in paths]
    assert all_ids == list(range(len(paths))), "run files are not run_000 .. run_N"

    split = split_runs(all_ids, cfg)
    for run in cfg["drop_runs"]:
        print(f"dropping run {run} from the {split[run]} split")
        del split[run]

    run_ids = np.array(sorted(split))
    states, actions, valid, contact = [], [], [], []
    for run in run_ids:
        s, a, v, c = load_run(paths[run], cfg)
        states.append(s)
        actions.append(a)
        valid.append(v)
        contact.append(c)

    data = {
        "run_ids": run_ids,
        "split": np.array([split[run] for run in run_ids]),
        "states": np.stack(states),
        "actions": np.stack(actions),
        "valid": np.stack(valid),
        "contact_step": np.stack(contact),
    }
    train = data["split"] == "train"
    stats = norm_stats(data["states"][train], data["actions"][train], data["valid"][train])
    for name, std in stats.items():
        if name.endswith("_std") and np.any(std < 1e-8):
            print(f"WARNING: {name} < 1e-8 at features {np.flatnonzero(std < 1e-8).tolist()}, set to 1")
            std[std < 1e-8] = 1.0
    data.update(stats)
    return data


def removal_counts(cfg):
    """Per split, how many transitions are removed for each reason, for meta.json.

    The split is the one before drop_runs are removed, so a dropped run counts in the
    split it was drawn into. Each reason is counted on its own over every transition it
    matches (so first_1s and push also count the dropped runs' transitions);
    more_than_one_reason counts transitions that match two or three reasons, and
    total_removed counts each removed transition once.
    Also returns the split each dropped run was in.
    """
    raw_dir = os.path.join(REPO_ROOT, cfg["raw_dir"])
    paths = sorted(glob.glob(os.path.join(raw_dir, "run_*.npz")))
    split = split_runs(list(range(len(paths))), cfg)
    keys = ["first_1s", "push", "dropped_runs", "more_than_one_reason", "total_removed"]
    counts = {name: {key: 0 for key in keys} for name in SPLITS}
    for run, path in enumerate(paths):
        early, push = removal_reasons(np.load(path), cfg)
        dropped = np.full(len(early), run in cfg["drop_runs"])
        n_reasons = early.astype(int) + push + dropped
        c = counts[split[run]]
        c["first_1s"] += int(early.sum())
        c["push"] += int(push.sum())
        c["dropped_runs"] += int(dropped.sum())
        c["more_than_one_reason"] += int((n_reasons > 1).sum())
        c["total_removed"] += int((n_reasons > 0).sum())
    dropped_split = {str(run): split[run] for run in cfg["drop_runs"]}
    return counts, dropped_split


def git_commit():
    """HEAD of this repo, and whether there are uncommitted changes."""
    def git(*args):
        return subprocess.run(["git", "-C", REPO_ROOT, *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    return git("rev-parse", "HEAD"), git("status", "--porcelain") != ""


def report(data):
    print()
    print(f"{'split':8s} {'runs':>5s} {'valid transitions':>18s} {'contact-step fraction':>22s}")
    for name in SPLITS:
        m = data["split"] == name
        v = data["valid"][m]
        frac = data["contact_step"][m][v].mean()
        print(f"{name:8s} {m.sum():5d} {v.sum():18d} {frac:22.4f}")

    for kind, names in [("state", STATE_NAMES), ("action", ACTION_NAMES), ("delta", STATE_NAMES)]:
        print()
        print(f"{kind} (train, valid transitions)   {'mean':>12s} {'std':>12s}")
        for i, name in enumerate(names):
            print(f"  {name:30s} {data[kind + '_mean'][i]:12.5f} {data[kind + '_std'][i]:12.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to build_dataset.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data = build(cfg)

    out_dir = os.path.join(REPO_ROOT, cfg["out_dir"], datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    os.makedirs(out_dir)  # raises if the folder already exists: never overwrite
    np.savez(os.path.join(out_dir, "dataset.npz"), **data)
    shutil.copy(args.config, os.path.join(out_dir, "build_dataset.yaml"))
    commit, dirty = git_commit()
    removed, dropped_split = removal_counts(cfg)
    for name in SPLITS:
        # Check against the saved mask: in kept runs, removed = all transitions - valid ones.
        in_split = data["split"] == name
        kept_removed = removed[name]["total_removed"] - removed[name]["dropped_runs"]
        assert kept_removed == data["valid"][in_split].size - data["valid"][in_split].sum(), name
    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_raw_dir": cfg["raw_dir"],
        "git_commit": commit,
        "git_uncommitted_changes": dirty,
        "split_seed": cfg["split_seed"],
        "dropped_runs": cfg["drop_runs"],
        "dropped_run_was_in_split": dropped_split,
        "runs_per_split": {name: int((data["split"] == name).sum()) for name in SPLITS},
        "run_ids_per_split": {name: data["run_ids"][data["split"] == name].tolist() for name in SPLITS},
        "valid_transitions_per_split": {name: int(data["valid"][data["split"] == name].sum())
                                        for name in SPLITS},
        "removed_transitions_per_split": removed,
        "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    report(data)
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
