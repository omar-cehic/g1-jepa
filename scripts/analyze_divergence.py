"""Per window analysis of rollout errors: are they large everywhere, or only in a few windows?

Takes one folder of scripts/eval_rollouts.py. Its meta.json says which model and split; the same
model is rolled out again on every usable window of that split (H = 100), one run at a time, and
the per window errors are kept instead of only their RMS over all windows.

For each state group:
  a. Per window group RMS error at j = 10, 50, 100 (square root of the mean over the group's
     features of the squared error): median, 90th and 99th percentile, max, and the RMS over all
     windows. The last one is asserted to equal rms_all_<group> in the folder's metrics.npz.
  b. Train range: min and max of each feature over the states of the valid transitions of the
     train runs (s_k and s_{k+1}). For each window, the first step j at which a predicted feature
     of the group is outside that range (or is not finite). Fraction of windows that have left by
     j = 10, 50, 100, and the median first exit step of the windows that leave. The same for the
     true states, as a reference: the tuning runs themselves can leave the train range.
     Group "any" uses all 34 features.
  c. Share of the total squared error at j = 100 (summed over the group's features) that comes
     from the worst 1% and the worst 10% of windows.
  d. windows.png: the median window and the 3 worst windows at j = 100 by height error, true vs
     predicted pelvis height and dq_left_knee over the 100 steps.

Usage: python scripts/analyze_divergence.py results/rollouts/<folder> --label <label>
Output: results/analysis/<YYYY-MM-DD_HHMMSS>_<label>/ with report.txt (the printed output),
        per_window.npz, windows.png and meta.json.
"""

import argparse
import datetime
import json
import math
import os
import resource
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from build_dataset import git_commit  # noqa: E402
from eval_rollouts import load_mlp_step, DATASET, MODELS, H  # noqa: E402
from eval.rollout import usable_starts, rollout, train_range, first_exit  # noqa: E402
from features import STATE_NAMES, STATE_GROUPS  # noqa: E402

REPORT_STEPS = [10, 50, 100]
J_IDX = [j - 1 for j in REPORT_STEPS]  # prediction j is at index j - 1
GROUPS_ANY = dict(STATE_GROUPS, any=slice(0, len(STATE_NAMES)))
KNEE = STATE_NAMES.index("dq_left_knee")
HEIGHT = STATE_NAMES.index("pelvis_height")


def top_share(x, fraction):
    """Share of sum(x) that comes from the largest ceil(fraction * len(x)) values of x (x >= 0)."""
    n = max(1, math.ceil(fraction * len(x)))
    return np.sort(x)[::-1][:n].sum() / x.sum()


def windows_of(data, run, starts):
    """s0 (W, 34), actions (W, H, 12) and truth (W, H, 34) of the windows starting at starts."""
    rows = starts[:, None] + np.arange(H)
    return data["states"][run, starts], data["actions"][run, rows], data["states"][run, rows + 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_dir", help="results/rollouts/<folder> written by scripts/eval_rollouts.py")
    parser.add_argument("--label", required=True, help="name for the output folder")
    args = parser.parse_args()

    rollout_dir = os.path.join(REPO_ROOT, args.rollout_dir)
    with open(os.path.join(rollout_dir, "meta.json"), "r") as f:
        rollout_meta = json.load(f)
    assert rollout_meta["dataset"] == DATASET and rollout_meta["H"] == H, rollout_meta
    reference = np.load(os.path.join(rollout_dir, "metrics.npz"))
    split = rollout_meta["split"]

    data = dict(np.load(os.path.join(REPO_ROOT, DATASET)))
    if rollout_meta["model"] == "mlp":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        step = load_mlp_step(os.path.join(REPO_ROOT, rollout_meta["run_dir"]), data, device)
    else:
        step = MODELS[rollout_meta["model"]]
    lo, hi = train_range(data)

    # Per window results, one list entry per run (concatenated after the loop).
    window_run, window_start = [], []
    group_mse = {g: [] for g in STATE_GROUPS}    # (W, 3): mean over the group's features of err^2 at j = 10, 50, 100
    exit_pred = {g: [] for g in GROUPS_ANY}      # (W,): first step out of the train range, H + 1 if never
    exit_true = {g: [] for g in GROUPS_ANY}
    for run in np.flatnonzero(data["split"] == split):
        starts = usable_starts(data["valid"][run], H)
        if len(starts) == 0:
            continue
        s0, actions, truth = windows_of(data, run, starts)
        preds = rollout(step, s0, actions)                               # (W, H, 34)
        sq_err = (preds - truth) ** 2
        out_pred = ~np.isfinite(preds) | (preds < lo) | (preds > hi)     # NaN compares False, so check it
        out_true = (truth < lo) | (truth > hi)
        for group, cols in GROUPS_ANY.items():
            if group in group_mse:
                group_mse[group].append(sq_err[:, J_IDX, cols].mean(axis=2))
            exit_pred[group].append(first_exit(out_pred[:, :, cols].any(axis=2)))
            exit_true[group].append(first_exit(out_true[:, :, cols].any(axis=2)))
        window_run.append(np.full(len(starts), run))
        window_start.append(starts)

    window_run, window_start = np.concatenate(window_run), np.concatenate(window_start)
    group_mse = {g: np.concatenate(v) for g, v in group_mse.items()}
    exit_pred = {g: np.concatenate(v) for g, v in exit_pred.items()}
    exit_true = {g: np.concatenate(v) for g, v in exit_true.items()}
    n_windows = len(window_run)
    assert n_windows == reference["n_windows"], (n_windows, reference["n_windows"])

    lines = []

    def out(line=""):
        print(line)
        lines.append(line)

    name = rollout_meta["model"] + (f" {rollout_meta['run_dir']}" if "run_dir" in rollout_meta else "")
    out(f"model {name}, split {split}, H = {H}, {n_windows} windows, errors in raw units")
    out(f"reference: {args.rollout_dir}")

    # a. Per window group RMS, and the RMS over all windows (checked against eval_rollouts.py).
    max_rel = 0.0
    out()
    out("a. per window group RMS error")
    out(f"{'group':10s} {'':16s}" + "".join(f"{'j = ' + str(j):>12s}" for j in REPORT_STEPS))
    for group, mse in group_mse.items():
        per_window = np.sqrt(mse)                                  # (W, 3)
        rms_all = np.sqrt(mse.mean(axis=0))                        # (3,)
        expected = reference[f"rms_all_{group}"][J_IDX]
        assert np.allclose(rms_all, expected, rtol=1e-6, atol=0), (group, rms_all, expected)
        with np.errstate(invalid="ignore"):  # inf - inf if a rollout diverged
            max_rel = max(max_rel, np.nanmax(np.abs(rms_all / expected - 1)))
        rows = [("median", np.median(per_window, axis=0)),
                ("90th pct", np.percentile(per_window, 90, axis=0)),
                ("99th pct", np.percentile(per_window, 99, axis=0)),
                ("max", per_window.max(axis=0)),
                ("RMS all windows", rms_all)]
        for i, (stat, values) in enumerate(rows):
            out(f"{group if i == 0 else '':10s} {stat:16s}" + "".join(f"{v:12.4e}" for v in values))
    out(f"RMS all windows matches eval_rollouts.py metrics.npz (max relative difference {max_rel:.1e})")

    # b. Leaving the train range.
    out()
    out("b. windows that have left the train range (min / max over valid train transitions)")
    out(f"{'group':10s} {'':6s}" + "".join(f"{'by j = ' + str(j):>12s}" for j in REPORT_STEPS)
        + f"{'median exit':>13s}")
    for group in GROUPS_ANY:
        for i, (who, first) in enumerate([("pred", exit_pred[group]), ("true", exit_true[group])]):
            fractions = "".join(f"{np.mean(first <= j):12.4f}" for j in REPORT_STEPS)
            leavers = first[first <= H]
            median = f"{np.median(leavers):13.1f}" if len(leavers) else f"{'-':>13s}"
            out(f"{group if i == 0 else '':10s} {who:6s}{fractions}{median}")

    # c. How concentrated the squared error at j = 100 is.
    out()
    out("c. share of the total squared error at j = 100 from the worst windows")
    out(f"{'group':10s} {'worst 1%':>10s} {'worst 10%':>10s}")
    for group, cols in STATE_GROUPS.items():
        sse = group_mse[group][:, -1] * (cols.stop - cols.start)   # sum over the group's features
        out(f"{group:10s} {top_share(sse, 0.01):10.3f} {top_share(sse, 0.10):10.3f}")

    # d. Median window and the 3 worst at j = 100 by height error.
    height_err = np.sqrt(group_mse["height"][:, -1])
    order = np.argsort(height_err)
    picked = [("median", order[n_windows // 2])] + [(f"worst {i + 1}", order[-1 - i]) for i in range(3)]
    fig, axes = plt.subplots(len(picked), 2, figsize=(11, 2.6 * len(picked)), sharex=True)
    j = np.arange(H + 1)  # j = 0 is the start state
    for row, (title, w) in enumerate(picked):
        run, start = window_run[w], window_start[w]
        s0, actions, truth = windows_of(data, run, np.array([start]))
        preds = rollout(step, s0, actions)
        for col, (feature, unit) in enumerate([(HEIGHT, "m"), (KNEE, "rad/s")]):
            ax = axes[row, col]
            ax.plot(j, np.r_[s0[0, feature], truth[0, :, feature]], color="#2a78d6", linewidth=2, label="true")
            ax.plot(j, np.r_[s0[0, feature], preds[0, :, feature]], color="#eb6834", linewidth=2,
                    linestyle="--", label="predicted")
            ax.set_ylabel(f"{STATE_NAMES[feature]} ({unit})", fontsize=8)
            ax.set_title(f"{title}: run {data['run_ids'][run]}, k = {start}, height error at j = 100:"
                         f" {height_err[w]:.3g} m", fontsize=8)
            ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("j (steps after the start state, 20 ms each)")
    fig.tight_layout()

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "results", "analysis", f"{stamp}_{args.label}")
    os.makedirs(out_dir)  # raises if the folder already exists: never overwrite
    fig.savefig(os.path.join(out_dir, "windows.png"), dpi=130)
    plt.close(fig)
    np.savez(os.path.join(out_dir, "per_window.npz"), window_run=window_run, window_start=window_start,
             report_steps=np.array(REPORT_STEPS), train_min=lo, train_max=hi,
             **{f"mse_{g}": v for g, v in group_mse.items()},
             **{f"exit_pred_{g}": v for g, v in exit_pred.items()},
             **{f"exit_true_{g}": v for g, v in exit_true.items()})
    commit, dirty = git_commit()
    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "rollout_dir": args.rollout_dir,
        "model": rollout_meta["model"],
        "run_dir": rollout_meta.get("run_dir"),
        "split": split,
        "H": H,
        "n_windows": n_windows,
        "git_commit": commit,
        "git_uncommitted_changes": dirty,
        "picked_windows": [{"title": t, "run_index": int(window_run[w]), "start": int(window_start[w])}
                           for t, w in picked],
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # ru_maxrss is in kB on Linux
    out()
    out(f"wrote {out_dir}")
    out(f"peak memory {peak_mb:.0f} MB")
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
