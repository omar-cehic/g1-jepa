"""Explore foot contact in a data folder, to help choose a landing definition.

Usage: python scripts/explore_contact.py data/raw/<folder>
Writes plots and report.txt to a new folder results/<YYYY-MM-DD_HHMMSS>_explore_contact/.

Exclusions (as asked): the first 1.0 s of every run, and anything that
contains a push.
  - 500 Hz segments: kept only if they start at or after 1.0 s with an
    observed change of the label, end before the run ends (so the full
    duration is known), and no push happens inside them.
  - Landings: the 50 ms checked before and after must lie after 1.0 s and
    contain no push.
  - 50 Hz transitions k -> k+1: kept if t_k >= 1.0 s and row k+1 is not a
    push row (the push is applied just before row k+1 is recorded).

Index convention: 500 Hz label i describes the state at t_fine[i] = i * dt,
and 50 Hz row k is the state at 500 Hz index step[k] = 10 (k + 1).

Parts:
  (a) durations of every contact-on and contact-off segment, per foot
  (b) landings as defined in src/contact.py (detect_landings): a False -> True
      change of the label where the foot was off for >= 50 ms before and
      stays on for >= 50 ms after
  (c) how much the 34-dim state (src/features.py) changes per 50 Hz
      transition around those landings
"""

import argparse
import datetime
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from contact import detect_landings, contact_steps, transition_zero  # noqa: E402
from features import make_state, STATE_GROUPS  # noqa: E402

SKIP_START_S = 1.0
SHORT_S = 0.010             # "short" segment: duration < 10 ms
BOUNCE_GAP_S = 0.010        # a short on-segment within 10 ms of a long one counts as part of its touchdown/lift-off
LANDING_OFF_S = 0.05        # landing windows (the src/contact.py defaults)
LANDING_ON_S = 0.05
CONTACT_WINDOW_S = 0.04     # contact step: a landing within 40 ms of the transition's time interval
ALIGN = 10                  # transitions before/after the landing in the aligned plot

FEET = ["left", "right"]
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def segments(label):
    """Split a boolean array into constant runs: (start, end exclusive, value) arrays."""
    edges = np.flatnonzero(label[1:] != label[:-1]) + 1
    start = np.concatenate([[0], edges])
    end = np.concatenate([edges, [len(label)]])
    return start, end, label[start]


def exclude_landings(land, first, push_idx, n_off, n_on):
    """Drop landings whose windows [i - n_off, i + n_on] start before index first or contain a push."""
    keep = [i for i in land
            if i - n_off >= first and not np.any((push_idx >= i - n_off) & (push_idx <= i + n_on))]
    return np.array(keep, dtype=int)


def style(ax):
    ax.grid(True, color="#e6e6e3", lw=0.6)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()
    out_dir = os.path.join(REPO_ROOT, "results",
                           datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_explore_contact")
    os.makedirs(out_dir)  # raises if it exists: never overwrite
    lines = []

    def report(text=""):
        print(text)
        lines.append(text)

    paths = sorted(glob.glob(os.path.join(args.folder, "run_*.npz")))
    report(f"folder: {args.folder} ({len(paths)} runs)")
    report(f"excluded: first {SKIP_START_S} s of each run, and anything containing a push")

    # Per-foot lists of (duration [s], peak force [N], neighbour info) collected over all runs.
    seg_on = {f: [] for f in FEET}
    seg_off = {f: [] for f in FEET}
    landings_per_run = {f: [] for f in FEET}
    gaps = {f: [] for f in FEET}
    per_run = []  # for part (c)

    for path in paths:
        d = np.load(path)
        t_fine, label, force, step = d["t_fine"], d["foot_contact"], d["foot_force"], d["step"]
        dt = t_fine[1] - t_fine[0]
        first = round(SKIP_START_S / dt)        # first included 500 Hz index
        push_idx = step[d["push_row"]]          # 500 Hz index of the state right after each push
        n = len(label)

        n_off, n_on = round(LANDING_OFF_S / dt), round(LANDING_ON_S / dt)
        all_landings = detect_landings(label, t_fine, LANDING_OFF_S, LANDING_ON_S)
        landings = []
        for f, foot in enumerate(FEET):
            # (a) segment durations
            start, end, value = segments(label[:, f])
            seg_peak = np.array([force[a:b, f].max() for a, b in zip(start, end)])
            for s, (a, b) in enumerate(zip(start, end)):
                if a < first or b >= n or np.any((push_idx >= a) & (push_idx <= b)):
                    continue
                dur = (b - a) * dt
                if value[s]:
                    # Is this on-segment next to a long on-segment, across a short off-gap?
                    after_long = (s >= 2 and (end[s - 2] - start[s - 2]) * dt >= SHORT_S
                                  and (end[s - 1] - start[s - 1]) * dt <= BOUNCE_GAP_S)
                    before_long = (s + 2 < len(start) and (end[s + 2] - start[s + 2]) * dt >= SHORT_S
                                   and (end[s + 1] - start[s + 1]) * dt <= BOUNCE_GAP_S)
                    seg_on[foot].append((dur, seg_peak[s], after_long, before_long))
                else:
                    # Peak force of the on-segments right before and after this off-segment.
                    seg_off[foot].append((dur, seg_peak[s], seg_peak[s - 1], seg_peak[s + 1],
                                          (end[s - 1] - start[s - 1]) * dt, (end[s + 1] - start[s + 1]) * dt))

            # (b) landings, with the same exclusions as everything else
            land = exclude_landings(all_landings[f], first, push_idx, n_off, n_on)
            landings_per_run[foot].append(len(land))
            gaps[foot].extend(np.diff(land) * dt)
            landings.append(land)

        # (c) state change per transition k -> k+1
        state, names = make_state(d["qpos"], d["qvel"])
        ds = state[1:] - state[:-1]
        k = np.arange(len(ds))
        included = (step[k] >= first) & ~np.isin(k + 1, d["push_row"])
        window = round(CONTACT_WINDOW_S / dt)
        contact = contact_steps(step, np.concatenate(landings), window)
        k0 = [transition_zero(step, land) for land in landings]
        per_run.append((ds, included, contact, k0))

    # ---------------- (a) ----------------
    report()
    report("(a) CONTACT SEGMENT DURATIONS (500 Hz label = total foot normal force > 1 N)")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    # Durations are multiples of 2 ms: below 19 ms the edges are at odd milliseconds (1, 3, ..., 19 ms),
    # so each short duration sits in the middle of its own bin. Above that, log-spaced edges.
    bins = np.concatenate([np.arange(1, 20, 2) / 1000, np.logspace(np.log10(0.019), np.log10(5.0), 40)[1:]])
    for f, foot in enumerate(FEET):
        on = np.array(seg_on[foot])
        off = np.array(seg_off[foot])
        for col, (arr, kind) in enumerate([(on, "on"), (off, "off")]):
            ax = axes[f, col]
            ax.hist(arr[:, 0], bins=bins, color=GROUP_COLORS[f], edgecolor="white", linewidth=0.5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.axvline(SHORT_S, color="#8a8a85", lw=1, ls=":", label="10 ms")
            ax.set_title(f"{foot} foot, contact-{kind} segments (n = {len(arr)})", fontsize=10)
            ax.set_ylabel("count (log)")
            ax.legend(fontsize=8, frameon=False)
            style(ax)
        short_on = on[on[:, 0] < SHORT_S]
        short_off = off[off[:, 0] < SHORT_S]
        report(f"  {foot}: {len(on)} on-segments, {len(off)} off-segments kept")
        report(f"    on-segments  < 10 ms: {len(short_on)} ({len(short_on) / len(on) * 100:.1f} %)")
        if len(short_on):
            p = short_on[:, 1]
            report(f"      their peak force [N]: median {np.median(p):.1f}, 90th pct {np.percentile(p, 90):.1f}, "
                   f"max {p.max():.1f}; durations [ms]: "
                   f"{ {int(v): int(c) for v, c in zip(*np.unique(np.round(short_on[:, 0] * 1000), return_counts=True))} }")
            liftoff = short_on[:, 2].astype(bool)
            touchdown = short_on[:, 3].astype(bool)
            report(f"      right after a long on-segment (lift-off bounce, off-gap <= 10 ms): {liftoff.sum()}; "
                   f"right before one (touchdown bounce): {touchdown.sum()}; "
                   f"both: {(liftoff & touchdown).sum()}; neither (isolated tap): {(~liftoff & ~touchdown).sum()}")
        report(f"    off-segments < 10 ms: {len(short_off)} ({len(short_off) / len(off) * 100:.1f} %)")
        if len(short_off):
            report(f"      force inside them is <= 1 N by definition (max {short_off[:, 1].max():.2f} N); "
                   f"peak of the on-segment before: median {np.median(short_off[:, 2]):.0f} N, "
                   f"after: median {np.median(short_off[:, 3]):.0f} N")
            both_long = (short_off[:, 4] >= SHORT_S) & (short_off[:, 5] >= SHORT_S)
            report(f"      between two on-segments >= 10 ms (dip inside a stance): {both_long.sum()}; "
                   f"next to a short on-segment (part of a bounce): {(~both_long).sum()}")
        long_on = on[on[:, 0] >= 0.1, 0]
        long_off = off[off[:, 0] >= 0.1, 0]
        report(f"    on-segments >= 100 ms: n = {len(long_on)}, median {np.median(long_on) * 1000:.0f} ms; "
               f"off-segments >= 100 ms: n = {len(long_off)}, median {np.median(long_off) * 1000:.0f} ms")
    for ax in axes[1]:
        ax.set_xlabel("segment duration [s] (log)")
    fig.suptitle("Contact segment durations (first 1 s and segments containing a push excluded)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "segment_durations.png"), dpi=130)
    plt.close(fig)

    # ---------------- (b) ----------------
    report()
    report("(b) LANDINGS (src/contact.py detect_landings):")
    report("    a False -> True change of the 500 Hz label with the foot off for >= 50 ms before")
    report("    and on for >= 50 ms after; landings whose 50 ms windows overlap a push or the first 1 s are dropped")
    fig, ax = plt.subplots(figsize=(10, 4))
    gap_bins = np.arange(0.0, 2.0, 0.02)
    for f, foot in enumerate(FEET):
        c = np.array(landings_per_run[foot])
        g = np.array(gaps[foot])
        report(f"  {foot}: landings per run min {c.min()}, median {np.median(c):.0f}, max {c.max()} "
               f"(total {c.sum()} over {len(c)} runs)")
        report(f"    gaps between consecutive landings [s]: min {g.min():.3f}, 5th {np.percentile(g, 5):.3f}, "
               f"25th {np.percentile(g, 25):.3f}, median {np.median(g):.3f}, 75th {np.percentile(g, 75):.3f}, "
               f"95th {np.percentile(g, 95):.3f}, max {g.max():.3f}")
        report(f"    gaps < 0.6 s: {(g < 0.6).sum()}, 0.6-1.0 s: {((g >= 0.6) & (g <= 1.0)).sum()}, "
               f"> 1.0 s: {(g > 1.0).sum()} (a gap > 1 s usually means a landing was dropped near a push)")
        ax.hist(g, bins=gap_bins, color=GROUP_COLORS[f], alpha=0.6, label=f"{foot} foot", edgecolor="white",
                linewidth=0.5)
    ax.axvline(0.8, color="#222222", lw=1, ls=":", label="gait period 0.8 s")
    ax.set_xlabel("time between consecutive landings of the same foot [s]")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Gaps between landings (rule: off >= 50 ms before, on >= 50 ms after)")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "landing_gaps.png"), dpi=130)
    plt.close(fig)

    # ---------------- (c) ----------------
    report()
    report("(c) STATE CHANGE PER 50 Hz TRANSITION AROUND LANDINGS")
    ds_inc = np.concatenate([ds[inc] for ds, inc, _, _ in per_run])
    contact_inc = np.concatenate([c[inc] for _, inc, c, _ in per_run])
    std = ds_inc.std(axis=0)
    nds = np.abs(ds_inc / std)
    report(f"  transitions kept: {len(ds_inc)}; contact steps: {contact_inc.sum()} "
           f"({contact_inc.mean() * 100:.1f} % of kept transitions)")
    report("  contact step rule: transition k -> k+1 spans the states at t_k and t_k+1 = t_k + 20 ms. It is a")
    report("  contact step if a landing (either foot; landing time = first 500 Hz state with contact)")
    report("  lies in [t_k - 40 ms, t_k+1 + 40 ms], ends included. That is transitions -2..+2 around")
    report("  transition 0 (5 transitions), or -2..+3 (6) when the landing state falls exactly on a policy step.")
    groups = list(STATE_GROUPS.items())
    on_mean = nds[contact_inc].mean(axis=0)
    off_mean = nds[~contact_inc].mean(axis=0)
    report("  mean |normalized ds| on contact steps / on other steps:")
    for name, sl in groups + [("overall", slice(0, 34))]:
        report(f"    {name:10s} {nds[contact_inc][:, sl].mean():.3f} / {nds[~contact_inc][:, sl].mean():.3f} = "
               f"{nds[contact_inc][:, sl].mean() / nds[~contact_inc][:, sl].mean():.2f}")
    ratio = on_mean / off_mean
    report("  10 state dimensions with the largest ratio:")
    for i in np.argsort(ratio)[::-1][:10]:
        report(f"    {names[i]:22s} ratio {ratio[i]:.2f}  (contact {on_mean[i]:.3f}, other {off_mean[i]:.3f})")
    report("  per-dimension std of ds used for normalization: "
           + ", ".join(f"{nm} {s:.3g}" for nm, s in zip(names, std)))

    # Landing-aligned mean |normalized ds| per group, one plot per foot.
    offsets = np.arange(-ALIGN, ALIGN + 1)
    for f, foot in enumerate(FEET):
        rows = []  # one (len(offsets), n_groups) array per landing, NaN where the transition is excluded
        for ds, inc, _, k0 in per_run:
            for k in k0[f]:
                r = np.full((len(offsets), len(groups)), np.nan)
                for j, o in enumerate(offsets):
                    if 0 <= k + o < len(ds) and inc[k + o]:
                        a = np.abs(ds[k + o] / std)
                        r[j] = [a[sl].mean() for _, sl in groups]
                rows.append(r)
        rows = np.array(rows)
        mean = np.nanmean(rows, axis=0)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axvspan(-2.5, 2.5, color="#e6e6e3", alpha=0.6, lw=0, label="contact steps (-2..+2)")
        for g, (name, _) in enumerate(groups):
            ax.plot(offsets, mean[:, g], color=GROUP_COLORS[g], lw=2, marker="o", ms=4, label=name)
        ax.axvline(0, color="#222222", lw=1, ls=":")
        ax.set_xticks(offsets)
        ax.set_xlabel("transition index relative to the landing (0 = transition containing the touchdown)")
        ax.set_ylabel("mean |ds| / std(ds), averaged over the group's dimensions")
        ax.set_title(f"{foot} foot landings (n = {len(rows)}): state change per 20 ms transition")
        ax.legend(fontsize=8, frameon=False, ncol=2)
        style(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"landing_aligned_{foot}.png"), dpi=130)
        plt.close(fig)
        report(f"  {foot} landings in the aligned plot: {len(rows)}; peak transition per group: "
               + ", ".join(f"{name} {offsets[np.nanargmax(mean[:, g])]:+d}" for g, (name, _) in enumerate(groups)))

    with open(os.path.join(out_dir, "report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nsaved plots and report.txt to {out_dir}")


if __name__ == "__main__":
    main()
