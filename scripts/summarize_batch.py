"""Print a summary of a folder written by collect_data.py.

One row per run:
  - fell or not (falls are listed in detail at the end: time, which geom/body
    touched the floor, and the command active at the fall)
  - landings per foot, as defined in src/contact.py (off >= 50 ms before,
    on >= 50 ms after), and separately the raw number of False -> True
    changes of the 500 Hz contact label (includes bounces and taps)
  - median and mean time between landings, per foot
  - fraction of time with both / neither foot in contact
  - peak foot normal force at landings divided by body weight (mean over the
    run's landings); the peak is the maximum force on that foot in the first
    40 ms after the landing
Then totals over the folder. Nothing is excluded here (no first-second or push cut).

Usage: python scripts/summarize_batch.py data/raw/<folder>
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from contact import detect_landings  # noqa: E402

PEAK_WINDOW_S = 0.04


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()
    with open(os.path.join(args.folder, "meta.json")) as f:
        meta = json.load(f)
    weight = meta["robot_mass_kg"] * np.linalg.norm(meta["gravity"])

    print("                     |  landings  | raw F->T  | gap median [s] | gap mean [s] |  contact time  | peak/weight")
    print(" run  seed  fell     |   L    R   |  L    R   |    L      R    |   L     R    |  both  neither |   L     R")
    falls = []
    n_landings, n_raw, n_transitions, n_push_transitions, durations = [], [], 0, 0, []
    for path in sorted(glob.glob(os.path.join(args.folder, "run_*.npz"))):
        d = np.load(path)
        contact, force, t_fine = d["foot_contact"], d["foot_force"], d["t_fine"]
        dt = t_fine[1] - t_fine[0]
        window = round(PEAK_WINDOW_S / dt)

        landings = detect_landings(contact, t_fine)
        raw = [np.sum(~contact[:-1, f] & contact[1:, f]) for f in range(2)]
        gap_median = [np.median(np.diff(land)) * dt if len(land) > 1 else np.nan for land in landings]
        gap_mean = [np.mean(np.diff(land)) * dt if len(land) > 1 else np.nan for land in landings]
        peak = [np.mean([force[i : i + window, f].max() for i in landings[f]]) / weight
                if len(landings[f]) else np.nan for f in range(2)]
        both = np.mean(contact[:, 0] & contact[:, 1])
        neither = np.mean(~contact[:, 0] & ~contact[:, 1])

        fell = bool(d["fell"])
        if fell:
            falls.append((int(d["run_index"]), int(d["seed"]), float(d["fall_time"]), int(d["fall_geom"]),
                          str(d["fall_body"]), d["cmd"][-1].astype(float).round(3).tolist()))
        n_landings.append([len(land) for land in landings])
        n_raw.append(raw)
        n_transitions += len(d["t"]) - 1
        n_push_transitions += len(d["push_row"])
        durations.append(len(t_fine) * dt)
        print(f" {int(d['run_index']):3d}  {int(d['seed']):4d}  "
              f"{'FELL' if fell else 'no  '}     | {len(landings[0]):3d}  {len(landings[1]):3d}   | "
              f"{raw[0]:3d}  {raw[1]:3d}  |  {gap_median[0]:.3f}  {gap_median[1]:.3f}  | "
              f"{gap_mean[0]:.3f} {gap_mean[1]:.3f}  | {both * 100:5.1f}  {neither * 100:5.1f}   | "
              f"{peak[0]:.2f}  {peak[1]:.2f}")

    n_landings, n_raw = np.array(n_landings), np.array(n_raw)
    print()
    print(f"runs: {len(n_landings)}, falls: {len(falls)} ({len(falls) / len(n_landings) * 100:.1f} %)")
    for run, seed, t, geom, body, cmd in falls:
        print(f"  run {run} (seed {seed}) fell at t = {t:.3f} s, geom {geom} on body {body}, "
              f"command at fall [vx, vy, yaw rate] = {cmd}")
    for f, foot in enumerate(["left", "right"]):
        print(f"{foot:5s} landings per run: min {n_landings[:, f].min()}, median {np.median(n_landings[:, f]):.0f}, "
              f"max {n_landings[:, f].max()}, total {n_landings[:, f].sum()}; raw False -> True changes per run: "
              f"min {n_raw[:, f].min()}, median {np.median(n_raw[:, f]):.0f}, max {n_raw[:, f].max()}")
    print(f"simulated time: {sum(durations):.1f} s; 50 Hz transitions (k -> k+1): {n_transitions}, "
          f"of which {n_push_transitions} contain a push")


if __name__ == "__main__":
    main()
