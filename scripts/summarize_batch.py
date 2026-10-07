"""Print a per-run summary of a folder written by collect_data.py.

For each run: fell or not (and if so, when, which geom/body touched the floor,
and the command active at the fall), foot landings per foot (a landing is a
False -> True change of the 500 Hz contact label), the mean time between
landings, the fraction of time with both / neither foot in contact, and the
peak foot normal force at landings divided by body weight. The peak at a
landing is the maximum force on that foot in the first 40 ms after touchdown.

Usage: python scripts/summarize_batch.py data/raw/<folder>
"""

import argparse
import glob
import json
import os

import numpy as np

PEAK_WINDOW_S = 0.04


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()
    with open(os.path.join(args.folder, "meta.json")) as f:
        meta = json.load(f)
    weight = meta["robot_mass_kg"] * np.linalg.norm(meta["gravity"])

    for path in sorted(glob.glob(os.path.join(args.folder, "run_*.npz"))):
        d = np.load(path)
        contact, force, t_fine = d["foot_contact"], d["foot_force"], d["t_fine"]
        dt = t_fine[1] - t_fine[0]
        window = round(PEAK_WINDOW_S / dt)

        print(f"{os.path.basename(path)} (seed {int(d['seed'])}): ", end="")
        if d["fell"]:
            print(f"FELL at t = {float(d['fall_time']):.3f} s, geom {int(d['fall_geom'])} "
                  f"on body {d['fall_body']}, command at fall {d['cmd'][-1].astype(float).round(3).tolist()}")
        else:
            print(f"no fall, {t_fine[-1] + dt:.2f} s")

        for foot, name in enumerate(["left", "right"]):
            landings = np.flatnonzero(~contact[:-1, foot] & contact[1:, foot]) + 1
            gaps = np.diff(t_fine[landings])
            peaks = np.array([force[i : i + window, foot].max() for i in landings]) / weight
            print(f"  {name:5s}: {len(landings)} landings, mean time between landings "
                  f"{gaps.mean():.3f} s (median {np.median(gaps):.3f}, min {gaps.min():.3f}, max {gaps.max():.3f}), "
                  f"peak force at landing / weight: mean {peaks.mean():.2f}, max {peaks.max():.2f}")
        both = np.mean(contact[:, 0] & contact[:, 1])
        neither = np.mean(~contact[:, 0] & ~contact[:, 1])
        print(f"  time with both feet in contact {both * 100:.1f} %, neither foot {neither * 100:.1f} %")


if __name__ == "__main__":
    main()
