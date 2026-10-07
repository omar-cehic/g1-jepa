"""Tests for src/contact.py.

1. detect_landings on synthetic label sequences (dt = 2 ms, so 50 ms = 25 samples):
   clean landing, lift-off bounce, touchdown bounce, isolated tap, dip inside a
   stance, run that ends mid-stance, change too close to the start of the run.
2. transition_zero and contact_steps on a synthetic step array.
3. With the same exclusions as scripts/explore_contact.py, detect_landings
   reproduces the pilot counts (685 left, 709 right). Skipped if the pilot
   folder is not on this machine.

Run: python tests/test_contact.py [pilot_folder]
"""

import glob
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from contact import detect_landings, contact_steps, transition_zero  # noqa: E402
from explore_contact import exclude_landings, SKIP_START_S  # noqa: E402

DT = 0.002
PILOT = os.path.join(REPO_ROOT, "data", "raw", "2026-10-07_144241")


def label_from(pieces):
    """Build a one-foot label (N x 1) from (value, n_samples) pieces."""
    label = np.concatenate([np.full(n, bool(v)) for v, n in pieces])
    return label[:, None]


def landings_of(pieces):
    label = label_from(pieces)
    t_fine = np.arange(len(label)) * DT
    return detect_landings(label, t_fine)[0].tolist()


def main():
    # --- 1. synthetic sequences ---
    cases = [
        ("clean landing", [(0, 100), (1, 100)], [100]),
        # stance from 50 ends at 150, off 1 sample, back on 2 samples, then a long swing
        ("lift-off bounce", [(0, 50), (1, 100), (0, 1), (1, 2), (0, 100)], [50]),
        # touches for 2 samples, off 1, then the real stance: the stance start has only 1 off sample
        # before it, so the rule does NOT count this landing (known limitation, see report)
        ("touchdown bounce", [(0, 100), (1, 2), (0, 1), (1, 100)], []),
        ("isolated tap", [(0, 100), (1, 3), (0, 100)], []),
        # stance with a 1-sample dip in the middle: one landing, the re-contact after the dip is not one
        ("dip inside a stance", [(0, 50), (1, 60), (0, 1), (1, 60), (0, 50)], [50]),
        ("run ends 40 ms into a stance", [(0, 50), (1, 20)], []),
        ("run ends exactly 50 ms into a stance", [(0, 50), (1, 25)], [50]),
        ("change only 20 ms after the run starts", [(0, 10), (1, 100)], []),
        ("change exactly 50 ms after the run starts", [(0, 25), (1, 100)], [25]),
        ("off 48 ms before (one sample short)", [(1, 50), (0, 24), (1, 100)], []),
    ]
    for name, pieces, expected in cases:
        got = landings_of(pieces)
        assert got == expected, f"{name}: expected {expected}, got {got}"
        print(f"  {name:42s} landings {got}")

    # Two feet are handled independently.
    two = np.concatenate([label_from([(0, 100), (1, 100)]), label_from([(0, 150), (1, 50)])], axis=1)
    got = detect_landings(two, np.arange(200) * DT)
    assert got[0].tolist() == [100] and got[1].tolist() == [150]
    print("1. detect_landings: all synthetic cases pass")

    # --- 2. transition_zero and contact_steps, step[k] = 10 (k + 1) ---
    step = 10 * (np.arange(20) + 1)
    # landing state i = 55: touchdown in physics step 54 -> 55, inside transition 50 -> 60, i.e. k = 4
    assert transition_zero(step, np.array([55])).tolist() == [4]
    # landing exactly on a policy step, i = 60: touchdown in physics step 59 -> 60, still k = 4
    assert transition_zero(step, np.array([60])).tolist() == [4]
    for i, expected in [(55, [2, 3, 4, 5, 6]), (60, [2, 3, 4, 5, 6, 7])]:
        c = contact_steps(step, np.array([i]), window=20)
        assert np.flatnonzero(c).tolist() == expected, (i, np.flatnonzero(c))
    print("2. transition_zero, contact_steps: pass (contact steps -2..+2 around transition 0, "
          "or -2..+3 when the landing is on a policy step)")

    # --- 3. pilot counts ---
    folder = sys.argv[1] if len(sys.argv) > 1 else PILOT
    paths = sorted(glob.glob(os.path.join(folder, "run_*.npz")))
    if not paths:
        print(f"3. SKIPPED: pilot folder {folder} not found")
    else:
        totals = np.zeros(2, dtype=int)
        for path in paths:
            d = np.load(path)
            first = round(SKIP_START_S / DT)
            push_idx = d["step"][d["push_row"]]
            for f, land in enumerate(detect_landings(d["foot_contact"], d["t_fine"])):
                totals[f] += len(exclude_landings(land, first, push_idx, 25, 25))
        print(f"3. pilot ({len(paths)} runs) with explore_contact.py exclusions: "
              f"{totals[0]} left, {totals[1]} right landings")
        assert totals.tolist() == [685, 709]
    print("PASS test_contact")


if __name__ == "__main__":
    main()
