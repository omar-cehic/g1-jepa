"""Tests for the helper functions of scripts/analyze_divergence.py.

(a) first_exit on hand built rows: first True at the right 1-based step, H + 1 if never.
(b) top_share on hand built values, including the rounding up of the number of windows.
(c) train_range on a tiny hand built dataset: invalid transitions and non-train runs are ignored,
    and s_{k+1} of the last valid transition counts.

Nothing is written. Run: python tests/test_analyze_divergence.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from analyze_divergence import first_exit, top_share, train_range  # noqa: E402


def test_first_exit():
    outside = np.array([
        [False, False, False, False],   # never   -> 5
        [True, False, False, False],    # j = 1
        [False, False, True, True],     # j = 3
        [False, False, False, True],    # j = 4
    ])
    assert first_exit(outside).tolist() == [5, 1, 3, 4]


def test_top_share():
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 91.0])  # sum 100
    assert top_share(x, 0.10) == 0.91          # 1 window
    assert top_share(x, 0.01) == 0.91          # ceil(0.1) = 1 window
    assert np.isclose(top_share(x, 0.20), 0.92)  # 2 windows
    assert top_share(x, 1.0) == 1.0


def test_train_range():
    # 2 runs, 4 rows (3 transitions), 1 feature.
    states = np.array([[[0.0], [1.0], [2.0], [50.0]],     # train: transition 2 (2 -> 50) is invalid
                       [[-9.0], [-9.0], [99.0], [99.0]]])  # tuning: ignored
    valid = np.array([[True, True, False], [True, True, True]])
    data = {"states": states, "valid": valid, "split": np.array(["train", "tuning"])}
    lo, hi = train_range(data)
    assert lo.tolist() == [0.0] and hi.tolist() == [2.0], (lo, hi)  # 2.0 is s_{k+1} of transition 1


if __name__ == "__main__":
    test_first_exit()
    test_top_share()
    test_train_range()
    print("all analyze_divergence tests passed")
