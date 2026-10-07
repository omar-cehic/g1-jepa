"""Sanity tests for the helpers left in scripts/explore_contact.py (the landing and
contact-step rules themselves are tested in tests/test_contact.py).

Run: python tests/test_explore_contact.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from explore_contact import segments, exclude_landings  # noqa: E402


def main():
    # segments: runs of equal values
    start, end, value = segments(np.array([0, 0, 1, 1, 1, 0, 1], dtype=bool))
    assert start.tolist() == [0, 2, 5, 6] and end.tolist() == [2, 5, 6, 7] and value.tolist() == [0, 1, 0, 1]

    # exclude_landings: window [i - 25, i + 25] must start at or after first and contain no push
    land = np.array([100, 524, 525, 700])
    assert exclude_landings(land, 500, np.array([], dtype=int), 25, 25).tolist() == [525, 700]
    assert exclude_landings(land, 0, np.array([725]), 25, 25).tolist() == [100, 524, 525]  # push at the window end
    assert exclude_landings(land, 0, np.array([726]), 25, 25).tolist() == [100, 524, 525, 700]
    print("segments, exclude_landings: all checks pass")
    print("PASS test_explore_contact")


if __name__ == "__main__":
    main()
