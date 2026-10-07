"""Check that two folders from collect_data.py hold identical data.

Every run file must exist in both folders, with the same array names, and
every array must have the same dtype and be np.array_equal (NaN == NaN for
float arrays, e.g. fall_time of a run that did not fall).

Usage: python scripts/compare_batches.py data/raw/<folder_a> data/raw/<folder_b>
"""

import argparse
import glob
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_a")
    parser.add_argument("folder_b")
    args = parser.parse_args()

    names_a = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.folder_a, "run_*.npz")))
    names_b = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.folder_b, "run_*.npz")))
    assert names_a == names_b, f"different run files: {names_a} vs {names_b}"

    all_same = True
    for name in names_a:
        a = np.load(os.path.join(args.folder_a, name))
        b = np.load(os.path.join(args.folder_b, name))
        assert sorted(a.files) == sorted(b.files), f"{name}: different arrays"
        different = []
        for key in a.files:
            equal_nan = np.issubdtype(a[key].dtype, np.floating)
            if a[key].dtype != b[key].dtype or not np.array_equal(a[key], b[key], equal_nan=equal_nan):
                different.append(key)
        print(f"{name}: {len(a.files)} arrays, " + ("all identical" if not different else f"DIFFERENT: {different}"))
        all_same = all_same and not different
    print("IDENTICAL" if all_same else "NOT IDENTICAL")


if __name__ == "__main__":
    main()
