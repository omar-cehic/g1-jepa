"""Tests for src/eval/rollout.py.

Uses the first 3 runs of the tuning split of data/processed/2026-10-07_182522.
The windows are built here with plain loops (not with the code under test).

(a) perfect model: a step function that ignores its input and returns the true next state
    of each window gives error exactly 0 at every step (checks the time indexing of rollout).
    Its step function also checks it gets s_k first and action a_{k+j-1} at call j.
    The same through evaluate: a step function that looks up the true next state of its
    input gives RMS exactly 0 at every step (checks the indexing inside evaluate).
(b) stale model: returning the true current state s_{k+j-1} instead of s_{k+j} gives error
    exactly -(s_{k+j} - s_{k+j-1}) at every j, which is not 0 (so (a) is not an accident).
(c) constant velocity: the rollout error at j = 1 equals step(s_k) - s_{k+1} exactly, and
    evaluate's RMS at j = 1 (all / contact / rest) matches the direct RMS to 1e-12 relative
    (only the order of the sums differs).
(d) contact labels: in a run where only one window is usable, evaluate's contact count at
    step j is 1 exactly where contact_step[run, k+j-1] is True.
(e) usable_starts on hand built valid rows.

Nothing is written. Run: python tests/test_rollout.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from eval.rollout import usable_starts, rollout, evaluate, SUBSETS  # noqa: E402
from features import STATE_GROUPS  # noqa: E402
from models import constant_velocity  # noqa: E402

DATASET = os.path.join(REPO_ROOT, "data", "processed", "2026-10-07_182522", "dataset.npz")
H = 100
N_RUNS = 3


def subset(data, runs):
    """dataset dict with only the given run indices."""
    keys = ["states", "actions", "valid", "contact_step", "split"]
    return {key: data[key][runs] for key in keys}


def windows(data, run, starts):
    """s0 (W, 34), actions (W, H, 12), truth (W, H, 34) with plain loops."""
    states, actions = data["states"][run], data["actions"][run]
    s0 = np.array([states[k] for k in starts])
    a = np.array([[actions[k + j - 1] for j in range(1, H + 1)] for k in starts])
    truth = np.array([[states[k + j] for j in range(1, H + 1)] for k in starts])
    return s0, a, truth


def test_perfect_and_stale(data):
    for run in range(N_RUNS):
        starts = usable_starts(data["valid"][run], H)
        s0, a, truth = windows(data, run, starts)

        calls = [0]

        def perfect(s, actions):
            j = calls[0] + 1
            if j == 1:
                assert np.array_equal(s, s0)                  # the rollout starts from s_k
            assert np.array_equal(actions, a[:, j - 1])       # call j gets a_{k+j-1}
            calls[0] += 1
            return truth[:, j - 1]                             # s_{k+j}
        err = rollout(perfect, s0, a) - truth
        assert np.all(err == 0.0), np.abs(err).max()

        calls[0] = 0

        def stale(s, actions):
            j = calls[0] + 1
            calls[0] += 1
            return s0 if j == 1 else truth[:, j - 2]          # s_{k+j-1}
        err = rollout(stale, s0, a) - truth
        one_step_change = truth - np.concatenate([s0[:, None], truth[:, :-1]], axis=1)
        assert np.array_equal(err, -one_step_change)
        assert np.abs(err).max() > 0.1
    print(f"(a) perfect model through rollout: error exactly 0 on {N_RUNS} runs, "
          f"{len(starts)} windows in the last")
    print(f"(b) stale model: error = -(one step change) exactly, max |error| = {np.abs(err).max():.3f}")

    # Perfect model through evaluate: look up the true next state of whatever state it gets.
    next_row = {}
    for run in range(N_RUNS):
        for k in range(data["states"].shape[1] - 1):
            next_row[data["states"][run, k].tobytes()] = (run, k)

    def lookup(s, actions):
        out = np.empty_like(s)
        for i in range(len(s)):
            run, k = next_row[s[i].tobytes()]
            assert np.array_equal(actions[i], data["actions"][run, k])
            out[i] = data["states"][run, k + 1]
        return out
    m = evaluate(lookup, data, "tuning", H)
    for subset_name in SUBSETS:
        assert np.all(m["rms_" + subset_name] == 0.0)
        for group in STATE_GROUPS:
            assert np.all(m[f"rms_{subset_name}_{group}"] == 0.0)
    print(f"    perfect model through evaluate: RMS exactly 0 at every step, {m['n_windows']} windows")


def test_constant_velocity(data):
    step = constant_velocity.step
    err1, contact1 = [], []
    for run in range(N_RUNS):
        starts = usable_starts(data["valid"][run], H)
        s0, a, truth = windows(data, run, starts)
        rollout_err = rollout(step, s0, a[:, :1])[:, 0] - truth[:, 0]
        direct_err = step(data["states"][run, starts], data["actions"][run, starts]) - data["states"][run, starts + 1]
        assert np.array_equal(rollout_err, direct_err)
        err1.append(direct_err)
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
    print(f"(c) constant velocity j = 1: rollout error == step(s_k) - s_(k+1) exactly; "
          f"evaluate RMS vs direct: max relative diff {worst:.1e}")
    print(f"    j = 1 samples: all {m['count_all'][0]}, contact {m['count_contact'][0]}, "
          f"rest {m['count_rest'][0]}")


def test_contact_labels(data):
    run, k = 0, usable_starts(data["valid"][0], H)[0]
    one = subset(data, [run])
    one["valid"] = np.zeros_like(one["valid"])
    one["valid"][0, k:k + H] = True                        # only the window starting at k is usable
    m = evaluate(constant_velocity.step, one, "tuning", H)
    assert m["n_windows"] == 1
    expected = np.array([data["contact_step"][run, k + j - 1] for j in range(1, H + 1)])
    assert np.array_equal(m["count_contact"], expected.astype(int))
    assert np.array_equal(m["count_rest"], (~expected).astype(int))
    assert 0 < expected.sum() < H                          # the window has both kinds of steps
    contact_j = [j for j in range(1, H + 1) if expected[j - 1]]
    print(f"(d) one window at k = {k}: contact predictions j = {contact_j}")


def test_usable_starts():
    T, F = True, False
    valid = np.array([T, T, T, T, F, T, T, T, F, F, T, T])
    assert np.array_equal(usable_starts(valid, 3), [0, 1, 5])
    assert np.array_equal(usable_starts(valid, 1), [0, 1, 2, 3, 5, 6, 7, 10, 11])
    assert np.array_equal(usable_starts(valid, 4), [0])
    assert len(usable_starts(valid, 5)) == 0
    assert np.array_equal(usable_starts(np.ones(5, dtype=bool), 5), [0])
    assert len(usable_starts(np.ones(5, dtype=bool), 6)) == 0
    assert len(usable_starts(np.zeros(5, dtype=bool), 1)) == 0
    print("(e) usable_starts on hand built rows: as expected")


def main():
    full = dict(np.load(DATASET))
    tuning_runs = np.flatnonzero(full["split"] == "tuning")[:N_RUNS]
    data = subset(full, tuning_runs)
    print(f"first {N_RUNS} tuning runs: run ids {full['run_ids'][tuning_runs].tolist()}")
    del full
    test_perfect_and_stale(data)
    test_constant_velocity(data)
    test_contact_labels(data)
    test_usable_starts()
    print("PASS test_rollout")


if __name__ == "__main__":
    main()
