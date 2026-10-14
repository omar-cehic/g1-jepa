"""Open loop rollouts of a one step model, and their errors against the true states.

A window starts at row k of a run. The model gets the true state s_k and the recorded
actions a_k ... a_{k+H-1}, and feeds its own predictions back in:
  s_hat_{k+1} = step(s_k, a_k),   s_hat_{k+j+1} = step(s_hat_{k+j}, a_{k+j})
Prediction j (j = 1 .. H) is compared with the true state s_{k+j}; error = s_hat - s in raw
units. Prediction j is a contact step if transition k+j-1 (the one that produced it) has
contact_step True.

In arrays, prediction j is at index j - 1 along the step axis.

Per window group error at step j: the RMS over the group's features of the error in that window
at step j. Divergence: a window has diverged by step j if at some step <= j any predicted feature
except gravity_z is outside the train range (or is not finite).
"""

import numpy as np

from features import STATE_GROUPS, STATE_NAMES

SUBSETS = ["all", "contact", "rest"]
# Left out of the divergence check: the true gravity_z is -1 when the pelvis is upright, the train
# minimum is -0.99999998, and a good prediction can land a hair below it.
GRAVITY_Z = STATE_NAMES.index("gravity_z")


def train_range(data):
    """Min and max (34,) of each feature over s_k and s_{k+1} of the valid train transitions."""
    train = data["split"] == "train"
    states, valid = data["states"][train], data["valid"][train]
    s_k, s_next = states[:, :-1][valid], states[:, 1:][valid]  # (n_valid, 34) each
    lo = np.minimum(s_k.min(axis=0), s_next.min(axis=0))
    hi = np.maximum(s_k.max(axis=0), s_next.max(axis=0))
    return lo, hi


def first_exit(outside):
    """First step j (1 .. H) where outside (W, H) is True, per window; H + 1 if never."""
    W, n_steps = outside.shape
    first = np.argmax(outside, axis=1) + 1  # argmax gives the first True (or 0 if none)
    first[~outside.any(axis=1)] = n_steps + 1
    return first


def outside_range(preds, lo, hi):
    """(W, H) bool: is any predicted feature except gravity_z outside [lo, hi] or not finite?"""
    out = ~np.isfinite(preds) | (preds < lo) | (preds > hi)  # NaN compares False, so check it
    out[:, :, GRAVITY_Z] = False
    return out.any(axis=2)


def usable_starts(valid_row, H):
    """Start rows k (sorted) for which transitions k ... k+H-1 are all valid."""
    n_transitions = len(valid_row)
    return np.array([k for k in range(n_transitions - H + 1) if valid_row[k:k + H].all()], dtype=int)


def rollout(step, s0, actions):
    """Predictions (N, H, 34) from start states s0 (N, 34) and actions (N, H, 12)."""
    N, H, _ = actions.shape
    preds = np.empty((N, H, s0.shape[1]))
    s = s0
    for j in range(H):
        s = step(s, actions[:, j])
        preds[:, j] = s
    return preds


def evaluate(step, data, split, H=100, bounds=None):
    """Roll out every usable window of every run in one split and return the RMS errors.

    data is the dict from dataset.npz (needs states, actions, valid, contact_step, split).
    bounds: (lo, hi) from train_range, or None to skip the divergence fraction.
    Returns a dict:
      n_windows                     number of windows
      count_<subset>  (H,)          samples (windows) at each step j
      rms_<subset>    (H, 34)       RMS error per step and feature
      rms_<subset>_<group> (H,)     RMS error per step, pooled over the group's features
      median_<subset>_<group> (H,)  median over windows of the per window group error at step j
      p90_<subset>_<group> (H,)     90th percentile of the same
      diverged_frac (H,)            fraction of all windows that have diverged by step j (only with bounds)
    for subset in all, contact, rest. A step with no samples gets NaN.
    """
    n_features = data["states"].shape[2]
    sse = {"contact": np.zeros((H, n_features)), "rest": np.zeros((H, n_features))}
    count = {"contact": np.zeros(H, dtype=int), "rest": np.zeros(H, dtype=int)}
    n_windows = 0
    # Kept for every window (one list entry per run): what the medians and the divergence need.
    group_err, contact_all, first = [], [], []

    for run in np.flatnonzero(data["split"] == split):
        starts = usable_starts(data["valid"][run], H)
        if len(starts) == 0:
            continue
        rows = starts[:, None] + np.arange(H)               # (W, H): row k+j-1 for prediction j
        s0 = data["states"][run, starts]                    # (W, 34)   s_k
        actions = data["actions"][run, rows]                # (W, H, 12) a_{k+j-1}
        truth = data["states"][run, rows + 1]               # (W, H, 34) s_{k+j}
        contact = data["contact_step"][run, rows]           # (W, H)     transition k+j-1

        preds = rollout(step, s0, actions)                  # (W, H, 34)
        sq_err = (preds - truth) ** 2                       # (W, H, 34)
        # np.where, not multiplying by the mask: a diverged rollout (inf) times 0 would give NaN.
        sse["contact"] += np.sum(np.where(contact[:, :, None], sq_err, 0.0), axis=0)
        sse["rest"] += np.sum(np.where(contact[:, :, None], 0.0, sq_err), axis=0)
        count["contact"] += np.sum(contact, axis=0)
        count["rest"] += np.sum(~contact, axis=0)
        n_windows += len(starts)
        group_err.append(np.stack([np.sqrt(sq_err[:, :, cols].mean(axis=2)) for cols in STATE_GROUPS.values()],
                                  axis=2))          # (W, H, n_groups)
        contact_all.append(contact)
        if bounds is not None:
            first.append(first_exit(outside_range(preds, *bounds)))

    sse["all"] = sse["contact"] + sse["rest"]
    count["all"] = count["contact"] + count["rest"]

    out = {"n_windows": n_windows}
    with np.errstate(invalid="ignore", divide="ignore"):  # 0 / 0 -> NaN when a step has no samples
        for subset in SUBSETS:
            n = count[subset][:, None]
            out["count_" + subset] = count[subset]
            out["rms_" + subset] = np.sqrt(sse[subset] / n)
            for group, cols in STATE_GROUPS.items():
                n_cols = cols.stop - cols.start
                group_sse = sse[subset][:, cols].sum(axis=1)
                out[f"rms_{subset}_{group}"] = np.sqrt(group_sse / (count[subset] * n_cols))

    group_err, contact_all = np.concatenate(group_err), np.concatenate(contact_all)
    in_subset = {"all": np.ones_like(contact_all), "contact": contact_all, "rest": ~contact_all}
    for subset in SUBSETS:
        median = np.full((H, len(STATE_GROUPS)), np.nan)
        p90 = np.full((H, len(STATE_GROUPS)), np.nan)
        for j in range(H):
            e = group_err[in_subset[subset][:, j], j]  # (windows in the subset at step j, n_groups)
            if len(e) > 0:
                median[j] = np.median(e, axis=0)
                p90[j] = np.percentile(e, 90, axis=0)
        for g, group in enumerate(STATE_GROUPS):
            out[f"median_{subset}_{group}"] = median[:, g]
            out[f"p90_{subset}_{group}"] = p90[:, g]
    if bounds is not None:
        first = np.concatenate(first)
        out["diverged_frac"] = np.array([np.mean(first <= j) for j in range(1, H + 1)])
    return out
