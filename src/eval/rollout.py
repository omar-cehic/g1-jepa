"""Open loop rollouts of a one step model, and their errors against the true states.

A window starts at row k of a run. The model gets the true state s_k and the recorded
actions a_k ... a_{k+H-1}, and feeds its own predictions back in:
  s_hat_{k+1} = step(s_k, a_k),   s_hat_{k+j+1} = step(s_hat_{k+j}, a_{k+j})
Prediction j (j = 1 .. H) is compared with the true state s_{k+j}; error = s_hat - s in raw
units. Prediction j is a contact step if transition k+j-1 (the one that produced it) has
contact_step True.

In arrays, prediction j is at index j - 1 along the step axis.
"""

import numpy as np

from features import STATE_GROUPS

SUBSETS = ["all", "contact", "rest"]


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


def evaluate(step, data, split, H=100):
    """Roll out every usable window of every run in one split and return the RMS errors.

    data is the dict from dataset.npz (needs states, actions, valid, contact_step, split).
    Returns a dict:
      n_windows                  number of windows
      count_<subset>  (H,)       samples (windows) at each step j
      rms_<subset>    (H, 34)    RMS error per step and feature
      rms_<subset>_<group> (H,)  RMS error per step, pooled over the group's features
    for subset in all, contact, rest. A step with no samples gets NaN.
    """
    n_features = data["states"].shape[2]
    sse = {"contact": np.zeros((H, n_features)), "rest": np.zeros((H, n_features))}
    count = {"contact": np.zeros(H, dtype=int), "rest": np.zeros(H, dtype=int)}
    n_windows = 0

    for run in np.flatnonzero(data["split"] == split):
        starts = usable_starts(data["valid"][run], H)
        if len(starts) == 0:
            continue
        rows = starts[:, None] + np.arange(H)               # (W, H): row k+j-1 for prediction j
        s0 = data["states"][run, starts]                    # (W, 34)   s_k
        actions = data["actions"][run, rows]                # (W, H, 12) a_{k+j-1}
        truth = data["states"][run, rows + 1]               # (W, H, 34) s_{k+j}
        contact = data["contact_step"][run, rows]           # (W, H)     transition k+j-1

        sq_err = (rollout(step, s0, actions) - truth) ** 2  # (W, H, 34)
        # np.where, not multiplying by the mask: a diverged rollout (inf) times 0 would give NaN.
        sse["contact"] += np.sum(np.where(contact[:, :, None], sq_err, 0.0), axis=0)
        sse["rest"] += np.sum(np.where(contact[:, :, None], 0.0, sq_err), axis=0)
        count["contact"] += np.sum(contact, axis=0)
        count["rest"] += np.sum(~contact, axis=0)
        n_windows += len(starts)

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
    return out
