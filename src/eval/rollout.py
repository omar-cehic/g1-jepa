"""Open loop rollouts of a one step model, and their errors against the true states.

A window starts at row k of a run. The model gets the true state s_k and the recorded
actions a_k ... a_{k+H-1}, and feeds its own predictions back in:
  s_hat_{k+1} = step(s_k, a_k),   s_hat_{k+j+1} = step(s_hat_{k+j}, a_{k+j})
Prediction j (j = 1 .. H) is compared with the true state s_{k+j}; error = s_hat - s in raw
units. Prediction j is a contact step if transition k+j-1 (the one that produced it) has
contact_step True.

In arrays, prediction j is at index j - 1 along the step axis.

Models with a window of history n (the transformer world model, the JEPAs) use rollout_window():
the window holds the true x_{k-n+1}, ..., x_k at the start (x = states or embeddings), and each
prediction is appended to it while the oldest entry drops out. Prediction j gets the recorded actions
a_{k+j-n}, ..., a_{k+j-1}. Such models give evaluate() a predict(run, starts) function instead of step.

Per window group error at step j: the RMS over the group's features of the error in that window
at step j.

Foot contact force (optional): prediction j also gives the mean force per foot over transition k+j-1
(the same transition whose contact_step label decides whether prediction j is a contact step),
compared with the recorded force of that transition. It is one more group, "force" (2 features,
left and right foot, in N). Divergence: a window has diverged by step j if at some step <= j any predicted feature
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


def rollout_window(step, x, actions, starts, n, H):
    """Predictions (W, H, d) of a model that sees a window of its last n inputs, for the windows starting at starts.

    x (steps, d): one row per step of one run (raw states, or embeddings); actions (steps - 1, 12): the
    run's recorded actions. starts (W,): the start rows k. step(x_window, action_window) gets
    x_window (W, n, d) and action_window (W, n, 12), oldest first, and returns the next x (W, d).
    The window starts as the true x_{k-n+1}, ..., x_k. Prediction j (j = 1 .. H) is
      step([x_{k+j-n}, ..., x_{k+j-1}], [a_{k+j-n}, ..., a_{k+j-1}])
    where an x with index above k is an earlier prediction (fed back). With n = 1 this is rollout().
    """
    assert starts.min() >= n - 1, f"a window of {n} would start before step 0 (first start {starts.min()})"
    W = len(starts)
    buf = np.empty((W, n + H, x.shape[1]), dtype=x.dtype)   # slot i holds x_{k-n+1+i}
    buf[:, :n] = x[starts[:, None] + np.arange(1 - n, 1)]     # true x_{k-n+1}, ..., x_k
    action_rows = starts[:, None] + np.arange(1 - n, H)       # (W, n-1+H): a_{k-n+1}, ..., a_{k+H-1}
    for j in range(1, H + 1):
        # slots j-1 .. j+n-2 hold x_{k+j-n} .. x_{k+j-1}; columns j-1 .. j+n-2 of action_rows the same steps
        buf[:, n - 1 + j] = step(buf[:, j - 1:j - 1 + n], actions[action_rows[:, j - 1:j - 1 + n]])
    return buf[:, n:]


def force_stats(pred, truth, contact):
    """Force group metrics from predicted and true forces (N, H, 2) in N and contact labels (N, H), N windows.

    Per window error at step j: sqrt of the mean over the two feet of the squared error (N per foot).
    Returns rms_<subset>_force (H,) (sqrt of the mean over windows in the subset and both feet of the squared
    error), median_<subset>_force and p90_<subset>_force (H,) (over the windows in the subset of the per window
    error), for subset in all, contact, rest. A step with no samples gets NaN.
    """
    sq_err = (pred - truth) ** 2                     # (N, H, 2)
    window_err = np.sqrt(sq_err.mean(axis=2))        # (N, H)
    in_subset = {"all": np.ones_like(contact), "contact": contact, "rest": ~contact}
    out = {}
    for subset in SUBSETS:
        mask = in_subset[subset]
        n = mask.sum(axis=0)                          # windows in the subset at each j
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"rms_{subset}_force"] = np.sqrt(np.where(mask[:, :, None], sq_err, 0.0).sum(axis=(0, 2)) / (2 * n))
        median, p90 = np.full(len(n), np.nan), np.full(len(n), np.nan)
        for j in range(len(n)):
            e = window_err[mask[:, j], j]
            if len(e) > 0:
                median[j], p90[j] = np.median(e), np.percentile(e, 90)
        out[f"median_{subset}_force"], out[f"p90_{subset}_force"] = median, p90
    return out


def evaluate_force(force_predict, data, force, split, H=100):
    """Force group metrics (force_stats) of force_predict(run, starts) -> (W, H, 2) in N, on the same windows
    as evaluate(). Prediction j is compared with force[run, k+j-1], the force over transition k+j-1, and is a
    contact step if that transition is. Used for the readout floor."""
    preds, truths, contacts = [], [], []
    for run in np.flatnonzero(data["split"] == split):
        starts = usable_starts(data["valid"][run], H)
        if len(starts) == 0:
            continue
        rows = starts[:, None] + np.arange(H)               # (W, H): row k+j-1 for prediction j
        preds.append(force_predict(run, starts))
        truths.append(force[run, rows])
        contacts.append(data["contact_step"][run, rows])
    return force_stats(np.concatenate(preds), np.concatenate(truths), np.concatenate(contacts))


def evaluate(step, data, split, H=100, bounds=None, predict=None, force=None):
    """Roll out every usable window of every run in one split and return the RMS errors.

    data is the dict from dataset.npz (needs states, actions, valid, contact_step, split).
    bounds: (lo, hi) from train_range, or None to skip the divergence fraction.
    predict: None to roll out step with rollout(). Otherwise a function predict(run, starts) that returns
    the predictions (W, H, 34) in raw units for the windows of run starting at starts (step is then unused).
    force: None, or the true mean force per transition (runs, transitions, 2) in N. Then predict must return
    (states (W, H, 34), forces (W, H, 2)), where forces[:, j - 1] predicts force[run, k+j-1], and the force
    group metrics of force_stats() are added.
    Returns a dict:
      n_windows                     number of windows
      count_<subset>  (H,)          samples (windows) at each step j
      rms_<subset>    (H, 34)       RMS error per step and feature
      rms_<subset>_<group> (H,)     RMS error per step, pooled over the group's features
      median_<subset>_<group> (H,)  median over windows of the per window group error at step j
      p90_<subset>_<group> (H,)     90th percentile of the same
      diverged_frac (H,)            fraction of all windows that have diverged by step j (only with bounds)
      rms_/median_/p90_<subset>_force (H,)  force group, N per foot (only with force; see force_stats)
    for subset in all, contact, rest. A step with no samples gets NaN.
    """
    n_features = data["states"].shape[2]
    sse = {"contact": np.zeros((H, n_features)), "rest": np.zeros((H, n_features))}
    count = {"contact": np.zeros(H, dtype=int), "rest": np.zeros(H, dtype=int)}
    n_windows = 0
    # Kept for every window (one list entry per run): what the medians and the divergence need.
    group_err, contact_all, first = [], [], []
    force_pred, force_true = [], []   # only with force

    for run in np.flatnonzero(data["split"] == split):
        starts = usable_starts(data["valid"][run], H)
        if len(starts) == 0:
            continue
        rows = starts[:, None] + np.arange(H)               # (W, H): row k+j-1 for prediction j
        s0 = data["states"][run, starts]                    # (W, 34)   s_k
        actions = data["actions"][run, rows]                # (W, H, 12) a_{k+j-1}
        truth = data["states"][run, rows + 1]               # (W, H, 34) s_{k+j}
        contact = data["contact_step"][run, rows]           # (W, H)     transition k+j-1

        if predict is None:
            preds = rollout(step, s0, actions)              # (W, H, 34)
        elif force is None:
            preds = predict(run, starts)                    # (W, H, 34)
        else:
            preds, f = predict(run, starts)                 # (W, H, 34), (W, H, 2)
            force_pred.append(f)
            force_true.append(force[run, rows])             # (W, H, 2) force over transition k+j-1
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
    if force is not None:
        out.update(force_stats(np.concatenate(force_pred), np.concatenate(force_true), contact_all))
    return out
