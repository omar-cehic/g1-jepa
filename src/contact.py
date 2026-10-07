"""Landing definition and contact-step rules, shared by the analysis and evaluation code.

Landing: a False -> True change of the 500 Hz contact label of one foot where
the foot was off for at least min_off_s before and stays on for at least
min_on_s after. Landing index i = the first 500 Hz state with contact, so the
landing time is t_fine[i] (the touchdown happened during physics step i-1 -> i).

Landings whose windows are not fully observed are dropped: if fewer than
min_off_s of data exist before the change (start of the run), or fewer than
min_on_s after it (end of the run, including a run cut short by a fall), we
cannot tell whether it is a landing, so it is not counted.

Index convention: 500 Hz label i describes the state at t_fine[i], and 50 Hz
row k is the state at 500 Hz index step[k] = 10 (k + 1).
"""

import numpy as np


def detect_landings(contact, t_fine, min_off_s=0.05, min_on_s=0.05):
    """Landing indices (into the 500 Hz arrays) for each foot.

    contact: (N, n_feet) bool, the 500 Hz contact label. t_fine: (N,) times.
    Returns a list with one int array of landing indices per foot.
    """
    dt = t_fine[1] - t_fine[0]
    n_off = round(min_off_s / dt)
    n_on = round(min_on_s / dt)
    landings = []
    for f in range(contact.shape[1]):
        label = contact[:, f]
        out = []
        for i in np.flatnonzero(~label[:-1] & label[1:]) + 1:  # every False -> True change
            if i - n_off < 0 or i + n_on > len(label):
                continue  # window not fully observed
            if not label[i - n_off : i].any() and label[i : i + n_on].all():
                out.append(i)
        landings.append(np.array(out, dtype=int))
    return landings


def contact_steps(step, landing_idx, window):
    """Boolean per transition k -> k+1 (k = 0 .. len(step) - 2): is it a contact step?

    Transition k spans the 500 Hz states step[k] .. step[k+1]. It is a contact
    step if a landing index i satisfies step[k] - window <= i <= step[k+1] + window,
    i.e. a landing within window samples (40 ms = 20 samples) of the transition's interval.
    """
    return np.array([np.any((landing_idx >= step[k] - window) & (landing_idx <= step[k + 1] + window))
                     for k in range(len(step) - 1)], dtype=bool)


def transition_zero(step, landing_idx):
    """Index k of the transition containing each touchdown: the physics step from state i-1
    (no contact) to state i (contact) lies inside k -> k+1, i.e. step[k] <= i - 1 < step[k+1]."""
    return np.searchsorted(step, landing_idx - 1, side="right") - 1
