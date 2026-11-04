"""Roll out a model on one split of the dataset and save the per step RMS error curves.

Uses every usable window (H = 100, stride 1) of every run in the split; see src/eval/rollout.py.
Also saves and prints robust statistics: the median and 90th percentile over windows of the per
window group error, and the fraction of windows that have left the train range (train_range).

Models (--model):
  constant_velocity, mlp   one step models: s_hat_{k+j+1} = step(s_hat_{k+j}, a_{k+j}) (rollout()).
  train_mean               reference: ignores the window and predicts the same state at every step j, the mean
                           raw state over the valid train rows (data["states"][run, k] for every valid train
                           transition k).
  transformer              transformer world model from src/train.py, history n from model.pt. The window
                           starts as the true s_{k-n+1}, ..., s_k with the recorded a_{k-n+1}, ..., a_k.
                           Each step normalizes the state and action windows, the network gives the
                           normalized change of the newest state, which is denormalized (delta stats) and
                           added to the newest state in float64. The new state is appended, the oldest
                           dropped, and the action window moves on by one recorded action (rollout_window()).
  jepa                     JEPA from src/train_jepa.py (predictor type and history n from model.pt).
                           e = encode(normalized s) of every state of the run, each on its own. The window
                           starts as the true e_{k-n+1}, ..., e_k and slides like the transformer's, with
                           e_hat_{k+j} = model.predict(e_hat_{k+j-1}, normalized action(s), older embeddings).
                           Predictions are never decoded and re-encoded. They are decoded for the errors
                           with an MLP probe (fit_mlp_probe, seed 0, 20 epochs, as in score_jepa.py) fit on
                           encode(s_k) -> normalized s_k of the valid train pairs, then denormalized.
                           Readout floor: the same probe on the true e_{k+j}, saved with the prefix floor_.
Networks run in eval mode, without gradients, in float32. Only recorded actions; predictions are fed back.

Foot contact force (all models except constant_velocity), the "force" group of metrics.npz, in N per foot.
The truth is the mean force per foot over each transition, from FORCE (scripts/build_force.py).
  Force readout (mlp, transformer, jepa): an MLP (fit_mlp_probe, seed 0, 20 epochs) from [x_k, normalized a_k]
  to the normalized force of transition k, fit on the valid train transitions (force mean and std over the
  same transitions). x is the normalized state s_k for mlp and transformer, the embedding encode(s_k) for jepa.
  Prediction j: the readout on (x of row k+j-1, recorded a_{k+j-1}), where x of row k is the true x_k (j = 1)
  and x of row k+j-1 for j >= 2 is the model's prediction j - 1. It predicts the force over transition k+j-1,
  the transition whose contact_step label decides whether prediction j is a contact step.
  Readout floor: the same readout on the true x of row k+j-1, saved as floor_<stat>_<subset>_force.
  train_mean: the mean force per foot over the valid train transitions, at every j.

--split (train, tuning or test, default tuning) only picks the runs whose windows are evaluated. Everything that
is fit or measured on data (train range, train mean, force readout, JEPA probe) always uses the train runs.

Usage: python scripts/eval_rollouts.py --model constant_velocity|train_mean [--split tuning|test]
       python scripts/eval_rollouts.py --model mlp|transformer|jepa --run_dir results/train/<run folder> [--split tuning|test]
Output: results/rollouts/<YYYY-MM-DD_HHMMSS>_<model>_<split>/ with metrics.npz and meta.json
        (with --run_dir: <YYYY-MM-DD_HHMMSS>_<model>_<run folder name>_<split>/).
"""

import argparse
import datetime
import json
import os
import resource
import sys

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from build_dataset import git_commit  # noqa: E402
from eval.probe import fit_mlp_probe, apply_mlp_probe, MLP_PROBE_WIDTH, MLP_PROBE_DEPTH  # noqa: E402
from eval.rollout import evaluate, evaluate_force, train_range, rollout, rollout_window, SUBSETS  # noqa: E402
from features import STATE_NAMES, STATE_GROUPS  # noqa: E402
from models import constant_velocity  # noqa: E402
from models.mlp import make_mlp, make_step, normalize, denormalize  # noqa: E402
from score_jepa import load_jepa, embed_windows  # noqa: E402
from train import make_model  # noqa: E402
from train_jepa import window_pairs  # noqa: E402

DATASET = os.path.join("data", "processed", "2026-10-07_182522", "dataset.npz")
FORCE = os.path.join("data", "processed", "2026-11-04_113348_foot_force", "foot_force.npz")  # built from DATASET
MODELS = {"constant_velocity": constant_velocity.step}
H = 100
REPORT_STEPS = [1, 10, 50, 100]
PROBE_SEED = 0
PROBE_EPOCHS = 20   # fit_mlp_probe's default, as in score_jepa.py
NO_RUN_DIR = sorted(MODELS) + ["train_mean"]   # models that need no --run_dir


def train_mean(data):
    """Mean raw state (34,) over the valid train rows: data["states"][run, k] for every valid train transition k."""
    train = data["split"] == "train"
    s_k = data["states"][train][:, :-1][data["valid"][train]]  # (n_valid, 34)
    return s_k.mean(axis=0)


def train_mean_predict(data, H=H):
    """predict(run, starts) for evaluate() that ignores the window: every prediction is train_mean(data).
    Returns predict and the mean."""
    mean = train_mean(data)

    def predict(run, starts):
        return np.tile(mean, (len(starts), H, 1))  # (W, H, 34)
    return predict, mean


def load_force(data):
    """Mean force per foot per transition (runs, transitions, 2) in N from FORCE, in the order of data's runs.
    Checks that FORCE was built from DATASET and has the same run ids."""
    with open(os.path.join(REPO_ROOT, os.path.dirname(FORCE), "meta.json"), "r") as f:
        source = json.load(f)["source_dataset"]
    assert os.path.normpath(source) == os.path.normpath(DATASET), f"{FORCE} was built from {source}"
    saved = np.load(os.path.join(REPO_ROOT, FORCE))
    assert np.array_equal(saved["run_ids"], data["run_ids"]), "FORCE and DATASET have different run ids"
    assert saved["foot_force"].shape == data["valid"].shape + (2,)
    return saved["foot_force"]


def force_train_mean(data, force):
    """Mean and std (2,) of the force per foot over the valid train transitions."""
    train = data["split"] == "train"
    f = force[train][data["valid"][train]]   # (n_valid, 2)
    return f.mean(axis=0), f.std(axis=0)


def fit_force_readout(x_of_run, data, force, device, epochs=PROBE_EPOCHS):
    """Fit the force readout on the valid train transitions. Returns readout(x, actions) -> force in N.

    x_of_run(run) gives x (rows, dx) of every row of a run (normalized states, or embeddings). The network is
    fit_mlp_probe (seed PROBE_SEED) from [x_k, normalized a_k] to the normalized force of transition k.
    readout(x (..., dx), raw actions (..., 12)) returns (..., 2): the network output, denormalized.
    """
    mean, std = force_train_mean(data, force)
    inputs, targets = [], []
    for run in np.flatnonzero(data["split"] == "train"):
        valid = data["valid"][run]
        inputs.append(np.concatenate([x_of_run(run)[:-1][valid],
                                      normalize(data["actions"][run][valid], data, "action")], axis=1))
        targets.append((force[run][valid] - mean) / std)
    net = fit_mlp_probe(np.concatenate(inputs), np.concatenate(targets), device, seed=PROBE_SEED, epochs=epochs)

    def readout(x, actions):
        lead = x.shape[:-1]
        inp = np.concatenate([x.reshape(-1, x.shape[-1]), normalize(actions, data, "action").reshape(-1, 12)],
                             axis=1)
        return (apply_mlp_probe(net, inp, device) * std + mean).reshape(lead + (2,))
    return readout


def readout_rollout(readout, x_true, x_hat, actions, starts):
    """Force (W, H, 2) of a rollout: prediction j from (x of row k+j-1, a_{k+j-1}).

    x_true (rows, dx): true x of every row of the run; x_hat (W, H, dx): the model's predictions, index j - 1
    holding prediction j (row k+j); actions: the run's recorded actions. The x of row k+j-1 is x_true[k]
    for j = 1 and x_hat[:, j - 2] for j >= 2.
    """
    H = x_hat.shape[1]
    x_in = np.concatenate([x_true[starts][:, None], x_hat[:, :-1]], axis=1)   # (W, H, dx)
    return readout(x_in, actions[starts[:, None] + np.arange(H)])


def readout_floor(readout, x_true, actions, starts, H):
    """Force (W, H, 2) of the readout floor: prediction j from the true (x of row k+j-1, a_{k+j-1})."""
    rows = starts[:, None] + np.arange(H)
    return readout(x_true[rows], actions[rows])


def add_state_force(state_predict, data, force, device, H=H, epochs=PROBE_EPOCHS):
    """For the mlp and transformer world models: fit the force readout on normalized states and return
    predict(run, starts) -> (raw states (W, H, 34), forces (W, H, 2)) for evaluate(..., force=...), and
    floor_predict(run, starts) -> forces (W, H, 2) for evaluate_force(). state_predict(run, starts) gives the
    model's raw state predictions (W, H, 34)."""
    def x_of_run(run):
        return normalize(data["states"][run], data, "state")
    readout = fit_force_readout(x_of_run, data, force, device, epochs)

    def predict(run, starts):
        preds = state_predict(run, starts)
        x_hat = normalize(preds, data, "state")
        return preds, readout_rollout(readout, x_of_run(run), x_hat, data["actions"][run], starts)

    def floor_predict(run, starts):
        return readout_floor(readout, x_of_run(run), data["actions"][run], starts, H)
    return predict, floor_predict


def mlp_predict(step, data, H=H):
    """predict(run, starts) -> raw states (W, H, 34) of a one step model: rollout() on the same arrays that
    evaluate(step, ...) uses, so the states are the same."""
    def predict(run, starts):
        rows = starts[:, None] + np.arange(H)
        return rollout(step, data["states"][run, starts], data["actions"][run, rows])
    return predict


def check_dataset(run_dir):
    """config.yaml of run_dir as a dict. Raises ValueError if the run was trained on another dataset than DATASET
    (DATASET is the base state dataset, so this also refuses full state runs)."""
    with open(os.path.join(run_dir, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    if os.path.normpath(cfg["dataset"]) != os.path.normpath(DATASET):
        raise ValueError(f"{run_dir} was trained on {cfg['dataset']}, but this script evaluates {DATASET}")
    return cfg


def load_mlp_step(run_dir, data, device):
    """step(states, actions) of the MLP saved by src/train.py in run_dir, normalized with data's stats.

    Raises ValueError if the run was trained on a different dataset than DATASET, or is a transformer
    world model run (model "transformer" in model.pt; checkpoints saved before model existed are MLP runs).
    """
    check_dataset(run_dir)
    saved = torch.load(os.path.join(run_dir, "model.pt"), map_location=device)
    if saved.get("model", "mlp") != "mlp":
        raise ValueError(f"{run_dir} is a {saved['model']} world model run: use --model {saved['model']} in "
                         f"eval_rollouts.py (analyze_divergence.py supports MLP runs only)")
    in_dim = len(data["state_mean"]) + len(data["action_mean"])  # 34 + 12
    model = make_mlp(in_dim, len(data["delta_mean"]), saved["width"], saved["depth"]).to(device)
    model.load_state_dict(saved["state_dict"])
    return make_step(model, data, device)


def make_window_step(model, stats, device):
    """step(state windows, action windows) -> next states, of the transformer world model, for rollout_window().

    states (W, n, 34) raw float64 and actions (W, n, 12) raw, oldest first. Both are normalized here (float64,
    then float32 for the network). Returns the newest state + the denormalized predicted delta, in float64.
    """
    def step(states, actions):
        s = torch.from_numpy(normalize(states, stats, "state").astype(np.float32)).to(device)
        a = torch.from_numpy(normalize(actions, stats, "action").astype(np.float32)).to(device)
        with torch.no_grad():
            out = model(s, a).cpu().numpy()
        return states[:, -1] + denormalize(out, stats, "delta")
    return step


def load_transformer_predict(run_dir, data, device, H=H):
    """predict(run, starts) -> raw predictions (W, H, 34) of the transformer world model saved by src/train.py
    in run_dir, for evaluate(), and the model (in eval mode; its history n is model.history).

    Raises ValueError if the run was trained on another dataset than DATASET or is not a transformer run.
    """
    check_dataset(run_dir)
    saved = torch.load(os.path.join(run_dir, "model.pt"), map_location=device)
    if saved.get("model", "mlp") != "transformer":
        raise ValueError(f"{run_dir} is not a transformer world model run (model.pt has no model: transformer)")
    n = saved["history"]
    model = make_model("transformer", len(data["state_mean"]), saved["width"], saved["depth"], n).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    step = make_window_step(model, data, device)

    def predict(run, starts):
        return rollout_window(step, data["states"][run], data["actions"][run], starts, n, H)
    return predict, model


def encode_run(model, data, run, device):
    """Embeddings (steps, embed_dim) float32 of every state of one run: encode(normalized s), each state on its own."""
    s = normalize(data["states"][run], data, "state").astype(np.float32)
    with torch.no_grad():
        return model.encode(torch.from_numpy(s).to(device)).cpu().numpy()


def make_jepa_step(model, stats, device):
    """step(embedding windows, action windows) -> next embeddings, of a JEPA, for rollout_window().

    z (W, n, embed_dim) float32 and actions (W, n, 12) raw, oldest first. The actions are normalized here
    (float64, then float32). Calls model.predict(z_k, actions, older embeddings) as in training: the mlp
    predictor gets a_k alone, the transformer predictor the action window; no older embeddings for history 1.
    """
    def step(z, actions):
        z = torch.from_numpy(z).to(device)
        a = torch.from_numpy(normalize(actions, stats, "action").astype(np.float32)).to(device)
        with torch.no_grad():
            z_hat = model.predict(z[:, -1], a if model.predictor_type == "transformer" else a[:, -1],
                                  z[:, :-1] if model.history > 1 else None)
        return z_hat.cpu().numpy()
    return step


def jepa_rollout(model, data, run, starts, H, device):
    """Predicted embeddings (W, H, embed_dim): index j - 1 holds e_hat_{k+j} of the window starting at k."""
    z = encode_run(model, data, run, device)
    return rollout_window(make_jepa_step(model, data, device), z, data["actions"][run], starts, model.history, H)


def true_embeddings(z_run, starts, H):
    """True embeddings (W, H, embed_dim) for the readout floor: index j - 1 holds e_{k+j} (the row evaluate()
    compares prediction j with). z_run from encode_run()."""
    return z_run[starts[:, None] + np.arange(1, H + 1)]


def load_jepa_predicts(run_dir, data, force, device, H=H, force_epochs=PROBE_EPOCHS):
    """For the JEPA saved by src/train_jepa.py in run_dir: predict(run, starts) of the rollout (raw states
    (W, H, 34) and forces (W, H, 2), for evaluate(..., force=...)), floor_predict(run, starts) of the state
    readout floor (raw (W, H, 34), for evaluate()), floor_force(run, starts) of the force readout floor
    ((W, H, 2), for evaluate_force()), and the model.

    Fits the MLP probe encode(s_k) -> normalized s_k on the valid train pairs, as score_jepa.py does, then
    the force readout on [encode(s_k), normalized a_k] (fit_force_readout).
    Raises ValueError if the run was trained on another dataset than DATASET or is not a JEPA run.
    """
    cfg = check_dataset(run_dir)
    if "embed_dim" not in cfg:
        raise ValueError(f"{run_dir} is not a JEPA run (config.yaml has no embed_dim)")
    model = load_jepa(run_dir, device)
    states, rows, a, actions = window_pairs(data, "train", model.history)
    z_train, _ = embed_windows(model, states, rows, a, actions, device)
    probe = fit_mlp_probe(z_train, states[rows], device, seed=PROBE_SEED, epochs=PROBE_EPOCHS)
    del z_train, states, actions

    def decode(z):
        """Embeddings (W, H, embed_dim) -> raw states (W, H, 34): probe, then denormalize."""
        W, n_steps, d = z.shape
        s = apply_mlp_probe(probe, z.reshape(W * n_steps, d), device)
        return denormalize(s, data, "state").reshape(W, n_steps, s.shape[1])

    readout = fit_force_readout(lambda run: encode_run(model, data, run, device), data, force, device,
                                force_epochs)

    def predict(run, starts):
        z_hat = jepa_rollout(model, data, run, starts, H, device)
        z_true = encode_run(model, data, run, device)
        return decode(z_hat), readout_rollout(readout, z_true, z_hat, data["actions"][run], starts)

    def floor_predict(run, starts):
        return decode(true_embeddings(encode_run(model, data, run, device), starts, H))

    def floor_force(run, starts):
        return readout_floor(readout, encode_run(model, data, run, device), data["actions"][run], starts, H)
    return predict, floor_predict, floor_force, model


def report(metrics):
    print(f"windows: {metrics['n_windows']}")
    header = f"{'':12s} {'subset':8s}" + "".join(f"{'j = ' + str(j):>12s}" for j in REPORT_STEPS)
    print(header)
    for subset in SUBSETS:
        counts = "".join(f"{metrics['count_' + subset][j - 1]:12d}" for j in REPORT_STEPS)
        print(f"{'samples':12s} {subset:8s}{counts}")
    for group in STATE_GROUPS:
        for subset in SUBSETS:
            rms = metrics[f"rms_{subset}_{group}"]
            values = "".join(f"{rms[j - 1]:12.4e}" for j in REPORT_STEPS)
            print(f"{group if subset == 'all' else '':12s} {subset:8s}{values}")


def report_force(metrics, title):
    print()
    print(f"{title}: force over transition k+j-1, N per foot (RMS over windows and both feet; median and 90th pct "
          f"over windows of the per window error)")
    print(f"{'':12s} {'subset':8s} {'stat':6s}" + "".join(f"{'j = ' + str(j):>12s}" for j in REPORT_STEPS))
    for subset in SUBSETS:
        for stat in ["rms", "median", "p90"]:
            values = "".join(f"{metrics[f'{stat}_{subset}_force'][j - 1]:12.4e}" for j in REPORT_STEPS)
            label = "force" if subset == "all" and stat == "rms" else ""
            print(f"{label:12s} {subset if stat == 'rms' else '':8s} {stat:6s}{values}")


def report_robust(metrics):
    print()
    print("per window group error (RMS over the group's features in one window), over windows: median and 90th pct")
    print(f"{'':12s} {'subset':8s} {'stat':6s}" + "".join(f"{'j = ' + str(j):>12s}" for j in REPORT_STEPS))
    for group in STATE_GROUPS:
        for subset in SUBSETS:
            for stat in ["median", "p90"]:
                values = "".join(f"{metrics[f'{stat}_{subset}_{group}'][j - 1]:12.4e}" for j in REPORT_STEPS)
                label = group if subset == "all" and stat == "median" else ""
                print(f"{label:12s} {subset if stat == 'median' else '':8s} {stat:6s}{values}")
    print()
    print("diverged fraction: windows where some predicted feature except gravity_z has left the train range"
          " (or is not finite) at a step <= j")
    print(f"{'diverged':12s} {'all':8s} {'frac':6s}" + "".join(f"{metrics['diverged_frac'][j - 1]:12.4f}" for j in REPORT_STEPS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=NO_RUN_DIR + ["mlp", "transformer", "jepa"])
    parser.add_argument("--run_dir", help="results/train/<run folder>, required with --model mlp, transformer, jepa")
    parser.add_argument("--split", default="tuning", choices=["train", "tuning", "test"])
    args = parser.parse_args()
    if args.model not in NO_RUN_DIR and args.run_dir is None:
        parser.error(f"--run_dir is required with --model {args.model}")

    data = dict(np.load(os.path.join(REPO_ROOT, DATASET)))
    bounds = train_range(data)
    floor = None
    force = None if args.model in MODELS else load_force(data)
    history, predictor = 1, None
    if args.model in MODELS:
        step = MODELS[args.model]
        name = args.model
        metrics = evaluate(step, data, args.split, H, bounds=bounds)
    elif args.model == "train_mean":
        name = args.model
        state_predict, _ = train_mean_predict(data)
        force_mean, _ = force_train_mean(data, force)

        def predict(run, starts):
            return state_predict(run, starts), np.tile(force_mean, (len(starts), H, 1))
        metrics = evaluate(None, data, args.split, H, bounds=bounds, predict=predict, force=force)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}")
        run_dir = os.path.join(REPO_ROOT, args.run_dir)
        name = f"{args.model}_{os.path.basename(os.path.normpath(args.run_dir))}"
        if args.model in ["mlp", "transformer"]:
            if args.model == "mlp":
                state_predict = mlp_predict(load_mlp_step(run_dir, data, device), data)
            else:
                state_predict, model = load_transformer_predict(run_dir, data, device)
                history = model.history
            predict, floor_force = add_state_force(state_predict, data, force, device)
            metrics = evaluate(None, data, args.split, H, bounds=bounds, predict=predict, force=force)
            floor = evaluate_force(floor_force, data, force, args.split, H)
        else:
            predict, floor_predict, floor_force, model = load_jepa_predicts(run_dir, data, force, device)
            history, predictor = model.history, model.predictor_type
            metrics = evaluate(None, data, args.split, H, bounds=bounds, predict=predict, force=force)
            floor = evaluate(None, data, args.split, H, bounds=bounds, predict=floor_predict)
            floor.update(evaluate_force(floor_force, data, force, args.split, H))
    print(f"model {args.model}, split {args.split}, H = {H}, RMS error in raw units")
    report(metrics)
    report_robust(metrics)
    if force is not None:
        report_force(metrics, "foot force")
    if floor is not None:
        if args.model == "jepa":
            print("\nreadout floor: the same probe on the true embeddings encode(s_{k+j}), same windows")
            report(floor)
            report_robust(floor)
        report_force(floor, "foot force readout floor (the same readout on the true x of row k+j-1)")
        metrics = {**metrics, **{"floor_" + key: value for key, value in floor.items()}}

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "results", "rollouts", f"{stamp}_{name}_{args.split}")
    os.makedirs(out_dir)  # raises if the folder already exists: never overwrite
    np.savez(os.path.join(out_dir, "metrics.npz"), **metrics)
    commit, dirty = git_commit()
    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": DATASET,
        "git_commit": commit,
        "git_uncommitted_changes": dirty,
        "model": args.model,
        "history": history,
        "split": args.split,
        "H": H,
        "n_windows": int(metrics["n_windows"]),
        "state_names": STATE_NAMES,  # columns of rms_<subset>
    }
    if args.model not in NO_RUN_DIR:
        meta["run_dir"] = args.run_dir
    if args.model == "train_mean":
        meta["prediction"] = ("train mean: the mean raw state over the valid train rows (data['states'][run, k] "
                              "for every valid train transition k), the same at every step j")
    if args.model == "jepa":
        meta["predictor"] = predictor
        meta["probe"] = {"type": "mlp", "width": MLP_PROBE_WIDTH, "depth": MLP_PROBE_DEPTH, "epochs": PROBE_EPOCHS,
                         "seed": PROBE_SEED, "fit_on": "valid train pairs: encode(s_k) -> normalized s_k",
                         "decode": "probe output denormalized with the state stats"}
    if force is not None:
        mean, std = force_train_mean(data, force)
        meta["force"] = {"file": FORCE, "units": "N", "feet": ["left", "right"],
                         "train_mean": mean.tolist(), "train_std": std.tolist(),
                         "target": "force over transition k+j-1 for prediction j"}
        if args.model != "train_mean":
            meta["force"]["readout"] = {
                "type": "mlp", "width": MLP_PROBE_WIDTH, "depth": MLP_PROBE_DEPTH, "epochs": PROBE_EPOCHS,
                "seed": PROBE_SEED,
                "x": "embedding encode(s_k)" if args.model == "jepa" else "normalized state s_k",
                "fit_on": "valid train transitions: [x_k, normalized a_k] -> (force_k - train_mean) / train_std",
                "rollout_input": "x of row k+j-1 (true for j = 1, prediction j - 1 after), recorded a_{k+j-1}",
                "floor_input": "true x of row k+j-1, recorded a_{k+j-1}"}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # ru_maxrss is in kB on Linux
    print(f"\nwrote {out_dir}")
    print(f"peak memory {peak_mb:.0f} MB")


if __name__ == "__main__":
    main()
