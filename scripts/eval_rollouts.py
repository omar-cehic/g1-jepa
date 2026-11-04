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

Usage: python scripts/eval_rollouts.py --model constant_velocity|train_mean [--split tuning]
       python scripts/eval_rollouts.py --model mlp|transformer|jepa --run_dir results/train/<run folder> [--split tuning]
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
from eval.rollout import evaluate, train_range, rollout_window, SUBSETS  # noqa: E402
from features import STATE_NAMES, STATE_GROUPS  # noqa: E402
from models import constant_velocity  # noqa: E402
from models.mlp import make_mlp, make_step, normalize, denormalize  # noqa: E402
from score_jepa import load_jepa, embed_windows  # noqa: E402
from train import make_model  # noqa: E402
from train_jepa import window_pairs  # noqa: E402

DATASET = os.path.join("data", "processed", "2026-10-07_182522", "dataset.npz")
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


def load_jepa_predicts(run_dir, data, device, H=H):
    """For the JEPA saved by src/train_jepa.py in run_dir: predict(run, starts) of the rollout and of the
    readout floor (both raw (W, H, 34), for evaluate()), and the model.

    Fits the MLP probe encode(s_k) -> normalized s_k on the valid train pairs, as score_jepa.py does.
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

    def predict(run, starts):
        return decode(jepa_rollout(model, data, run, starts, H, device))

    def floor_predict(run, starts):
        return decode(true_embeddings(encode_run(model, data, run, device), starts, H))
    return predict, floor_predict, model


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
    parser.add_argument("--split", default="tuning", choices=["train", "tuning"])
    args = parser.parse_args()
    if args.model not in NO_RUN_DIR and args.run_dir is None:
        parser.error(f"--run_dir is required with --model {args.model}")

    data = dict(np.load(os.path.join(REPO_ROOT, DATASET)))
    bounds = train_range(data)
    floor = None
    history, predictor = 1, None
    if args.model in MODELS:
        step = MODELS[args.model]
        name = args.model
        metrics = evaluate(step, data, args.split, H, bounds=bounds)
    elif args.model == "train_mean":
        name = args.model
        predict, _ = train_mean_predict(data)
        metrics = evaluate(None, data, args.split, H, bounds=bounds, predict=predict)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}")
        run_dir = os.path.join(REPO_ROOT, args.run_dir)
        name = f"{args.model}_{os.path.basename(os.path.normpath(args.run_dir))}"
        if args.model == "mlp":
            metrics = evaluate(load_mlp_step(run_dir, data, device), data, args.split, H, bounds=bounds)
        elif args.model == "transformer":
            predict, model = load_transformer_predict(run_dir, data, device)
            history = model.history
            metrics = evaluate(None, data, args.split, H, bounds=bounds, predict=predict)
        else:
            predict, floor_predict, model = load_jepa_predicts(run_dir, data, device)
            history, predictor = model.history, model.predictor_type
            metrics = evaluate(None, data, args.split, H, bounds=bounds, predict=predict)
            floor = evaluate(None, data, args.split, H, bounds=bounds, predict=floor_predict)
    print(f"model {args.model}, split {args.split}, H = {H}, RMS error in raw units")
    report(metrics)
    report_robust(metrics)
    if floor is not None:
        print("\nreadout floor: the same probe on the true embeddings encode(s_{k+j}), same windows")
        report(floor)
        report_robust(floor)
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
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # ru_maxrss is in kB on Linux
    print(f"\nwrote {out_dir}")
    print(f"peak memory {peak_mb:.0f} MB")


if __name__ == "__main__":
    main()
