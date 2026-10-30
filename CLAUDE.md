# CLAUDE.md

## What this repo is
Course project for CSC 383/483 Deep Learning (DePaul). I compare two learned world models of the Unitree G1 humanoid walking in MuJoCo:
- an MLP that predicts the change in state from (state, action)
- a JEPA (following LeWorldModel): MLP encoder, MLP predictor that also takes the action, loss = embedding prediction error + lambda * SIGReg, trained end to end with no EMA target encoder; stop gradient on the target is an option (--stop_grad), off by default

The question is which model predicts better around foot contact, over 10, 50, and 100 step rollouts. Final report due November 12, 2026.

## About me
I am a CS junior and I need to understand every line of this code, because I will explain it to my professor, who is an expert in contact simulation. When you write or change code:
- Explain what you did and why in plain language at the end, briefly.
- Prefer simple, readable code over clever code.
- Tell me when you are unsure about something (an API, a physics detail, a paper's method) instead of guessing.

## Environment
- WSL2, Ubuntu 24.04, Python virtual environment.
- Data source: unitree_rl_gym (deploy/deploy_mujoco). Its G1 config has 12 leg actions, sim timestep 0.002 s, control decimation 10 (50 Hz policy), gait period 0.8 s.
- Training may happen on Colab if there is no local GPU. Keep training scripts runnable there.

## Layout (to be created, adjust as needed)
- `configs/` YAML configs for data collection, models, training, evaluation
- `scripts/collect_data.py` runs the walking policy and logs data
- `src/models/` mlp.py, jepa.py, sigreg.py
- `src/train.py` trains either model from a config
- `src/eval/` rollouts, probes, metrics, contact split
- `notebooks/` quick analysis and plots
- `data/` and `results/` are gitignored

## Rules
- Keep it simple. Write the simplest code that does the job. No extra abstractions, base classes, plugin systems, or config frameworks until there is a real need.
- Only do what the task asks. No unrequested features, options, or refactors of unrelated code.
- Prefer few dependencies. Do not add a library for something a few lines can do.
- Check real APIs and files before using them. Do not guess names.
- Run the code before saying it works, and say what you tested.
- Log raw simulator data (full qpos, qvel, actions, commands, gait phase, pushes, per foot contact). Build model features in a separate step.
- Never overwrite existing data or results. Write to new, dated folders.
- Every run takes a seed and sets it for numpy, torch, and the simulator. Save the config with the results.
- Split data by run (episode), never by step: 80% train, 10% tuning, 10% test.
- Keep the MLP and JEPA comparable: same data, optimizer, training budget, and the same width and depth as the MLP (3 hidden layers of 1024, ReLU). Flag anything that breaks this.
- Do not change the evaluation protocol (horizons, metrics, 40 ms contact window, probe setup) without asking me first. It is what my proposal promised. Agreed change (10/20): contact force replaced foot landing times as a metric.
- Small commits with clear messages. Do not commit data, checkpoints, or large files.
- Add a quick sanity test for anything math heavy (SIGReg, rollout code, metric code) before using it on real data.
- Never delete, move, or rename anything in data/ or results/. If cleanup seems needed, ask me first.
- data/raw/2026-10-07_164422, data/raw/2026-10-07_144241 and data/processed/2026-10-07_182522 are read only on purpose. Do not change their permissions.
- The official training dataset is data/processed/2026-10-07_182522 (commit 1823ea4).

## Commands
(Fill in as the project grows: how to collect data, train, evaluate.)
- Collect data: `python scripts/collect_data.py configs/collect_data.yaml` (writes data/raw/<timestamp>/)
- Summarize a batch: `python scripts/summarize_batch.py data/raw/<folder>`
- Plot one run: `python scripts/plot_test_batch.py data/raw/<folder> --run 0` (writes results/<timestamp>_test_batch_plots/)
- Check two batches are identical: `python scripts/compare_batches.py data/raw/<a> data/raw/<b>`
- Build the training dataset: `python scripts/build_dataset.py configs/build_dataset.yaml` (writes data/processed/<timestamp>/)
- Build the full state dataset (raw qpos and qvel, 37 numbers): `python scripts/build_dataset.py configs/build_dataset_full_state.yaml` (writes data/processed/<timestamp>/; current one is data/processed/2026-10-28_175622)
- Build the full state dataset without pelvis x and y (qpos[2:] and qvel, 35 numbers): `python scripts/build_dataset.py configs/build_dataset_full_no_xy.yaml` (writes data/processed/<timestamp>/; current one is data/processed/2026-10-29_162623)
- Full state dataset test (full or full_no_xy): `python tests/test_full_state_dataset.py data/processed/<full state folder>`
- Train on the full_no_xy dataset: `python src/train_jepa.py configs/train_jepa_full_no_xy.yaml` and `python src/train.py configs/train_mlp_full_no_xy.yaml` (run folders get "_fullnoxy")
- Contact exploration plots: `python scripts/explore_contact.py data/raw/<folder>` (writes results/<timestamp>_explore_contact/)
- Logger tests (plain scripts, no pytest): `for t in tests/test_*.py; do python $t || break; done`
- SIGReg test (synthetic data, about 2.5 min on CPU): `python tests/test_sigreg.py`
- Train the JEPA: `python src/train_jepa.py configs/train_jepa.yaml [--seed S] [--embed_dim D] [--sigreg_weight W] [--lr LR] [--epochs E]` (writes results/train/<timestamp>_jepa_d<D>_sw<W>_lr<LR>_s<S>/; about 6.5 s per epoch on the RTX 3060)
- JEPA test (real data, about 1.3 min on CPU): `python tests/test_jepa.py`
- Score JEPA runs with the linear and MLP probes: `python scripts/score_jepa.py results/train/<run folder> [...]` (writes results/<timestamp>_score_jepa/; about 22 s per run on the RTX 3060)
- Probe diagnostics for JEPA runs (per feature errors, train vs tuning, bigger probe, probe seeds): `python scripts/probe_diagnostics.py results/train/<run folder> [...]` (writes results/<timestamp>_probe_diag/; about 3.5 to 7.5 min per run on the RTX 3060, mostly the bigger probe)
- Probe test (synthetic data plus real data, about 70 s with GPU): `python tests/test_probe.py`
- Readout controls (untrained JEPA encoder probed, and MLP world model one step error, both in normalized state units): `python scripts/readout_controls.py` (writes results/<timestamp>_readout_controls/; about 1.2 min on the RTX 3060)
