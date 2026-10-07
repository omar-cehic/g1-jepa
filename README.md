# g1-jepa
## Setup
- unitree_rl_gym is cloned next to this repo (../unitree_rl_gym) at commit 276801e (Jul 25 2025). It is not pip installed.
- Environment: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Check the walking policy: `cd ../unitree_rl_gym && PYTHONPATH=$PWD python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml`
