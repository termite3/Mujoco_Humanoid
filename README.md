# Humanoid PPO with Mujoco (termPro)

PPO training and rollout scripts for the `Humanoid-v5` Mujoco task using Stable-Baselines3. The repo contains a custom training loop with cosine LR schedule, multi-env setup, and callbacks that save the best models by episode length or reward.

## Project layout
- `train.py`: PPO training entry point (default `Humanoid-v5`), custom hyperparameters, cosine LR schedule, VecMonitor logging.
- `callback.py`: Callbacks to save the best model by episode length or average reward; simple early-stopping helper.
- `run.py`: Rendered rollout of a saved PPO policy (loads a chosen checkpoint and runs human-rendered episodes).
- `models/`: Saved checkpoints (created at runtime).
- `logs/`: Monitor/TensorBoard logs (created at runtime).
- `assets/`: Screenshots and a humanoid GIF for reports.
- `requirements.txt`: Exact Python dependencies.

## Prerequisites
- Python 3.10+ recommended.
- Mujoco installed and GPU drivers/CUDA that match the pinned `torch` version if you want GPU training.
- Set `MUJOCO_GL=egl` (or `osmesa`) if you train on a headless server.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Training
Default training runs 5,000,000 timesteps on `Humanoid-v5` with 32 parallel envs and logs to `logs/Humanoid-v5`:
```bash
python train.py
```
Key behaviors:
- Model name includes total steps, net arch, LR schedule, gamma, and timestamp.
- Best-by-episode-length model is saved under `logs/Humanoid-v5/best_ep_len_model/`.
- Final checkpoint is saved to `models/<model_name>/<lifespan>` where `<lifespan>` is the mean of the last 10 episode lengths.

To change env or steps, edit the `train(env_id, ...)` call at the bottom of `train.py` (e.g., different `total_timesteps`, `n_envs`, or `net_arch`).

## Evaluation / Rendering
`run.py` loads a specific checkpoint and renders episodes with `render_mode='human'`:
```bash
python run.py
```
Edit `model_path` in `run.py` to point to the checkpoint you want to visualize. Use a GUI-capable session (X11/Wayland) or set up virtual display tooling if running remotely.

## Logging and TensorBoard
TensorBoard logs are under `logs/<env_id>/`. To inspect:
```bash
tensorboard --logdir logs/Humanoid-v5 --bind_all
```
Open the shown URL in a browser.

## Notes
- `makefile` currently only stores placeholder credentials; it is not used by the training scripts.
- The exact Mujoco XML used is `humanoid.xml` from Gymnasium/Mujoco; adjust kwargs in `train.py`/`run.py` for different dynamics or rewards.
