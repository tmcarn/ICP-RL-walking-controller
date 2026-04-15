import argparse
import glob
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from walker_env import WalkerEnv


def find_latest_run(runs_dir="./runs"):
    runs = sorted(glob.glob(os.path.join(runs_dir, "[0-9]*")))
    return runs[-1] if runs else None


parser = argparse.ArgumentParser()
parser.add_argument("--run_dir",   default=None, help="Path to run directory (default: latest in ./runs)")
parser.add_argument("--push_mag",  type=float,   default=0.0, help="Push disturbance magnitude")
parser.add_argument("--duration",  type=float,   default=30.0, help="Playback duration in seconds")
parser.add_argument("--base_only", action="store_true", help="Run base ICP controller only (no RL)")
args = parser.parse_args()

run_dir = args.run_dir or find_latest_run()
if run_dir is None:
    raise RuntimeError("No run directory found under ./runs. Pass --run_dir explicitly.")
print(f"Run: {run_dir}")

# Raw env for rendering
raw_env = WalkerEnv(render_mode="human", base_controller_only=args.base_only)

# Build a DummyVecEnv to carry VecNormalize stats
dummy = DummyVecEnv([lambda: WalkerEnv(render_mode=None)])
candidates = sorted(glob.glob(os.path.join(run_dir, "**", "vecnormalize*.pkl"), recursive=True))
if candidates:
    vecnorm_path = candidates[-1]
    print(f"VecNormalize: {os.path.relpath(vecnorm_path, run_dir)}")
    norm_env = VecNormalize.load(vecnorm_path, dummy)
    norm_env.training = False
    norm_env.norm_reward = False

    def normalize_obs(obs):
        return norm_env.normalize_obs(obs)
else:
    print("WARNING: No VecNormalize stats found — obs will not be normalized.")
    norm_env = None

    def normalize_obs(obs):
        return obs

if not args.base_only:
    model_path = os.path.join(run_dir, "best_model", "best_model.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No best_model at {model_path}")
    print(f"Model: {model_path}")
    policy = PPO.load(model_path)
else:
    policy = None
    print("Running base controller only.")

raw_env.set_push_magnitude(args.push_mag)
print(f"Push magnitude: {args.push_mag}")

obs, _ = raw_env.reset()
timestep_count = int(args.duration / (raw_env.dt * raw_env.decimation))

for _ in range(timestep_count):
    if policy is not None:
        norm = normalize_obs(obs[np.newaxis])[0]
        action, _ = policy.predict(norm, deterministic=True)
    else:
        action = np.zeros(raw_env.action_space.shape, dtype=np.float32)

    obs, reward, terminated, truncated, info = raw_env.step(action)
    raw_env.render()
    time.sleep(1 / raw_env.control_freq)

    if terminated or truncated:
        obs, _ = raw_env.reset()

raw_env.close()
if norm_env is not None:
    norm_env.close()
