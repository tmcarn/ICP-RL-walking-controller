import argparse
import glob
import os

import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from walker_env import WalkerEnv


def find_latest_run(runs_dir="./runs"):
    runs = sorted(glob.glob(os.path.join(runs_dir, "[0-9]*")))
    return runs[-1] if runs else None


def make_eval_env(run_dir, push_magnitude):
    raw = DummyVecEnv([lambda: WalkerEnv(render_mode=None)])

    candidates = sorted(glob.glob(os.path.join(run_dir, "**", "vecnormalize*.pkl"), recursive=True))
    if candidates:
        vecnorm_path = candidates[-1]
        print(f"  VecNormalize: {os.path.relpath(vecnorm_path, run_dir)}")
        env = VecNormalize.load(vecnorm_path, raw)
        env.training = False
        env.norm_reward = False
    else:
        print("  WARNING: No VecNormalize stats found — obs will not be normalized. Results may be unreliable.")
        env = raw

    env.env_method("set_push_magnitude", push_magnitude)
    return env


def run_eval(policy, env, total_steps, label):
    obs = env.reset()
    terminations = 0
    ep_lens = []
    current_len = 0
    lin_errs, yaw_errs, orients = [], [], []

    for _ in tqdm(range(total_steps), desc=label, leave=False):
        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = np.zeros((1, env.action_space.shape[0]), dtype=np.float32)

        obs, _, dones, infos = env.step(action)
        info = infos[0]

        lin_errs.append(info["reward/total_lin_vel_error"])
        yaw_errs.append(info["reward/total_yaw_vel_error"])
        orients.append(info["reward/orientation"])
        current_len += 1

        if dones[0]:
            ep_lens.append(current_len)
            current_len = 0
            if not info.get("TimeLimit.truncated", False):
                terminations += 1

    if current_len > 0:
        ep_lens.append(current_len)

    ep_arr = np.array(ep_lens) if ep_lens else np.array([0])
    return {
        "total_episodes":    len(ep_lens),
        "terminations":      terminations,
        "ep_len_mean":       ep_arr.mean(),
        "ep_len_std":        ep_arr.std(),
        "ep_len_min":        int(ep_arr.min()),
        "ep_len_max":        int(ep_arr.max()),
        "lin_vel_err_mean":  float(np.mean(lin_errs)),
        "yaw_vel_err_mean":  float(np.mean(yaw_errs)),
        "orientation_mean":  float(np.mean(orients)),
    }


def print_sweep_table(push_mags, residual_results, base_results):
    w = 100
    print("\n" + "=" * w)
    print("ROBUSTNESS SWEEP: Residual RL vs Base Controller")
    print("=" * w)
    print(f"{'Push':>6}  {'Ep Len (RL)':>16}  {'Ep Len (base)':>16}  {'Falls RL':>9}  {'Falls base':>10}  {'Lin Err RL':>11}  {'Lin Err base':>12}")
    print("-" * w)
    for pm, res, base in zip(push_mags, residual_results, base_results):
        res_len  = f"{res['ep_len_mean']:5.0f} ± {res['ep_len_std']:4.0f}"
        base_len = f"{base['ep_len_mean']:5.0f} ± {base['ep_len_std']:4.0f}"
        print(
            f"{pm:>6.2f}  {res_len:>16}  {base_len:>16}"
            f"  {res['terminations']:>9d}  {base['terminations']:>10d}"
            f"  {res['lin_vel_err_mean']:>11.4f}  {base['lin_vel_err_mean']:>12.4f}"
        )
    print("=" * w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",   default=None,  help="Path to run directory (default: latest in ./runs)")
    parser.add_argument("--steps",     type=int,      default=10_000, help="Sim steps per push level")
    parser.add_argument("--push_mags", type=float,    nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0],
                        help="Push magnitudes to sweep over")
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run()
    if run_dir is None:
        raise RuntimeError("No run directory found under ./runs. Pass --run_dir explicitly.")
    print(f"\nEvaluating run: {run_dir}")

    model_path = os.path.join(run_dir, "best_model", "best_model.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No best_model found at {model_path}")

    residual_results = []
    base_results = []

    for pm in args.push_mags:
        print(f"\n--- push_magnitude = {pm:.2f} ---")

        env_res = make_eval_env(run_dir, pm)
        policy = PPO.load(model_path, env=env_res)
        res = run_eval(policy, env_res, args.steps, f"RL   push={pm:.2f}")
        residual_results.append(res)
        env_res.close()

        env_base = make_eval_env(run_dir, pm)
        base = run_eval(None, env_base, args.steps, f"base push={pm:.2f}")
        base_results.append(base)
        env_base.close()

    print_sweep_table(args.push_mags, residual_results, base_results)
