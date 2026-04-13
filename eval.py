from stable_baselines3 import PPO
from walker_env import WalkerEnv
from tqdm import tqdm
import numpy as np


def evaluate(policy=None, total_steps=100_000):
    eval_env = WalkerEnv(render_mode=None)
    obs, _ = eval_env.reset()

    total_terminations = 0
    episode_lengths = []
    current_episode_len = 0
    lin_vel_errors = []
    yaw_vel_errors = []
    orientations = []

    for _ in tqdm(range(total_steps)):
        if policy:
            action, _ = policy.predict(obs)
        else:
            action = np.zeros(eval_env.action_space.shape)

        obs, reward, terminated, truncated, info = eval_env.step(action)

        lin_vel_errors.append(info["reward/total_lin_vel_error"])
        yaw_vel_errors.append(info["reward/total_yaw_vel_error"])
        orientations.append(info["reward/orientation"])
        current_episode_len += 1

        if terminated or truncated:
            episode_lengths.append(current_episode_len)
            current_episode_len = 0
            obs, _ = eval_env.reset()
            if terminated:
                total_terminations += 1

    if current_episode_len > 0:
        episode_lengths.append(current_episode_len)

    lin_arr = np.array(lin_vel_errors)
    yaw_arr = np.array(yaw_vel_errors)
    ep_arr = np.array(episode_lengths)
    ori_arr = np.array(orientations)

    return {
        "total_steps": total_steps,
        "total_episodes": len(episode_lengths),
        "total_terminations": total_terminations,
        "lin_vel_error_mean": lin_arr.mean(),
        "lin_vel_error_std": lin_arr.std(),
        "lin_vel_error_max": lin_arr.max(),
        "yaw_vel_error_mean": yaw_arr.mean(),
        "yaw_vel_error_std": yaw_arr.std(),
        "yaw_vel_error_max": yaw_arr.max(),
        "orientation_mean": ori_arr.mean(),
        "orientation_std": ori_arr.std(),
        "orientation_min": ori_arr.min(),
        "episode_length_mean": ep_arr.mean(),
        "episode_length_std": ep_arr.std(),
        "episode_length_min": ep_arr.min(),
        "episode_length_max": ep_arr.max(),
    }


def print_comparison(base_stats, residual_stats):
    header = f"{'Metric':<30} {'Base':>20} {'Residual':>20} {'Delta':>15}"
    print("=" * 85)
    print("CONTROLLER COMPARISON")
    print("=" * 85)
    print(header)
    print("-" * 85)

    rows = [
        ("Total Episodes",      "total_episodes",      None,                  "d"),
        ("Terminations",        "total_terminations",  None,                  "d"),
        ("Ep Length",           "episode_length_mean", "episode_length_std",  ".1f"),
        ("Lin Vel Error",      "lin_vel_error_mean",  "lin_vel_error_std",   ".4f"),
        ("Yaw Vel Error",      "yaw_vel_error_mean",  "yaw_vel_error_std",   ".4f"),
        ("Orientation",        "orientation_mean",     "orientation_std",     ".4f"),
    ]

    for label, mean_key, std_key, fmt in rows:
        if mean_key is None:
            b_str = f"{base_stats['episode_length_min']}/{base_stats['episode_length_max']}"
            r_str = f"{residual_stats['episode_length_min']}/{residual_stats['episode_length_max']}"
            print(f"{label:<30} {b_str:>20} {r_str:>20} {'':>15}")
        elif fmt == "d":
            b = base_stats[mean_key]
            r = residual_stats[mean_key]
            print(f"{label:<30} {b:>20d} {r:>20d} {r - b:>+15d}")
        elif std_key:
            b_m, b_s = base_stats[mean_key], base_stats[std_key]
            r_m, r_s = residual_stats[mean_key], residual_stats[std_key]
            b_str = f"{b_m:{fmt}} ± {b_s:{fmt}}"
            r_str = f"{r_m:{fmt}} ± {r_s:{fmt}}"
            delta = r_m - b_m
            print(f"{label:<30} {b_str:>20} {r_str:>20} {delta:>+15{fmt}}")
        else:
            b = base_stats[mean_key]
            r = residual_stats[mean_key]
            print(f"{label:<30} {b:>20{fmt}} {r:>20{fmt}} {r - b:>+15{fmt}}")

    print("=" * 85)

model_path = "./best_model/best_model.zip"
policy = PPO.load(model_path)

print("\nEvaluating residual controller...")
residual_stats = evaluate(policy)

print("\nEvaluating base controller...")
base_stats = evaluate()

print_comparison(base_stats, residual_stats)