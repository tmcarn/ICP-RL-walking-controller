from stable_baselines3 import PPO
from walker_env import WalkerEnv, TerrainAwareWalkerEnv
from tqdm import tqdm
import numpy as np

from matplotlib import pyplot as plt


def evaluate(eval_env, policy=None, total_steps=100_000):
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

        if info["reward/total_lin_vel_error"] < 10: # Filters out Simulation Divergence
            lin_vel_errors.append(info["reward/total_lin_vel_error"])
        
        if info["reward/total_yaw_vel_error"] < 10: # Filters out Simulation Divergence
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
        "episode_length_std": ep_arr.std()
    }

def plot_terrain_comparison(results, save_path=None):
    """
    Plot 2x2 grouped bar charts comparing base vs residual vs terrain-aware
    policy across terrain types.
    """
    terrains = list(results.keys())
    x = np.arange(len(terrains))
    width = 0.25  # narrower to fit 3 bars

    labels = ["Base controller", "Residual RL policy", "Terrain-Aware policy"]
    keys = ["base", "res_policy", "terrain_policy" ]
    colors = ["tab:gray", "tab:blue", "tab:green"]
    offsets = [-width, 0, width]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    metrics = [
        (axes[0, 0], "episode_length_mean", "episode_length_std", "Steps", "Episode length"),
        (axes[0, 1], "total_terminations", None, "# of terminations", "Terminations"),
        (axes[1, 0], "lin_vel_error_mean", "lin_vel_error_std", "Error (m/s)", "Linear velocity error"),
        (axes[1, 1], "yaw_vel_error_mean", "yaw_vel_error_std", "Error (rad/s)", "Yaw velocity error"),
    ]

    for ax, val_key, std_key, ylabel, title in metrics:
        for i, (key, label, color, offset) in enumerate(zip(keys, labels, colors, offsets)):
            vals = [results[t][key][val_key] for t in terrains]
            bar_kwargs = dict(width=width, label=label, color=color)
            if std_key:
                errs = [results[t][key][std_key] for t in terrains]
                bar_kwargs.update(yerr=errs, capsize=4, error_kw={"linewidth": 1.2})
            ax.bar(x + offset, vals, **bar_kwargs)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in terrains], fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle("Controller comparison across terrain types", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()

def run_terrain_comparison(terrain_types, res_path, terrain_path):
    eval_data = {}
    res_policy = PPO.load(res_path)
    terrain_policy = PPO.load(terrain_path)
    for terrain in terrain_types:
        print(f"Evaluating with {terrain} terrain...")
        terrain_env = TerrainAwareWalkerEnv(render_mode=None, terrain_types=[terrain])
        res_env = WalkerEnv(render_mode=None, terrain_types=[terrain])
        base_evaluation = evaluate(res_env)
        res_policy_evaluation = evaluate(res_env, res_policy)
        terrain_policy_evaluation = evaluate(terrain_env, terrain_policy)

        eval_data[terrain] = {}
        eval_data[terrain]["base"] = base_evaluation
        eval_data[terrain]["res_policy"] = res_policy_evaluation
        eval_data[terrain]["terrain_policy"] = terrain_policy_evaluation

    return eval_data



res_model_path = "./checkpoints/terrain_walker_v1/residual_rl_15000000_steps.zip"
terrain_model_path = "./checkpoints/terrainwalker_terrain_non_random/residual_rl_15000000_steps.zip"

results = run_terrain_comparison(["flat", "moderate", "rough"], res_model_path, terrain_model_path)
plot_terrain_comparison(results, save_path="./plots/New_Controller_Comparison.png")