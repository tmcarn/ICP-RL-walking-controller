from stable_baselines3 import PPO
from walker_env import WalkerEnv, TerrainAwareWalkerEnv
from tqdm import tqdm
import numpy as np

from matplotlib import pyplot as plt


def evaluate(eval_env, policy=None, total_steps=1_000):
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
    Plot 2x2 grouped bar charts comparing base vs policy across terrain types.
    Top row: episode length, terminations
    Bottom row: linear vel error, yaw vel error
    """
    terrains = list(results.keys())
    x = np.arange(len(terrains))
    width = 0.3
 
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
 
    # --- Top left: Episode length ---
    ax = axes[0, 0]
    base_vals = [results[t]["base"]["episode_length_mean"] for t in terrains]
    policy_vals = [results[t]["policy"]["episode_length_mean"] for t in terrains]
    base_err = [results[t]["base"]["episode_length_std"] for t in terrains]
    policy_err = [results[t]["policy"]["episode_length_std"] for t in terrains]
 
    ax.bar(x - width / 2, base_vals, width, label="Base controller",
           color="tab:gray", yerr=base_err, capsize=4, error_kw={"linewidth": 1.2})
    ax.bar(x + width / 2, policy_vals, width, label="Residual RL policy",
           color="tab:blue", yerr=policy_err, capsize=4, error_kw={"linewidth": 1.2})
    ax.set_ylabel("Steps", fontsize=11)
    ax.set_title("Episode length", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in terrains], fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
 
    # --- Top right: Terminations ---
    ax = axes[0, 1]
    base_vals = [results[t]["base"]["total_terminations"] for t in terrains]
    policy_vals = [results[t]["policy"]["total_terminations"] for t in terrains]
 
    ax.bar(x - width / 2, base_vals, width, label="Base controller", color="tab:gray")
    ax.bar(x + width / 2, policy_vals, width, label="Residual RL policy", color="tab:blue")
    ax.set_ylabel("# of terminations", fontsize=11)
    ax.set_title("Terminations", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in terrains], fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
 
    # --- Bottom left: Linear velocity error ---
    ax = axes[1, 0]
    base_vals = [results[t]["base"]["lin_vel_error_mean"] for t in terrains]
    policy_vals = [results[t]["policy"]["lin_vel_error_mean"] for t in terrains]
    base_err = [results[t]["base"]["lin_vel_error_std"] for t in terrains]
    policy_err = [results[t]["policy"]["lin_vel_error_std"] for t in terrains]
 
    ax.bar(x - width / 2, base_vals, width, label="Base controller",
           color="tab:gray", yerr=base_err, capsize=4, error_kw={"linewidth": 1.2})
    ax.bar(x + width / 2, policy_vals, width, label="Residual RL policy",
           color="tab:blue", yerr=policy_err, capsize=4, error_kw={"linewidth": 1.2})
    ax.set_ylabel("Error (m/s)", fontsize=11)
    ax.set_title("Linear velocity error", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in terrains], fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
 
    # --- Bottom right: Yaw velocity error ---
    ax = axes[1, 1]
    base_vals = [results[t]["base"]["yaw_vel_error_mean"] for t in terrains]
    policy_vals = [results[t]["policy"]["yaw_vel_error_mean"] for t in terrains]
    base_err = [results[t]["base"]["yaw_vel_error_std"] for t in terrains]
    policy_err = [results[t]["policy"]["yaw_vel_error_std"] for t in terrains]
 
    ax.bar(x - width / 2, base_vals, width, label="Base controller",
           color="tab:gray", yerr=base_err, capsize=4, error_kw={"linewidth": 1.2})
    ax.bar(x + width / 2, policy_vals, width, label="Residual RL policy",
           color="tab:blue", yerr=policy_err, capsize=4, error_kw={"linewidth": 1.2})
    ax.set_ylabel("Error (rad/s)", fontsize=11)
    ax.set_title("Yaw velocity error", fontsize=12, fontweight="bold")
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
 

def run_terrain_comparison(terrain_types, model_path):
    eval_data = {}
    policy = PPO.load(model_path)
    for terrain in terrain_types:
        print(f"Evaluating with {terrain} terrain...")
        eval_env = TerrainAwareWalkerEnv(render_mode=None, terrain_types=[terrain])
        base_evaluation = evaluate(eval_env)
        policy_evaluation = evaluate(eval_env, policy)

        eval_data[terrain] = {}
        eval_data[terrain]["base"] = base_evaluation
        eval_data[terrain]["policy"] = policy_evaluation

    return eval_data



model_path = "./checkpoints/terrain_aware_v3_curriculum/residual_rl_5000000_steps.zip" # Best So Far

results = run_terrain_comparison(["flat", "moderate", "rough", "platforms"], model_path)
plot_terrain_comparison(results, save_path="./plots/Controller_Comparison.png")