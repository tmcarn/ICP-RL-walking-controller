from tbparse import SummaryReader
import matplotlib.pyplot as plt
import numpy as np

# ── Config ──────────────────────────────────────────────────────
TRIAL_PATHS = {
    "Residual RL policy": "./tb_logs/terrain_walker/redidual_rl_2",
    "Terrain-Aware policy": "./tb_logs/terrainwalker_terrain_non_random/residual_rl_1"
    # "Platform Environment": "./tb_logs/terrainwalker_platform_v2/residual_rl_1",
}

TAGS = {
    "ep_len":        "rollout/ep_len_mean",
    "ep_rew":        "rollout/ep_rew_mean",
    "lin_vel_error": "reward/total_lin_vel_error",
    "yaw_vel_error": "reward/total_yaw_vel_error",
}

PLOT_CONFIGS = [
    ("ep_len",        "Episode Length",        "Steps"),
    ("ep_rew",        "Mean Episode Reward",   "Reward"),
    ("lin_vel_error", "Linear Velocity Error", "Error"),
    ("yaw_vel_error", "Yaw Velocity Error",    "Error (rad)"),
]

Y_LIMITS = {
    # "ep_len":        (0, 1000),
    # "ep_rew":        (-100, 300),
    "lin_vel_error": (0, 1.5),
    "yaw_vel_error": (0, 5),
}

# ── Load all trials ─────────────────────────────────────────────
trials = {}
for name, path in TRIAL_PATHS.items():
    reader = SummaryReader(path)
    trials[name] = reader.scalars

# ── Plot (2x2) ─────────────────────────────────────────────────
colors = ["tab:blue","tab:green"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, (key, title, ylabel) in zip(axes, PLOT_CONFIGS):
    tag = TAGS[key]
    for (name, df), color in zip(trials.items(), colors):
        if tag not in df["tag"].values:
            print(f"Warning: '{tag}' not found in {name}")
            continue
        subset = df[df["tag"] == tag].sort_values("step")
        ax.plot(subset["step"].values, subset["value"].values,
                color=color, linewidth=1, label=name)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlabel("Timesteps")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if key in Y_LIMITS:
        ax.set_ylim(Y_LIMITS[key])

fig.suptitle("Training Metrics — Multi-Trial Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("training_metrics_comparison.png", dpi=150)
plt.show()

print("Saved to training_metrics_comparison.png")