from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback
from gymnasium.wrappers import RecordVideo
import torch

from walker_env import WalkerEnv, TerrainAwareWalkerEnv

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ── Curricula ────────────────────────────────────────────────────────────────

TERRAIN_CURRICULUM = [
    {
        "progress": 0.0,
        "terrain_types": ["flat", "moderate"],
        "terrain_weights": [0.7, 0.3],
        "cmd_duration": 4,
    },
    {
        "progress": 0.3,
        "terrain_types": ["flat", "moderate", "rough"],
        "terrain_weights": [0.3, 0.5, 0.2],
        "cmd_duration": 4,
    },
    {
        "progress": 0.6,
        "terrain_types": ["flat", "moderate", "rough"],
        "terrain_weights": [0.2, 0.3, 0.5],
        "cmd_duration": 4,
    },
]

PLATFORM_CURRICULUM = [
    {
        "progress": 0.0,
        "terrain_types": ["flat", "moderate"],
        "terrain_weights": [0.7, 0.3],
        "cmd_duration": 8,
    },
    {
        "progress": 0.3,
        "terrain_types": ["flat", "moderate", "platforms"],
        "terrain_weights": [0.2, 0.2, 0.6],
        "cmd_duration": 8,
    },
]


# ── Callbacks ────────────────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    def __init__(self, curriculum, total_timesteps, eval_env=None, verbose=1):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.total_timesteps = total_timesteps
        self.current_phase = -1
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps

        phase = 0
        for i, stage in enumerate(self.curriculum):
            if progress >= stage["progress"]:
                phase = i

        if phase != self.current_phase:
            self.current_phase = phase
            stage = self.curriculum[phase]

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"CURRICULUM: Phase {phase} at {self.num_timesteps} steps")
                print(f"  Terrains: {stage['terrain_types']}")
                print(f"  Weights:  {stage['terrain_weights']}")
                print(f"{'='*60}\n")

            self.training_env.env_method(
                "set_terrain_config",
                stage["terrain_types"],
                stage["terrain_weights"],
            )
            self.training_env.env_method(
                "set_cmd_duration",
                stage["cmd_duration"],
            )

            if self.eval_env is not None:
                self.eval_env.env_method(
                    "set_terrain_config",
                    stage["terrain_types"],
                    stage["terrain_weights"],
                )
                self.eval_env.env_method(
                    "set_cmd_duration",
                    stage["cmd_duration"],
                )

            self.logger.record("curriculum/phase", phase)

        return True


class RewardLoggingCallback(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            for key, value in info.items():
                if key.startswith("reward/"):
                    self.logger.record_mean(key, value)
        return True


# ── Core training function ───────────────────────────────────────────────────

def train(
    curriculum: list[dict],
    env_cls: type,
    run_name: str = "run",
    num_envs: int = 10,
    steps_per_env: int = 2_000_000,
):
    """
    Train a residual-RL PPO policy.

    Args:
        curriculum:    List of curriculum stage dicts (progress, terrain_types,
                       terrain_weights, cmd_duration).
        env_cls:       Gymnasium env class to instantiate (WalkerEnv or
                       TerrainAwareWalkerEnv).
        run_name:      Tag used for log / checkpoint / model-save paths.
        num_envs:      Number of parallel training environments.
        steps_per_env: Timesteps collected *per environment*.
    """
    total_timesteps = num_envs * steps_per_env
    initial_stage = curriculum[0]

    # ── Helper to build one env instance ─────────────────────────────────
    def make_env(render_mode=None):
        def _init():
            kwargs = dict(render_mode=render_mode)
            # Only pass terrain kwargs if the env class supports them
            if env_cls is TerrainAwareWalkerEnv:
                kwargs.update(
                    terrain_types=initial_stage["terrain_types"],
                    terrain_weights=initial_stage["terrain_weights"],
                )
            return env_cls(**kwargs)
        return _init

    # ── Training env ─────────────────────────────────────────────────────
    train_env = SubprocVecEnv(
        [make_env(render_mode=None) for _ in range(num_envs)],
        start_method="forkserver",
    )
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    # ── Eval env (DummyVecEnv + RecordVideo to avoid Metal crashes) ──────
    def make_eval_env():
        inner = make_env(render_mode="rgb_array")()
        return RecordVideo(
            inner,
            video_folder=f"./videos/{run_name}",
            episode_trigger=lambda ep: ep % 50 == 0,
        )

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )

    # ── Callbacks ────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./best_model/{run_name}",
        eval_freq=10_000,
        n_eval_episodes=5,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=f"./checkpoints/{run_name}",
        name_prefix="residual_rl",
    )
    curriculum_callback = CurriculumCallback(
        curriculum,
        total_timesteps=total_timesteps,
        eval_env=eval_env,
    )

    # ── PPO ──────────────────────────────────────────────────────────────
    try:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=8,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn=torch.nn.ELU,
            ),
            tensorboard_log=f"./tb_logs/{run_name}",
            verbose=1,
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=[
                checkpoint_callback,
                RewardLoggingCallback(),
                eval_callback,
                curriculum_callback,
            ],
            tb_log_name="residual_rl",
        )
    finally:
        train_env.close()
        eval_env.close()


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(
        curriculum=PLATFORM_CURRICULUM,
        env_cls=TerrainAwareWalkerEnv,
        run_name="terrainwalker_platform",
    )