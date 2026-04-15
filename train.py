from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import RecordVideo
import torch

from walker_env import WalkerEnv, TerrainAwareWalkerEnv

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CURRICULUM = [
    {
        "progress": 0.0,
        "terrain_types": ["flat", "moderate"],
        "terrain_weights": [0.7, 0.3],
        "cmd_duration": 4
    },
    {
        "progress": 0.2,
        "terrain_types": ["flat", "moderate", "rough"],
        "terrain_weights": [0.4, 0.4, 0.2],
        "cmd_duration": 4
    },
    {
        "progress": 0.4,
        "terrain_types": ["flat", "moderate", "rough", "platforms"],
        "terrain_weights": [0.1, 0.2, 0.4, 0.3],
        "cmd_duration": 6
    },
    {
        "progress": 0.6,
        "terrain_types": ["flat", "moderate", "rough", "platforms"],
        "terrain_weights": [0.1, 0.1, 0.3, 0.5],
        "cmd_duration": 8
    },
]

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
                stage["cmd_duration"]
            )

            # Update eval env too
            if self.eval_env is not None:
                
                self.eval_env.env_method(
                    "set_terrain_config",
                    stage["terrain_types"],
                    stage["terrain_weights"],
                )

                self.training_env.env_method(
                    "set_cmd_duration",
                    stage["cmd_duration"]
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


def make_env(render_mode):
    def _init():
        return TerrainAwareWalkerEnv(
            render_mode=render_mode,
            terrain_types=CURRICULUM[0]["terrain_types"],
            terrain_weights=CURRICULUM[0]["terrain_weights"],
        )
    return _init

if __name__ == "__main__":
    # No rendering for training
    num_envs = 10
    train_env = SubprocVecEnv([make_env(render_mode=None) for _ in range(num_envs)], start_method='forkserver')
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    eval_env = DummyVecEnv([lambda: RecordVideo(
        TerrainAwareWalkerEnv(
            render_mode='rgb_array',
            terrain_types=CURRICULUM[0]["terrain_types"],
            terrain_weights=CURRICULUM[0]["terrain_weights"],
        ),
        video_folder="./videos/",
        episode_trigger=lambda ep: ep % 50 == 0,
    )])

    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/terrain_aware_v3_curriculum",
        eval_freq=10_000,
        n_eval_episodes=5,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints/terrain_aware_v3_curriculum",
        name_prefix="residual_rl",
    )

    def linear_schedule(initial: float, final: float = 0.0):
        def func(progress_remaining: float) -> float:
            return final + progress_remaining * (initial - final)
        return func
    
    steps_per_env = 4_000_000
    total_timesteps = num_envs * steps_per_env

    curriculum_callback = CurriculumCallback(CURRICULUM, total_timesteps=total_timesteps, eval_env=eval_env)

    try:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=1e-4, #linear_schedule(1e-4),
            n_steps=2048,
            batch_size=256,
            n_epochs=8,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2, #linear_schedule(0.2, 0.05),
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn=torch.nn.ELU,
            ),
            tensorboard_log="./tb_logs/terrain_aware_v3_curriculum",
            verbose=1,
        )
        
        model.learn(total_timesteps=total_timesteps,
                    callback=[checkpoint_callback, RewardLoggingCallback(), eval_callback, curriculum_callback],
                    tb_log_name="residual_rl")
    finally:
        train_env.close()
        eval_env.close()