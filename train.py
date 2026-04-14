from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import RecordVideo
import torch

from walker_env import WalkerEnv

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def make_env(render_mode):
    def _init():
        return WalkerEnv(render_mode=render_mode)
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
        WalkerEnv(render_mode='rgb_array'),
        video_folder="./videos/",
        episode_trigger=lambda ep: ep % 2 == 0,
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
        best_model_save_path="./best_model/",
        eval_freq=10_000,
        n_eval_episodes=5,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints/randomized_terrain_v2",
        name_prefix="residual_rl",
    )

    class RewardLoggingCallback(BaseCallback):
        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                for key, value in info.items():
                    if key.startswith("reward/"):
                        self.logger.record_mean(key, value)
            return True



    def linear_schedule(initial: float, final: float = 0.0):
        def func(progress_remaining: float) -> float:
            return final + progress_remaining * (initial - final)
        return func

    try:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=linear_schedule(1e-4),
            n_steps=2048,
            batch_size=256,
            n_epochs=8,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=linear_schedule(0.2, 0.05),
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn=torch.nn.ELU,
            ),
            tensorboard_log="./tb_logs/terrain_walker",
            verbose=1,
        )
        
        steps_per_env = 1_500_000
        model.learn(total_timesteps=steps_per_env * num_envs,
                    callback=[checkpoint_callback, RewardLoggingCallback(), eval_callback],
                    tb_log_name="residual_rl")
    finally:
        train_env.close()
        eval_env.close()