from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import RecordVideo
import torch
from copy import deepcopy

from walker_env import WalkerEnv

import os
import datetime
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"./runs/{run_id}"
os.makedirs(run_dir, exist_ok=True)
print(f"Run directory: {run_dir}")


def make_env(render_mode):
    def _init():
        return WalkerEnv(render_mode=render_mode)
    return _init

# No rendering for training
num_envs = 10
train_env = SubprocVecEnv([make_env(render_mode=None) for _ in range(num_envs)], start_method='fork')
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
    video_folder=f"{run_dir}/videos/",
    episode_trigger=lambda ep: True,
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
    best_model_save_path=f"{run_dir}/best_model/",
    eval_freq=10_000,
    n_eval_episodes=5,
)

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path=f"{run_dir}/checkpoints/",
    name_prefix="standingv2",
)

class RewardLoggingCallback(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            for key, value in info.items():
                if key.startswith("reward/"):
                    self.logger.record_mean(key, value)
        return True


class CurriculumCallback(BaseCallback):
    """Linearly ramp push disturbance magnitude from start_mag to end_mag over training."""
    def __init__(self, train_env, start_mag=0.3, end_mag=2.0, total_steps=15_000_000, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env
        self.start_mag = start_mag
        self.end_mag = end_mag
        self.total_steps = total_steps

    def _on_rollout_end(self) -> None:
        progress = min(self.num_timesteps / self.total_steps, 1.0)
        mag = self.start_mag + progress * (self.end_mag - self.start_mag)
        self.train_env.env_method("set_push_magnitude", mag)
        self.logger.record("curriculum/push_magnitude", mag)

    def _on_step(self) -> bool:
        return True


class SyncNormCallback(BaseCallback):
    """Copy obs running stats from train env to eval env before each evaluation."""
    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize, verbose: int = 0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        self.eval_env.obs_rms = deepcopy(self.train_env.obs_rms)
        return True


class SaveNormCallback(BaseCallback):
    """Save VecNormalize stats to disk alongside each checkpoint."""
    def __init__(self, train_env: VecNormalize, save_path: str, save_freq: int, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env
        self.save_path = save_path
        self.save_freq = save_freq
        self._last_save = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self.save_freq:
            path = os.path.join(self.save_path, f"vecnormalize_{self.num_timesteps}_steps.pkl")
            self.train_env.save(path)
            self._last_save = self.num_timesteps
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
        clip_range=linear_schedule(0.1, 0.02),
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.ELU,
        ),
        tensorboard_log=f"{run_dir}/tb_logs/",
        verbose=1,
    )
    
    # Start push disturbance at minimum magnitude
    train_env.env_method("set_push_magnitude", 0.3)

    save_norm_callback = SaveNormCallback(train_env, save_path=f"{run_dir}/checkpoints/", save_freq=100_000)

    steps_per_env = 1_500_000
    model.learn(total_timesteps=steps_per_env * num_envs,
                callback=[checkpoint_callback, RewardLoggingCallback(), SyncNormCallback(train_env, eval_env), CurriculumCallback(train_env), eval_callback, save_norm_callback],
                tb_log_name="initial_trials")
finally:
    train_env.save(f"{run_dir}/vecnormalize_final.pkl")
    train_env.close()
    eval_env.close()