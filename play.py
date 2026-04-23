import imageio
from stable_baselines3 import PPO
from walker_env import WalkerEnv, TerrainAwareWalkerEnv
import time

if __name__ == '__main__':
    model_path = "./checkpoints/terrain_walker_v1/residual_rl_15000000_steps.zip"
    model = PPO.load(model_path)
    eval_env = WalkerEnv(render_mode="human", terrain_types=["flat"], control_mode="joystick")
    eval_env.reset()

    obs, _ = eval_env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = eval_env.step(action)

        eval_env.render()
        
        if terminated or truncated:
            obs, _ = eval_env.reset()

        time.sleep(1 / eval_env.control_freq)

