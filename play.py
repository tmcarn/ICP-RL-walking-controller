import imageio
from stable_baselines3 import PPO
from walker_env import WalkerEnv


model_path = "./checkpoints/v2/residual_rl_7000000_steps.zip"
model = PPO.load(model_path)
eval_env = WalkerEnv(render_mode="human")
eval_env.reset()

duration = 30

timestep_count = int(duration / eval_env.dt)

obs, _ = eval_env.reset()
for _ in range(timestep_count):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = eval_env.step(action)

    eval_env.render()
    
    if terminated or truncated:
        obs, _ = eval_env.reset()

    import time
    time.sleep(1 / eval_env.control_freq)

