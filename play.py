import imageio
from stable_baselines3 import PPO
from walker_env import WalkerEnv

# def evaluate(model, eval_env, duration=5):
#     frames = []
#     rewards = []

#     timestep_count = int(duration / eval_env.env.control_timestep())

#     obs, _ = eval_env.reset()
#     for _ in range(timestep_count):
#         action, _ = model.predict(obs)
#         obs, reward, terminated, truncated, info = eval_env.step(action)

#         frame = eval_env.env.physics.render(height=480, width=640, camera_id=0)
#         frames.append(frame)
#         rewards.append(reward)

#         if terminated or truncated:
#             obs, _ = eval_env.reset()

#     return frames, rewards

# def save_video(frames, filename="eval.mp4", fps=40):
#     imageio.mimsave(filename, frames, fps=fps)
#     print(f"Saved to {filename}")

model_path = "./checkpoints/standingv2_15000000_steps.zip"
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

