import gymnasium as gym
import mujoco
import mujoco.viewer

import numpy as np
import os
import time
from obs_manager import ObservationManager, CommandGenerator, PushDisturbance
from icp_controller import WalkingController

import utils

class WalkerEnv(gym.Env):
    """
    Custom Gymnasium environment for a simple 2D walker.
    
    Observation Space (continuous):
        - Joint angles, joint velocities, body position, body velocity, etc.
    
    Action Space (continuous):
        - Torques applied to each joint
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    control_freq = 50 # Hz


    def __init__(self, render_mode="rgb_array", base_controller=True):
        xml_path = os.path.join("xml_files", "biped_3d_5dof_leg.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # self.mjx_model = mjx.put_model(mj_model)
        # self.mjx_data = mjx.put_data(mj_model, mj_data)

        self.n_joints = 10 # Hip: (Roll, Pitch, Yaw), Knee: Pitch, Ankle: Pitch

        # --- Action Space ---
        # Residual velocity for each joint
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_joints,),
            dtype=np.float32,
        )

        # --- Observation Space ---
        '''
        Base Angular Velocity (3, 5)
        Base Orientation (3, 5)
        Command Vel (3,): [x_lin, y_lin, z_ang]
        Joint Positions (10, 5)
        Joint Velocity (10,)
        Previous Action (10,)
        '''
        self.obs_hist = 5 # Observation contains previous 5 joint pos, base ang_vel, and base_orientation

        self.obs_dim = 103
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.obs_manager = ObservationManager(self.model, self.data, num_joints=self.n_joints, history_length=self.obs_hist)
        self.prev_action = None

        self.dt = self.model.opt.timestep
        self.decimation = int(1 / (self.control_freq * self.dt))

        
        self.command_manager = CommandGenerator(dt=(self.dt * self.decimation))
        self.command_manager.reset()

        self.controller = WalkingController(self.model, self.data, t0=0.0)

        self.push_disturbance = PushDisturbance(self.dt)

        

        self._step_count = 0

        episode_length = 30 # seconds
        self._max_steps_per_episode = self.control_freq * episode_length

        self.residual_scale = 1

        self.render_mode = render_mode
        self._viewer = None
        self._renderer = None

        self.prev_action = np.zeros(self.n_joints, dtype=np.float32)


    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)  

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.command_manager.reset()
        cmd_vel = self.command_manager.step()

        self.obs_manager.reset()
        self.obs_manager.warmup(self.data, cmd_vel)

        self.controller = WalkingController(self.model, self.data, t0=0.0) # Reinitialize Controller

        self.push_disturbance.reset()

        observation = self._get_obs(cmd_vel, self.prev_action)

        self._step_count = 0

        info = {}

        return observation, info

    def step(self, action):
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Get command velocity for this step
        self._cmd_vel = - self.command_manager.step()
        dx_des, dy_des, dz_omega = self._cmd_vel

        # Compute base controller output once per step
        icp_ctrl = self.controller.step(
            t=self.data.time,
            dx_des=dx_des,
            dy_des=dy_des,
            dz_omega=dz_omega,
        )

        # Add residual
        ctrl = icp_ctrl + (self.residual_scale * action)

        # Perturbation
        push = self.push_disturbance.step()    # <-- here
        if push is not None:
            self.data.qvel[0:2] += push


        # Run substeps
        for _ in range(self.decimation):
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)

        # Observation
        obs = self._get_obs(self._cmd_vel, action)
        self.prev_action = action.copy()

        # Reward
        reward, info = self._compute_reward(action)

        # Termination
        terminated = self._check_terminated()
        truncated = self._check_truncated()

        # info = {
        #     "reward" : reward,
        #     "icp_ctrl": icp_ctrl.copy(),
        #     "cmd_vel": self._cmd_vel.copy(),
        # }

        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, action):
        """
        Reward structure:
          - Track commanded velocity
          - Penalize large residual actions (encourage using the base controller)
          - Penalize excessive joint velocities
          - Alive bonus
        """
        # Velocity tracking: get torso velocity in world frame
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        torso_vel = self.data.cvel[torso_id]  # (6,) — [angular(3), linear(3)]
        lin_vel = torso_vel[3:]  # world-frame linear velocity

        dx_des, dy_des, dz_omega = self._cmd_vel

        # Forward/lateral velocity tracking
        vel_error_x = (lin_vel[0] - dx_des) ** 2
        vel_error_y = (lin_vel[1] - dy_des) ** 2
        r_velocity = np.exp(-2.0 * (vel_error_x + vel_error_y))

        # Yaw rate tracking
        ang_vel_z = torso_vel[2]
        yaw_error = (ang_vel_z - dz_omega) ** 2
        r_yaw = np.exp(-2.0 * yaw_error)

        # Action penalty: discourage large residuals
        r_action = -0.01 * np.sum(action ** 2)

        # Joint velocity smoothness penalty
        joint_vels = self.data.qvel[-self.n_joints:]
        r_smooth = -0.001 * np.sum(joint_vels ** 2)

        # Alive bonus
        r_alive = 0.5

        reward = r_velocity + 0.3 * r_yaw + r_action + r_smooth + r_alive

        info = {
            "reward/total" : reward
        }

        return reward, info
    
    def _check_truncated(self):
        if self._step_count > self._max_steps_per_episode:
            return True
        
        return False

    def _check_terminated(self):
        """Terminate if torso is too low (fallen)."""
        # Torso Height
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        torso_height = self.data.xpos[torso_id][2]

        # Torso orientation — check if upright
        torso_mat = self.data.xmat[torso_id].reshape(3, 3)
        up_vec = torso_mat[:, 2]  # z-column of rotation matrix
        upright = up_vec[2]  # dot product with world z

        # TODO: Add termination condition for excessive body velocity

        if torso_height < 0.5:
            return True
        if upright < 0.3:  # tilted more than ~70 degrees
            return True

        return False

    def _get_obs(self, cmd_vel, prev_action):
        self.obs_manager.update(cmd_vel, prev_action) # Updates History Buffers
        return self.obs_manager.get_obs()
    
    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

            # Clear previous debug geoms
            self._viewer._user_scn.ngeom = 0

            torso_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
            )
            torso_pos = self.data.xpos[torso_id]

            # Actual body velocity (green)
            body_vel = 2 * self.data.cvel[torso_id][3:]  # linear velocity
            body_vel[-1] = 0

            utils.draw_arrow(
                self._viewer,
                pos=torso_pos,
                direction=body_vel,
                radius=0.1,
                rgba=[0, 1, 0, 0.8],
            )

            # Commanded velocity (blue)
            if hasattr(self, '_cmd_vel'):
                cmd_dir = 2 * np.array([
                    self._cmd_vel[0],
                    self._cmd_vel[1],
                    0.0,
                ])
                utils.draw_arrow(
                    self._viewer,
                    pos=torso_pos,  # offset up slightly so they don't overlap
                    direction=cmd_dir,
                    radius=0.1,
                    rgba=[0, 0.5, 1, 0.8],
                )

            self._viewer.sync()
            return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        super().close()  



# env = WalkerEnv(render_mode="human")

# while True:
#     obs, r, term, trunc, info = env.step(np.zeros(10))
#     env.render()
#     time.sleep(1 / 50)  # match your 50Hz control rate
#     if term:
#         env.reset()

#     if trunc:
#         break

# print(env.data.time)