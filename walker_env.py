import gymnasium as gym
import mujoco
import mujoco.viewer

import numpy as np
import os
import time
from obs_manager import ObservationManager, CommandGenerator, PushDisturbance
from icp_controller import WalkingController
from terrain import generate_terrain_hfield, generate_platforms

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


    def __init__(self, render_mode="rgb_array", base_controller_only=False):
        xml_path = os.path.join("xml_files", "biped_3d_5dof_leg.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.hfield_data[:] = 0.5 # Crashes otherwise

        self.data = mujoco.MjData(self.model)

        # Generate terrain heightfield
        # generate_terrain_hfield(self.model)
        generate_platforms(self.model)
        nrow = self.model.hfield_nrow[0]
        ncol = self.model.hfield_ncol[0]
        origin_height = self.model.hfield_data[(nrow // 2) * ncol + (ncol // 2)]
        self.data.qpos[2] = origin_height + 1.4  # Set initial height above terrain

        self.base_controller_only = base_controller_only

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
        Current Base Action (10,)
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
        self._new_terrain_rendered = False

        self.prev_action = np.zeros(self.n_joints, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)  

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)
        self.model.hfield_data[:] = 0.5 # Crashes otherwise

        mujoco.mj_forward(self.model, self.data)

        # Reset Terrain
        # generate_terrain_hfield(self.model)
        generate_platforms(self.model)
        self._new_terrain_rendered = False

        nrow = self.model.hfield_nrow[0]
        ncol = self.model.hfield_ncol[0]
        origin_height = self.model.hfield_data[(nrow // 2) * ncol + (ncol // 2)]
        self.data.qpos[2] = origin_height + 1.4  # Set initial height above terrain

        self.command_manager.reset()
        cmd_vel = self.command_manager.step()

        self.prev_action = np.zeros_like(self.prev_action)
        self.obs_manager.reset()
        self.obs_manager.warmup(self.data, cmd_vel)

        self.controller = WalkingController(self.model, self.data, t0=0.0) # Reinitialize Controller

        self.push_disturbance.reset()

        observation = self._get_obs(cmd_vel, self.prev_action)

        self._step_count = 0

        info = {}

        return observation, info

    def step(self, action):
        "Stepping"
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # For Baseline with No RL
        if self.base_controller_only:
            action = np.zeros_like(action)


        # Get command velocity for this step
        self._cmd_vel = self.command_manager.step()
        dx_des, dy_des, dz_omega = self._cmd_vel

        # Compute base controller output once per step
        base_action = self.controller.step(
            t=self.data.time,
            dx_des=dx_des, # Negated in original repo for some reason
            dy_des=dy_des, # Negated in original repo for some reason
            dz_omega=dz_omega,
        )

        # Add residual
        ctrl = base_action + (self.residual_scale * action)

        # Perturbation
        push = self.push_disturbance.step()
        if push is not None:
            self.data.qvel[0:2] += push

        # Run substeps
        for _ in range(self.decimation):
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)

        # Observation
        obs = self._get_obs(self._cmd_vel, action)

        # Reward
        reward, info = self._compute_reward(action)

        # Termination
        termination_penalty = 0
        terminated = self._check_terminated() or self._check_sim_unstable()
        if terminated:
            termination_penalty = -100
            reward += termination_penalty
        
        truncated = self._check_truncated()

        info["reward/term_penalty"] = termination_penalty
        info["reward/total_reward"] = reward

        self.prev_action = action.copy()

        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, action):
        """
        Reward structure:
          - Track commanded velocity
          - Penalize large residual actions (encourage using the base controller)
          - Penalize excessive joint velocities
          - Alive bonus
        """
        # Velocity tracking: get torso velocity in Body Frame
        lin_vel_body, ang_vel_body = self._get_body_vel()

        dx_des, dy_des, dz_omega = self._cmd_vel

        # Forward/lateral velocity tracking
        vel_error_x = (lin_vel_body[0] - dx_des) ** 2
        vel_error_y = (lin_vel_body[1] - dy_des) ** 2
        r_velocity = np.exp(-5.0 * (vel_error_x + vel_error_y))

        total_lin_vel_error = np.sqrt(vel_error_x + vel_error_y)

        # Yaw rate tracking
        yaw_vel_body = ang_vel_body[2]
        yaw_error = (yaw_vel_body - dz_omega) ** 2
        r_yaw = np.exp(-2.0 * yaw_error)

        total_yaw_vel_error = np.sqrt(yaw_error)

        # Orientation
        up_vector = np.zeros(3)
        mujoco.mju_rotVecQuat(up_vector, np.array([0.0, 0.0, 1.0]), self.data.qpos[3:7])
        r_orientation = np.exp(-2.0 * (1.0 - up_vector[2]))

        # Residual Normalization
        r_residual = np.exp(-0.5 * np.linalg.norm(action)**2)
        
        # Action Smoothing Reward
        r_action_smooth = np.exp(-1 * np.linalg.norm(action - self.prev_action))

        # Alive bonus
        r_alive = 0.5

        reward = r_velocity + 0.5 * r_orientation + 0.3 * r_yaw + r_action_smooth  + 0.1 * r_residual + r_alive

        info = {
            "reward/total_reward": reward,
            "reward/total_lin_vel_error": total_lin_vel_error,
            "reward/total_yaw_vel_error": total_yaw_vel_error,
            "reward/orientation": up_vector[2],
            "reward/residual_norm": np.linalg.norm(action),
            "reward/action_smoothness": np.linalg.norm(action - self.prev_action),
        }

        return reward, info
    
    def _get_body_vel(self):
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        R_t = self.data.xmat[torso_id].reshape(3, 3)

        v_6d = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, torso_id, v_6d, 0)

        lin_vel_body = R_t.T @ v_6d[3:]
        ang_vel_body = R_t.T @ v_6d[:3]

        return lin_vel_body, ang_vel_body

    
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

        if torso_height < 0.5:
            return True
        if upright < 0.3:  # tilted more than ~70 degrees
            return True

        return False
    
    def _check_sim_unstable(self):
        if np.any(np.isnan(self.data.qacc)) or np.any(np.isinf(self.data.qacc)):
            return True
        if np.any(np.abs(self.data.qvel) > 100):
            return True
        return False

    def _get_obs(self, cmd_vel, prev_action):
        self.obs_manager.update(cmd_vel, prev_action) # Updates History Buffers
        return self.obs_manager.get_obs()
    
    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

            if not self._new_terrain_rendered:
                self._viewer.update_hfield(0)
                self._new_terrain_rendered = True

            self._viewer._user_scn.ngeom = 0
            self._draw_debug_arrows(self._viewer._user_scn)
            self._viewer.sync()
            return None

        elif self.render_mode == "rgb_array":
            if not self._new_terrain_rendered:
                # Recreate renderer to pick up new heightfield
                if self._renderer is not None:
                    self._renderer.close()
                self._renderer = None
                self._new_terrain_rendered = True

            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)

            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
            self._renderer.update_scene(self.data, camera=cam_id)
            return self._renderer.render()


    def _draw_debug_arrows(self, scene):
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        torso_pos = self.data.xpos[torso_id]

        lin_vel_body, ang_vel_body = self._get_body_vel()
        lin_vel_body[2] = 0

        R_t = self.data.xmat[torso_id].reshape(3, 3)

        # Actual velocity (green)
        lin_vel_world = R_t @ lin_vel_body
        utils.draw_arrow(
            scene,
            pos=torso_pos,
            direction=lin_vel_world,
            radius=0.08,
            rgba=[0, 1, 0, 0.8],
        )

        # Commanded velocity (blue)
        
        cmd_body = np.array([self._cmd_vel[0], self._cmd_vel[1], 0.0])
        cmd_world = R_t @ cmd_body
        utils.draw_arrow(
            scene,
            pos=torso_pos,
            direction=4 * cmd_world,
            radius=0.08,
            rgba=[0, 0.5, 1, 0.8],
        )

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        super().close()  


if __name__ == '__main__':
    env = WalkerEnv(render_mode="human")

    try:
        while True:
            obs, r, term, trunc, info = env.step(np.zeros(10))
            env.render()
            time.sleep(1 / 50)  # match your 50Hz control rate
            if term:
                env.reset()

            if trunc:
                break

    finally:
        env.close()

