import gymnasium as gym
import mujoco
import numpy as np
import os
from obs_manager import ObservationManager

class WalkerEnv(gym.Env):
    """
    Custom Gymnasium environment for a simple 2D walker.
    
    Observation Space (continuous):
        - Joint angles, joint velocities, body position, body velocity, etc.
    
    Action Space (continuous):
        - Torques applied to each joint
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        xml_path = os.path.join("xml_files", "biped_3d_5dof_leg.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.n_joints = 10 # Hip: (Roll, Pitch, Yaw), Knee: Pitch, Ankle: Pitch

        # --- Action Space ---
        # Residual velocity for each joint
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_joints,),
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


    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)  

        # Initialize state (e.g., standing upright with small random perturbation)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(self.obs_dim,))

        cmd_vel = None

        observation = self._get_obs(cmd_vel, self.prev_action)

        info = {}

        return observation, info
        pass

    def step():
        pass

    def render():
        pass

    def _get_obs(self, cmd_vel, prev_action):
        self.obs_manager.update(cmd_vel, prev_action) # Updates History Buffers
        return self.obs_manager.get_obs()

