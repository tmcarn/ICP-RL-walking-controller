from collections import deque
import numpy as np
import mujoco

class ObservationManager():

    def __init__(self, model, data, num_joints=10, history_length=5):
        self.model = model
        self.data = data

        self.num_joints = num_joints
        self.history_length = history_length

        # Ring buffers for terms that need history
        self.base_ang_vel_buf = deque(
            [np.zeros(3) for _ in range(history_length)], maxlen=history_length
        )
        self.base_orientation_buf = deque(
            [np.zeros(3) for _ in range(history_length)], maxlen=history_length
        )
        self.joint_pos_buf = deque(
            [np.zeros(num_joints) for _ in range(history_length)], maxlen=history_length
        )

        # Previous action (no history)
        self.prev_action = np.zeros(num_joints)

    def reset(self):
        """Clear all history buffers."""
        for buf in [self.base_ang_vel_buf, self.base_orientation_buf, self.joint_pos_buf]:
            for i in range(len(buf)):
                buf[i] = np.zeros_like(buf[i])
        self.prev_action = np.zeros(self.num_joints)

    def update(self, velocity_command, prev_action):
        """
        Push new readings into ring buffers and store latest action.
        Call this once per timestep BEFORE get_obs().
        """
        # Base angular velocity from gyroscope
        base_ang_vel = self.data.sensordata[0:3].copy()
        self.base_ang_vel_buf.append(base_ang_vel)

        # Base orientation as projected gravity
        # MuJoCo quat is [w, x, y, z] in qpos[3:7] for a free joint
        quat = self.data.qpos[3:7]
        self.base_orientation_buf.append(self._projected_gravity(quat))

        # Joint positions (relative to default)
        joint_pos = self.data.qpos[7:7 + self.num_joints].copy()
        self.joint_pos_buf.append(joint_pos)

        # Joint velocity (current only, no history)
        self.joint_vel = self.data.qvel[6:6 + self.num_joints].copy()

        # Store command and previous action
        self.velocity_command = velocity_command.copy()
        self.prev_action = prev_action.copy()

    def get_obs(self):
        """
        Return flattened observation vector (103,).
        
        History terms are stacked as (dim, history_length) then flattened,
        with oldest first (index 0 = oldest, index 4 = newest).
        """
        # (3, 5) -> flatten to (15,)
        base_ang_vel = np.stack(list(self.base_ang_vel_buf), axis=-1).flatten()

        # (3, 5) -> flatten to (15,)
        base_orientation = np.stack(list(self.base_orientation_buf), axis=-1).flatten()

        # (3,)
        vel_cmd = self.velocity_command

        # (10, 5) -> flatten to (50,)
        joint_pos = np.stack(list(self.joint_pos_buf), axis=-1).flatten()

        # (10,)
        joint_vel = self.joint_vel

        # (10,)
        prev_action = self.prev_action

        obs = np.concatenate([
            base_ang_vel,       # 15
            base_orientation,   # 15
            vel_cmd,            #  3
            joint_pos,          # 50
            joint_vel,          # 10
            prev_action,        # 10
        ]).astype(np.float32)

        return obs  # (103,)
    
    def warmup(self, data, velocity_command):
        """
        Fill history buffers by repeating the current state.
        Used for resets when there the buffers are empty.
        """
        action = np.zeros(self.num_joints)
        for _ in range(self.history_length):
            self.update(velocity_command, action)

    @staticmethod
    def _projected_gravity(quat):
        """
        Project gravity vector [0, 0, -1] into the body frame
        using the base quaternion. Returns (3,) vector.
        """
        w, x, y, z = quat
        # Rotate world gravity [0, 0, -1] by inverse of body quaternion
        gx = -2.0 * (x * z - w * y)
        gy = -2.0 * (y * z + w * x)
        gz = -(w * w - x * x - y * y + z * z)
        return np.array([gx, gy, gz])
    
class CommandGenerator:
    def __init__(self, dt, change_interval=4): # Changes every 4 seconds
        self.dt = dt
        self.change_interval = change_interval
        self.timer = 0.0
        self.cmd = np.zeros(3)

    def reset(self):
        self.timer = 0.0
        self._sample()
        return self.cmd

    def step(self):
        self.timer += self.dt        
        if self.timer >= self.change_interval:
            self._sample()
            self.timer = 0.0
        return self.cmd

    def _sample(self):
        self.cmd = np.array([
            -np.random.uniform(-0.5, 0.5),   # x_lin (forward/backward) (originally 0.5)
            -np.random.uniform(-0.3, 0.3),   # y_lin (lateral) (originally 0.3)
            np.random.uniform(-1.0, 1.0),   # z_ang (yaw rate)
        ])


class PushDisturbance:
    def __init__(self, dt, push_interval=4.0, velocity_range=(0.5, 2.0)):
        self.dt = dt
        self.push_interval = push_interval
        self.velocity_range = velocity_range
        self.timer = 0.0

    def reset(self):
        self.timer = 0.0

    def step(self):
        self.timer += self.dt

        if self.timer >= self.push_interval:
            self.timer = 0.0
            return self._sample()

        return None

    def _sample(self):
        return np.array([
            np.random.uniform(-self.velocity_range[1], self.velocity_range[1]),
            np.random.uniform(-self.velocity_range[1], self.velocity_range[1]),
        ])

class HeightScanner:
    def __init__(self, model, data, extent, grid_size):
        self.model = model
        self.data = data
        self.extent = extent
        self.grid_size = grid_size

    def sample_heightfield(self):
        # Heightfield info
        hfield_id = 0
        nrow = self.model.hfield_nrow[hfield_id]
        ncol = self.model.hfield_ncol[hfield_id]
        x_half = self.model.hfield_size[hfield_id][0]
        y_half = self.model.hfield_size[hfield_id][1]
        z_max = self.model.hfield_size[hfield_id][2]
        z_min = self.model.hfield_size[hfield_id][3]

        # Robot position and yaw
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        robot_x = self.data.xpos[torso_id][0]
        robot_y = self.data.xpos[torso_id][1]
        robot_z = self.data.xpos[torso_id][2]
    
        # Extract yaw from quaternion
        quat = self.data.qpos[3:7]
        # yaw = atan2(2(wz + xy), 1 - 2(yy + zz))
        w, qx, qy, qz = quat
        yaw = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
    
        # Local grid points from -extent to +extent
        spacing = np.linspace(-self.extent, self.extent, self.grid_size)
        local_x, local_y = np.meshgrid(spacing, spacing)
        local_x = local_x.ravel()
        local_y = local_y.ravel()
    
        # Rotate to world frame
        world_x = robot_x + cos_yaw * local_x - sin_yaw * local_y
        world_y = robot_y + sin_yaw * local_x + cos_yaw * local_y
    
        # Convert world coordinates to heightfield indices
        col = ((world_x + x_half) / (2 * x_half) * ncol).astype(int)
        row = ((world_y + y_half) / (2 * y_half) * nrow).astype(int)
    
        # Clamp to valid range
        col = np.clip(col, 0, ncol - 1)
        row = np.clip(row, 0, nrow - 1)
    
        # Sample heights
        raw_heights = self.model.hfield_data[row * ncol + col]
    
        # Convert to world heights: z_min + raw * (z_max - z_min)
        world_heights = z_min + raw_heights * (z_max - z_min)
    
        # Make relative to robot height
        relative_heights = world_heights - robot_z

        relative_heights = np.reshape(relative_heights, (1, self.grid_size, self.grid_size))

        # Build world points for visualization
        world_points = np.stack([world_x, world_y, world_heights], axis=1)

        return relative_heights.astype(np.float32), world_points
    

    def draw_heightfield_samples(self, scene, world_points, robot_z, radius=0.03):
        """
        Draw sample points as colored spheres in the MuJoCo viewer.
        
        Color indicates height relative to robot:
            - Green: at robot height
            - Red: below robot
            - Blue: above robot
        
        Args:
            scene: MuJoCo scene (viewer._user_scn)
            world_points: (N, 3) array of world positions from sample_heightfield
            robot_z: robot's current z position
            radius: sphere radius
        """
        for i in range(len(world_points)):
            if scene.ngeom >= scene.maxgeom:
                break
    
            pos = world_points[i]
            rel_height = pos[2] - robot_z
    
            # Color by relative height
            if rel_height > 0.05:
                # Above robot — blue
                t = min(rel_height / 0.5, 1.0)
                rgba = [0.0, 0.0, 0.3 + 0.7 * t, 0.6]
            elif rel_height < -0.05:
                # Below robot — red
                t = min(abs(rel_height) / 0.5, 1.0)
                rgba = [0.3 + 0.7 * t, 0.0, 0.0, 0.6]
            else:
                # Near robot height — green
                rgba = [0.0, 0.8, 0.0, 0.6]
    
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[radius, 0, 0],
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=np.array(rgba, dtype=np.float32),
            )
            scene.ngeom += 1
