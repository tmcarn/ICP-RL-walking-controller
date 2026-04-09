from collections import deque
import numpy as np

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
            self.update(data, velocity_command, action)



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
    def __init__(self, dt, change_interval=2.0):
        self.dt = dt
        self.change_interval = change_interval
        self.timer = 0.0
        self.cmd = np.zeros(3)

    def reset(self, np_random):
        self.timer = 0.0
        self._sample(np_random)
        return self.cmd

    def step(self):
        self.timer += self.dt
        if self.timer >= self.change_interval:
            self._sample()
            self.timer = 0.0
        return self.cmd

    def _sample(self):
        self.cmd = np.array([
            np.random.uniform(-0.5, 0.5),   # x_lin (forward/backward)
            np.random.uniform(-0.2, 0.2),   # y_lin (lateral)
            np.random.uniform(-1.0, 1.0),   # z_ang (yaw rate)
        ])

    