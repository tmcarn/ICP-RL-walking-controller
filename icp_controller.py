import numpy as np
import copy
import mujoco
from dataclasses import dataclass, field
from get_jacobian_3d_5dof_leg import get_pos_3d_jacobians
import utils


@dataclass
class WalkingConfig:
    """All tunable parameters in one place."""
    # Capture point gain
    K_xi: float = -1.0
    # Height regulation gain
    K_pz: float = 5.0
    # Pitch/roll regulation gain
    K_pt: float = -15.0
    # Lateral rocking offset for weight shift
    dy_rocking: float = 0.1
    # Desired step period
    T_des: float = 0.4
    # Minimum time before allowing leg switch
    min_step_time: float = 0.2
    # Target CoM height
    z_target: float = 1.45
    # Swing foot lift height
    h_target: float = 0.2
    # Swing leg proportional gains
    swing_gain_xy: float = 10.0
    swing_gain_y_scale: float = 0.8
    swing_vel_limit: float = 10.0
    # Joint velocity clamp
    ctrl_clamp: float = 5.0
    # Gravity
    g: float = 9.81
    # Foot placement clamp limits
    p_min: float = -0.6
    p_max: float = 0.6
    py_min: float = 0.15
    py_max: float = 0.6
    # Ground geom names for contact detection
    ground_geoms: list = field(default_factory=lambda: ["ground", "obstacle_box", "obstacle_box_2"])
    # Foot body names
    foot_bodies: dict = field(default_factory=lambda: {"Right": "right_shin", "Left": "left_shin"})
    # Foot geom names for contact
    foot_geoms: dict = field(default_factory=lambda: {
        "Left": "left_foot_geom",
        "Right": "right_foot_geom"
    })
    # Joint names per leg (order: hip_z, hip_y, hip, knee, foot)
    joint_names: dict = field(default_factory=lambda: {
        "Left":  ["left_hip_z_j", "left_hip_y_j", "left_hip", "left_knee", "left_foot_j"],
        "Right": ["right_hip_z_j", "right_hip_y_j", "right_hip", "right_knee", "right_foot_j"],
    })
    # Control indices per leg (order: hip_z, hip_y, hip, knee, foot)
    ctrl_indices: dict = field(default_factory=lambda: {
        "Right": {"hip_z": 0, "hip_y": 1, "hip": 2, "knee": 3, "foot": 4},
        "Left":  {"hip_z": 5, "hip_y": 6, "hip": 7, "knee": 8, "foot": 9},
    })


class WalkingController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, t0: float = 0.0):
        self.model = model
        self.data = data
        self.cfg = WalkingConfig()

        # State: which leg is stance vs swing
        self.stance_side = "Right"
        self.swing_side = "Left"

        # Swing trajectory initial conditions (captured at leg switch)
        self._switch_t0 = t0
        self._swing_foot_at_switch = np.zeros(3)
        self._com_at_switch = np.zeros(3)

        # Leg switch gating
        self._last_switch_time = 0.0
        self._contact_lifted = False

        # Target orientation (identity = upright)
        self.R_target = np.eye(3)

        # Initialize sensor state
        self._torso_pos = np.zeros(3)
        self._torso_vel = np.zeros(3)
        self._torso_acc = np.zeros(3)
        self._torso_R = np.eye(3)
        self._swing_pos = np.zeros(3)
        self._q = np.zeros(10)
        self._contact = {"Left": False, "Right": False}

        # Run first sensor read and capture switch properties
        self._read_sensors()
        self._swing_foot_at_switch = self._swing_pos.copy()
        self._com_at_switch = self._torso_pos.copy()

    # -------------------------------------------------------------------------
    # Sensor reading — all MuJoCo queries happen here
    # -------------------------------------------------------------------------
    def _read_sensors(self):
        """Read all needed state from MuJoCo. Only place that touches model/data."""
        model, data, cfg = self.model, self.data, self.cfg

        # Joint angles: left leg then right leg
        self._q = np.array([
            self._get_joint_angle(name)
            for side in ["Left", "Right"]
            for name in cfg.joint_names[side]
        ])

        # Contact detection
        contact = utils.geoms_contacting_geoms(
            model, data,
            list(cfg.foot_geoms.values()) + ["left_shin_geom", "right_shin_geom"],
            cfg.ground_geoms
        )
        self._contact = {
            "Left": contact.get(cfg.foot_geoms["Left"], False),
            "Right": contact.get(cfg.foot_geoms["Right"], False),
        }

        # Stance foot frame
        p_stance_w, R_stance_w = utils.capsule_end_frame_world(
            model, data, cfg.foot_bodies[self.stance_side], torso_name="torso"
        )

        # Swing foot position in stance frame
        p_swing_w, _ = utils.capsule_end_frame_world(
            model, data, cfg.foot_bodies[self.swing_side], torso_name="torso"
        )
        self._swing_pos = utils.world_p_to_frame(
            p_world=p_swing_w, p_frame=p_stance_w, R_frame=R_stance_w
        )

        # Torso state in stance frame
        torso = utils.torso_state_in_stance_frame(
            model, data, p_c=p_stance_w, R_c=R_stance_w, torso_name="torso"
        )
        torso["position"][0] -= 0.05  # offset from original code
        self._torso_pos = torso["position"]
        self._torso_vel = torso["velocity"]
        self._torso_acc = torso["acceleration"]
        self._torso_R = torso["orientation"]

    def _get_joint_angle(self, joint_name: str) -> float:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return self.data.qpos[self.model.jnt_qposadr[jid]]

    # -------------------------------------------------------------------------
    # Leg switching
    # -------------------------------------------------------------------------
    def _try_switch(self, t: float) -> bool:
        """Check if swing foot has landed and enough time has passed. If so, swap legs."""
        swing_contact = self._contact[self.swing_side]

        if not swing_contact:
            self._contact_lifted = True

        min_time_passed = (t > self._last_switch_time + self.cfg.min_step_time)

        if swing_contact and min_time_passed:
            self._contact_lifted = False
            # Swap
            self.stance_side, self.swing_side = self.swing_side, self.stance_side
            self._last_switch_time = t
            return True
        return False

    def _capture_switch_state(self, t: float):
        """Save initial conditions for swing trajectory after a leg switch."""
        self._switch_t0 = t
        self._swing_foot_at_switch = self._swing_pos.copy()
        self._com_at_switch = self._torso_pos.copy()

    # -------------------------------------------------------------------------
    # Foot placement (capture point / LIPM)
    # -------------------------------------------------------------------------
    def _compute_capture_point(self, x: float, dx: float, ddx: float, z: float, dx_des: float) -> float:
        """
        Compute target foot placement along one axis using the 
        Linear Inverted Pendulum capture point.

        x, dx, ddx: CoM position, velocity, acceleration in stance frame
        z: CoM height above stance foot
        dx_des: desired velocity along this axis
        Returns: target foot placement p
        """
        cfg = self.cfg
        z = np.clip(z, 0.1, 1000)
        omega = np.sqrt(cfg.g / z)

        xi = x + dx / omega           # capture point
        dxi = dx + ddx / omega         # capture point velocity
        xi_des = x + dx_des / omega    # desired capture point

        p = xi - dxi / omega - cfg.K_xi * (xi - xi_des)
        return p

    def _compute_foot_placement(self, dx_des: float, dy_des: float):
        """
        Compute (p_x, p_y, e_step_down) for swing foot target.
        Applies rocking offset, clipping, and step-down urgency.
        """
        cfg = self.cfg
        tp = self._torso_pos

        p_x = self._compute_capture_point(tp[0], self._torso_vel[0], self._torso_acc[0], tp[2], dx_des)

        # Lateral rocking: shift desired landing toward stance foot side
        dy_rock = -cfg.dy_rocking if self.stance_side == "Right" else cfg.dy_rocking
        p_y = self._compute_capture_point(tp[1], self._torso_vel[1], self._torso_acc[1], tp[2], dy_des + dy_rock)

        # Step-down urgency: if foot placement is out of comfortable range, speed up the step
        e_step_down = 0.0
        if self.stance_side == "Right":
            if p_y < 0.0:
                e_step_down += -2 * p_y
            elif p_y > 0.5:
                e_step_down += 2 * (p_y - 0.5)
            p_y = np.clip(p_y, cfg.py_min, cfg.py_max)
        else:
            if p_y > 0.0:
                e_step_down += 2 * p_y
            elif p_y < -0.5:
                e_step_down += -2 * (p_y + 0.5)
            p_y = np.clip(p_y, -cfg.py_max, -cfg.py_min)

        if abs(p_x) > 0.5:
            e_step_down += 2 * (abs(p_x) - 0.5)
        p_x = np.clip(p_x, cfg.p_min, cfg.p_max)

        return p_x, p_y, e_step_down

    # -------------------------------------------------------------------------
    # Swing foot height trajectory
    # -------------------------------------------------------------------------
    def _compute_swing_height(self, t: float, e_step_down: float):
        """
        Sinusoidal swing foot trajectory + CoM height target.
        Returns (z_swing, z_com_target).
        """
        cfg = self.cfg
        h_0 = self._swing_foot_at_switch[2]
        delta_t = t - self._switch_t0
        tau = delta_t / cfg.T_des + e_step_down

        # Sinusoidal lift: first half rises from h_0, second half descends to 0
        tau_c = np.clip(tau, 0.0, 1.0)
        h_m = max(1.1 * h_0, cfg.h_target)
        if tau_c < 0.5:
            z_swing = (h_m - h_0) * np.sin(np.pi * tau_c) + h_0
        else:
            z_swing = h_m * np.sin(np.pi * tau_c)

        # If overdue (tau > 1), push foot down
        z_offset = min(0.0, 1 - tau) if tau > 1 else 0.0

        return z_swing + z_offset, cfg.z_target + z_offset

    # -------------------------------------------------------------------------
    # Stance leg: height + orientation regulation
    # -------------------------------------------------------------------------
    def _compute_stance_velocity(self, z_com_target: float) -> np.ndarray:
        """
        Velocity command for stance foot to regulate CoM height and body orientation.
        Returns 3D velocity in stance frame.
        """
        cfg = self.cfg
        p = self._torso_pos

        # Unit vector from foot toward CoM
        vec_h = -p / np.linalg.norm(p)
        # Pitch correction direction
        vec_theta = np.array([-vec_h[2], 0, vec_h[0]])
        # Roll correction direction
        vec_phi = np.array([0, -1, 0])

        # Height error
        ez = p[2] - z_com_target
        v_z = cfg.K_pz * ez * vec_h

        # Orientation error (skew-symmetric extraction)
        R, Rd = self._torso_R, self.R_target
        skew = 0.5 * (Rd.T @ R - R.T @ Rd)
        e_R = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])

        v_theta = cfg.K_pt * e_R[1] * vec_theta  # pitch
        v_phi = cfg.K_pt * e_R[0] * vec_phi       # roll

        return -v_z + v_theta + v_phi

    # -------------------------------------------------------------------------
    # Swing leg: position tracking
    # -------------------------------------------------------------------------
    def _compute_swing_velocity(self, p_des: np.ndarray) -> np.ndarray:
        """Proportional controller driving swing foot toward target position."""
        cfg = self.cfg
        err = p_des - self._swing_pos
        vel = cfg.swing_gain_xy * np.array([err[0], cfg.swing_gain_y_scale * err[1], err[2]])
        return np.clip(vel, -cfg.swing_vel_limit, cfg.swing_vel_limit)

    # -------------------------------------------------------------------------
    # Yaw / turning
    # -------------------------------------------------------------------------
    def _compute_turning(self, dz_omega: float, t: float):
        """
        Compute hip-z velocity corrections for turning.
        Returns (dq_stance_z, dq_swing_z).
        """
        cfg = self.cfg
        q_l_z = self._q[0]   # left hip z
        q_r_z = self._q[5]   # right hip z
        delta_omega = dz_omega * cfg.T_des
        t_elapsed = np.clip(t - self._switch_t0, 0, cfg.T_des)
        progress = t_elapsed / cfg.T_des

        if self.stance_side == "Right":
            q_stance_z, q_swing_z = q_r_z, q_l_z
        else:
            q_stance_z, q_swing_z = q_l_z, q_r_z

        if delta_omega > 0:
            offset = min(delta_omega / 2, q_stance_z)
        else:
            offset = max(delta_omega / 2, q_stance_z)

        q_stance_des = offset - delta_omega * progress
        q_swing_des = -delta_omega / 2 + delta_omega * progress

        dq_stance_z = q_stance_des - q_stance_z
        dq_swing_z = q_swing_des - q_swing_z
        return dq_stance_z, dq_swing_z

    # -------------------------------------------------------------------------
    # Jacobian → joint velocity conversion
    # -------------------------------------------------------------------------
    def _cartesian_to_joint(self, vel_3d: np.ndarray, dq_z: float, jacobian: np.ndarray) -> np.ndarray:
        """Convert 3D Cartesian velocity + yaw correction to joint velocities via pseudoinverse."""
        v_body = self._torso_R.T @ vel_3d.reshape(3, 1)
        vel_4d = np.array([[v_body[0, 0]], [v_body[1, 0]], [v_body[2, 0]], [5 * dq_z]])
        return np.linalg.pinv(jacobian) @ vel_4d

    # -------------------------------------------------------------------------
    # Main step
    # -------------------------------------------------------------------------
    def step(self, t: float, dx_des: float, dy_des: float, dz_omega: float) -> np.ndarray:
        """
        Run one control cycle.

        Args:
            t: current simulation time
            dx_des: desired forward velocity
            dy_des: desired lateral velocity
            dz_omega: desired yaw rate (rad/s)

        Returns:
            ctrl: numpy array of length 10 (control signals for all joints)
        """
        cfg = self.cfg

        # 1. Check leg switch
        switched = self._try_switch(t)

        # 2. Read sensors (pose updates use new stance/swing labels)
        self._read_sensors()

        # 3. Capture switch state AFTER sensor read (need new swing foot pos)
        if switched:
            self._capture_switch_state(t)

        # 4. Foot placement
        p_x, p_y, e_step_down = self._compute_foot_placement(dx_des, dy_des)

        # 5. Swing height trajectory
        z_swing, z_com = self._compute_swing_height(t, e_step_down)

        # 6. Stance leg velocity (height + orientation regulation)
        stance_vel = self._compute_stance_velocity(z_com)

        # 7. Swing leg velocity (track target position)
        swing_vel = self._compute_swing_velocity(np.array([p_x, p_y, z_swing]))

        # 8. Turning
        dq_stance_z, dq_swing_z = self._compute_turning(dz_omega, t)

        # 9. Jacobians → joint velocities
        Jr, Jl = get_pos_3d_jacobians(self._q)
        if self.stance_side == "Right":
            J_stance, J_swing = Jr, Jl
        else:
            J_stance, J_swing = Jl, Jr

        dq_stance = self._cartesian_to_joint(stance_vel, dq_stance_z, J_stance)
        dq_swing = self._cartesian_to_joint(swing_vel, dq_swing_z, J_swing)

        # 10. Pack into control array
        ctrl = np.zeros(10)
        stance_idx = cfg.ctrl_indices[self.stance_side]
        swing_idx = cfg.ctrl_indices[self.swing_side]

        ctrl[stance_idx["hip_z"]] = np.clip(dq_stance[0, 0], -cfg.ctrl_clamp, cfg.ctrl_clamp)
        ctrl[stance_idx["hip_y"]] = np.clip(dq_stance[1, 0], -cfg.ctrl_clamp, cfg.ctrl_clamp)
        ctrl[stance_idx["hip"]]   = np.clip(dq_stance[2, 0], -cfg.ctrl_clamp, cfg.ctrl_clamp)
        ctrl[stance_idx["knee"]]  = np.clip(dq_stance[3, 0], -cfg.ctrl_clamp, cfg.ctrl_clamp)
        ctrl[stance_idx["foot"]]  = 0.0

        ctrl[swing_idx["hip_z"]] = np.clip(dq_swing[0, 0], -cfg.ctrl_clamp, cfg.ctrl_clamp)
        ctrl[swing_idx["hip_y"]] = np.clip(dq_swing[1, 0], -cfg.ctrl_clamp, cfg.ctrl_clamp)
        ctrl[swing_idx["hip"]]   = np.clip(dq_swing[2, 0], -cfg.ctrl_clamp, cfg.ctrl_clamp)
        ctrl[swing_idx["knee"]]  = np.clip(dq_swing[3, 0], -cfg.ctrl_clamp, cfg.ctrl_clamp)
        ctrl[swing_idx["foot"]]  = 0.0

        return ctrl