"""
Nero dual-arm robot implementation.

Final right_umi_14 deployment contract for your current dataset:

Training dataset schema:
  observation.state: 14D = [left7, right7]
  action:            14D = [left7_action, right7_action]

For your single-arm dataset:
  left7 = zeros

right7 observation:
  [
    x - x0,
    y - y0,
    z - z0,
    wrap(rx - rx0),
    wrap(ry - ry0),
    wrap(rz - rz0),
    gripper_width / first_gripper_width
  ]

right7 action:
  [
    dx,
    dy,
    dz,
    drx,
    dry,
    drz,
    next_gripper_normalized
  ]

Important Nero interface facts:
  Nero observation pose order:
    [x, y, z, rz, ry, rx]

  Nero servo action order:
    [x, y, z, rx, ry, rz]

Therefore:
  Observation conversion:
    Nero obs [x,y,z,rz,ry,rx]
      -> policy absolute pose [x,y,z,rx,ry,rz]
      -> policy observation delta from first frame

  Action conversion:
    policy [dx,dy,dz,drx,dry,drz]
      -> Nero servo [dx,dy,dz,drx,dry,drz]
"""

import logging
import time
from typing import Any, Optional

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_nero import NeroDualArmConfig
from .nero_interface_client import NeroDualArmClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NeroDualArm(Robot):
    config_class = NeroDualArmConfig
    name = "nero_dual_arm"

    def __init__(self, config: NeroDualArmConfig):
        super().__init__(config)

        self.cameras = make_cameras_from_configs(config.cameras)

        self.config = config
        self._is_connected = False
        self._robot: Optional[NeroDualArmClient] = None
        self._prev_observation = None
        self._num_joints_per_arm = 7

        # Gripper settings.
        self._gripper_force = config.gripper_force
        self._left_gripper_cmd = 1.0
        self._right_gripper_cmd = 1.0

        # Action sending frequency.
        self.action_send_freq = float(getattr(config, "action_send_freq", 100.0))
        self.action_send_dt = 1.0 / self.action_send_freq
        self.last_action_send_time = 0.0

        # right_umi_14 reference for observation delta-from-first.
        # Training observation.state is relative to first frame of each episode.
        self._right_umi_ref_pose6_policy: Optional[np.ndarray] = None
        self._right_umi_ref_gripper_width: Optional[float] = None

        logger.info(
            "[NeroDualArm] policy_compat_mode=%s active_arm=%s",
            getattr(config, "policy_compat_mode", "dual"),
            getattr(config, "active_arm", "both"),
        )

    # ---------------------------------------------------------------------
    # Mode helpers
    # ---------------------------------------------------------------------

    def _is_right_umi_14_mode(self) -> bool:
        return (
            str(getattr(self.config, "policy_compat_mode", "dual")) == "right_umi_14"
            and str(getattr(self.config, "active_arm", "both")) == "right"
        )

    @staticmethod
    def _right_umi_names() -> list[str]:
        return [
            "localization.pose.pika_l.x",
            "localization.pose.pika_l.y",
            "localization.pose.pika_l.z",
            "localization.pose.pika_l.roll",
            "localization.pose.pika_l.pitch",
            "localization.pose.pika_l.yaw",
            "localization.pose.pika_l.gripper",
            "localization.pose.pika_r.x",
            "localization.pose.pika_r.y",
            "localization.pose.pika_r.z",
            "localization.pose.pika_r.roll",
            "localization.pose.pika_r.pitch",
            "localization.pose.pika_r.yaw",
            "localization.pose.pika_r.gripper",
        ]

    @staticmethod
    def _wrap_to_pi(x: Any) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _nero_obs6_to_policy_pose6(nero_obs6: Any) -> np.ndarray:
        """
        Nero observation:
          [x, y, z, rz, ry, rx]

        Policy pose:
          [x, y, z, rx, ry, rz]
        """
        p = np.asarray(nero_obs6, dtype=np.float64).reshape(-1)

        if p.shape[0] < 6:
            raise ValueError(f"Expected Nero obs pose len>=6, got {p.shape}")

        return np.array(
            [
                p[0],  # x
                p[1],  # y
                p[2],  # z
                p[5],  # rx
                p[4],  # ry
                p[3],  # rz
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _policy_delta6_to_nero_action6(policy_delta6: Any) -> np.ndarray:
        """
        Policy action:
          [dx, dy, dz, drx, dry, drz]

        Nero action:
          [dx, dy, dz, drx, dry, drz]

        Your latest check says Nero action is already [x,y,z,rx,ry,rz],
        so no rotation reordering here.
        """
        d = np.asarray(policy_delta6, dtype=np.float64).reshape(-1)

        if d.shape[0] < 6:
            raise ValueError(f"Expected policy delta len>=6, got {d.shape}")

        return np.array(
            [
                d[0],
                d[1],
                d[2],
                d[3],
                d[4],
                d[5],
            ],
            dtype=np.float64,
        )

    def _reset_right_umi_reference(self) -> None:
        """
        Reset first-frame reference for right_umi_14 observation.
        Called on reset / new deployment start.
        """
        self._right_umi_ref_pose6_policy = None
        self._right_umi_ref_gripper_width = None

    def _get_current_right_gripper_width(self) -> float:
        """
        Estimate current gripper physical width.

        The robot API in this file only stores the last normalized command.
        So current width is approximated as:
          right_gripper_cmd * gripper_max_open

        If you later add real gripper feedback, replace this function.
        """
        return float(self._right_gripper_cmd) * float(self.config.gripper_max_open)

    def _make_right_umi_state14(self, right_ee_pose_nero_order: Any) -> np.ndarray:
        """
        Build 14D observation.state for current new pipeline.

        left7:
          zeros

        right7:
          [
            x - x0,
            y - y0,
            z - z0,
            wrap(rx - rx0),
            wrap(ry - ry0),
            wrap(rz - rz0),
            gripper_width / first_gripper_width
          ]
        """
        left7 = np.zeros(7, dtype=np.float64)

        right_pose6_policy_abs = self._nero_obs6_to_policy_pose6(right_ee_pose_nero_order)
        right_gripper_width = self._get_current_right_gripper_width()

        # Initialize first-frame reference.
        if self._right_umi_ref_pose6_policy is None:
            self._right_umi_ref_pose6_policy = right_pose6_policy_abs.copy()
            logger.info(
                "[right_umi_14] Set first pose reference: %s",
                np.round(self._right_umi_ref_pose6_policy, 6).tolist(),
            )

        if self._right_umi_ref_gripper_width is None:
            # Avoid division by zero. If first width is invalid, fall back to max_open.
            ref_width = float(right_gripper_width)

            if abs(ref_width) < 1e-9:
                ref_width = float(self.config.gripper_max_open)

            if abs(ref_width) < 1e-9:
                ref_width = 1.0

            self._right_umi_ref_gripper_width = ref_width

            logger.info(
                "[right_umi_14] Set first gripper width reference: %.6f",
                self._right_umi_ref_gripper_width,
            )

        ref_pose = self._right_umi_ref_pose6_policy
        ref_gripper_width = float(self._right_umi_ref_gripper_width)

        right_pose_delta = np.zeros(6, dtype=np.float64)

        # xyz delta
        right_pose_delta[0:3] = right_pose6_policy_abs[0:3] - ref_pose[0:3]

        # rpy delta with angle wrap
        right_pose_delta[3] = self._wrap_to_pi(right_pose6_policy_abs[3] - ref_pose[3])
        right_pose_delta[4] = self._wrap_to_pi(right_pose6_policy_abs[4] - ref_pose[4])
        right_pose_delta[5] = self._wrap_to_pi(right_pose6_policy_abs[5] - ref_pose[5])

        # normalized gripper.
        if abs(ref_gripper_width) < 1e-9:
            right_gripper_norm = 1.0
        else:
            right_gripper_norm = float(right_gripper_width) / ref_gripper_width

        right_gripper_norm = float(np.clip(right_gripper_norm, 0.0, 1.0))

        right7 = np.concatenate(
            [
                right_pose_delta,
                np.array([right_gripper_norm], dtype=np.float64),
            ],
            axis=0,
        )

        state14 = np.concatenate([left7, right7], axis=0)

        if state14.shape != (14,):
            raise RuntimeError(f"state14 shape error: {state14.shape}")

        return state14.astype(np.float64)

    def _policy_gripper_norm_to_cmd(self, policy_gripper_norm: float) -> float:
        """
        Convert policy normalized gripper target to Nero normalized command.

        Training:
          gripper_norm = gripper_width / first_gripper_width

        Deployment:
          target_width = gripper_norm * first_gripper_width
          command = target_width / gripper_max_open

        If first_gripper_width == gripper_max_open, this is almost identity.
        """
        g = float(policy_gripper_norm)
        g = float(np.clip(g, 0.0, 1.0))

        ref_width = self._right_umi_ref_gripper_width

        if ref_width is None or abs(float(ref_width)) < 1e-9:
            ref_width = float(self.config.gripper_max_open)

        target_width = g * float(ref_width)

        max_open = float(self.config.gripper_max_open)

        if abs(max_open) < 1e-9:
            return self._clip_gripper_cmd(g)

        cmd = target_width / max_open

        return self._clip_gripper_cmd(cmd)

    def _get_action_vector14(self, action: dict[str, Any]) -> Optional[np.ndarray]:
        """
        Accept either:
          1. action["action"] is a 14D vector
          2. action contains 14 individual keys from _right_umi_names()
        """
        if "action" in action:
            vec = action["action"]

            try:
                if hasattr(vec, "detach"):
                    vec = vec.detach().cpu().numpy()

                vec = np.asarray(vec, dtype=np.float64).reshape(-1)

                if vec.shape[0] == 14:
                    return vec

            except Exception:
                pass

        names = self._right_umi_names()

        if all(k in action for k in names):
            try:
                return np.array([action[k] for k in names], dtype=np.float64)
            except Exception:
                return None

        return None

    def _limit_policy_delta(self, delta6: np.ndarray, current_right_ee_pose: Optional[Any]) -> np.ndarray:
        """
        Limit policy delta before sending to Nero.

        delta6 order:
          [dx, dy, dz, drx, dry, drz]
        """
        delta6 = np.asarray(delta6, dtype=np.float64).reshape(6).copy()

        max_pos = float(getattr(self.config, "max_single_arm_pos_delta", 0.01))
        max_rot = float(getattr(self.config, "max_single_arm_rot_delta", 0.05))

        enable_xyz = bool(getattr(self.config, "enable_policy_xyz", True))
        enable_rot = bool(getattr(self.config, "enable_policy_rotation", True))

        if enable_xyz:
            delta6[0:3] = np.clip(delta6[0:3], -max_pos, max_pos)
        else:
            delta6[0:3] = 0.0

        if enable_rot:
            delta6[3:6] = np.clip(delta6[3:6], -max_rot, max_rot)
        else:
            delta6[3:6] = 0.0

        min_ee_z = getattr(self.config, "min_ee_z", None)
        max_ee_z = getattr(self.config, "max_ee_z", None)

        if current_right_ee_pose is not None and (min_ee_z is not None or max_ee_z is not None):
            cur = np.asarray(current_right_ee_pose, dtype=np.float64).reshape(-1)

            if cur.shape[0] >= 3:
                cur_z = float(cur[2])
                next_z = cur_z + float(delta6[2])

                if min_ee_z is not None and next_z < float(min_ee_z):
                    delta6[2] = float(min_ee_z) - cur_z

                if max_ee_z is not None and next_z > float(max_ee_z):
                    delta6[2] = float(max_ee_z) - cur_z

        return delta6

    # ---------------------------------------------------------------------
    # Connection
    # ---------------------------------------------------------------------

    def _should_send_action(self) -> bool:
        current_time = time.time()

        if current_time - self.last_action_send_time >= self.action_send_dt:
            self.last_action_send_time = current_time
            return True

        return False

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected.")

        logger.info("\n" + "=" * 60)
        logger.info("[ROBOT] Connecting to Nero Dual-Arm System")
        logger.info("=" * 60)

        self._robot = self.check_nero_connection()

        if self.config.use_gripper:
            self.initialize_grippers()

        self._reset_right_umi_reference()

        logger.info("\n===== [CAM] Initializing Cameras =====")

        for cam_name, cam in self.cameras.items():
            cam.connect()
            logger.info(f"[CAM] {cam_name} connected successfully.")

        logger.info("===== [CAM] Cameras Initialized Successfully =====\n")

        self.is_connected = True
        logger.info(f"[INFO] {self.name} initialization completed successfully.\n")

    def check_nero_connection(self) -> NeroDualArmClient:
        try:
            logger.info("\n===== [ROBOT] Connecting to Nero dual-arm =====")

            robot = NeroDualArmClient(ip=self.config.robot_ip, port=self.config.robot_port)

            left_ee_pose = robot.left_robot_get_ee_pose()
            right_ee_pose = robot.right_robot_get_ee_pose()
            left_joint_pos = robot.left_robot_get_joint_positions()
            right_joint_pos = robot.right_robot_get_joint_positions()

            if left_ee_pose is not None and len(left_ee_pose) == 6:
                logger.info(f"[LEFT ARM] End-effector pose: {[round(j, 4) for j in left_ee_pose]}")

            if right_ee_pose is not None and len(right_ee_pose) == 6:
                logger.info(f"[RIGHT ARM] End-effector pose: {[round(j, 4) for j in right_ee_pose]}")

            if left_joint_pos is not None and len(left_joint_pos) == self._num_joints_per_arm:
                logger.info(f"[LEFT ARM] Joint positions: {[round(j, 4) for j in left_joint_pos]}")

            if right_joint_pos is not None and len(right_joint_pos) == self._num_joints_per_arm:
                logger.info(f"[RIGHT ARM] Joint positions: {[round(j, 4) for j in right_joint_pos]}")

            logger.info("===== [ROBOT] Nero dual-arm connected successfully =====\n")

            return robot

        except Exception as e:
            logger.error("===== [ERROR] Failed to connect to Nero dual-arm =====")
            logger.error(f"Exception: {e}\n")
            raise

    def initialize_grippers(self) -> None:
        try:
            logger.info("\n===== [GRIPPER] Initializing grippers =====")

            self._robot.left_gripper_goto(
                width=self.config.gripper_max_open,
                force=self._gripper_force,
            )
            logger.info("[LEFT GRIPPER] Initialized successfully")

            self._robot.right_gripper_goto(
                width=self.config.gripper_max_open,
                force=self._gripper_force,
            )
            logger.info("[RIGHT GRIPPER] Initialized successfully")

            self._left_gripper_cmd = 1.0
            self._right_gripper_cmd = 1.0

            logger.info("===== [GRIPPER] Grippers initialized successfully =====\n")

        except Exception as e:
            logger.error("===== [ERROR] Failed to initialize grippers =====")
            logger.error(f"Exception: {e}\n")

    def reset(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        logger.info("[ROBOT] Resetting dual-arm system...")

        self._robot.robot_go_home()

        if self.config.use_gripper:
            self._robot.left_gripper_goto(
                width=self.config.gripper_max_open,
                force=self._gripper_force,
            )
            self._robot.right_gripper_goto(
                width=self.config.gripper_max_open,
                force=self._gripper_force,
            )

            self._left_gripper_cmd = 1.0
            self._right_gripper_cmd = 1.0

        self._reset_right_umi_reference()

        logger.info("===== [ROBOT] Dual-arm system reset successfully =====\n")

    # ---------------------------------------------------------------------
    # Features
    # ---------------------------------------------------------------------

    @property
    def motor_features(self) -> dict[str, type]:
        if self._is_right_umi_14_mode():
            return {name: float for name in self._right_umi_names()}

        features: dict[str, type] = {}

        for i in range(self._num_joints_per_arm):
            features[f"left_joint_{i+1}.pos"] = float

        for i in range(self._num_joints_per_arm):
            features[f"right_joint_{i+1}.pos"] = float

        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"left_ee_pose.{axis}"] = float

        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"right_ee_pose.{axis}"] = float

        if self.config.use_gripper:
            features["left_gripper_cmd"] = float
            features["right_gripper_cmd"] = float

        return features

    @property
    def action_features(self) -> dict[str, type]:
        if self._is_right_umi_14_mode():
            return {name: float for name in self._right_umi_names()}

        features: dict[str, type] = {}

        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"left_delta_ee_pose.{axis}"] = float

        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"right_delta_ee_pose.{axis}"] = float

        if self.config.use_gripper:
            features["left_gripper_cmd"] = float
            features["right_gripper_cmd"] = float
            features["gripper_cmd"] = float

        return features

    @property
    def cameras_features(self) -> dict[str, tuple]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self.motor_features, **self.cameras_features}

    # ---------------------------------------------------------------------
    # Gripper
    # ---------------------------------------------------------------------

    @staticmethod
    def _clip_gripper_cmd(value: float) -> float:
        return min(1.0, max(0.0, float(value)))

    def handle_gripper(self, arm_side: str, gripper_value: float, is_binary: bool = False) -> None:
        if not self.config.use_gripper:
            return

        enable_gripper = bool(getattr(self.config, "enable_policy_gripper", True))

        if not enable_gripper and arm_side == "right":
            return

        gripper_cmd_attr = f"_{arm_side}_gripper_cmd"
        last_cmd = getattr(self, gripper_cmd_attr)

        if is_binary:
            gripper_cmd = 0.0 if gripper_value < self.config.close_threshold else 1.0
        else:
            gripper_cmd = self._clip_gripper_cmd(gripper_value)

        if self.config.gripper_reverse:
            gripper_cmd = 1.0 - gripper_cmd

        max_delta = getattr(self.config, "max_gripper_cmd_delta", None)

        if max_delta is not None and last_cmd is not None:
            max_delta = float(max_delta)
            gripper_cmd = float(
                np.clip(
                    gripper_cmd,
                    float(last_cmd) - max_delta,
                    float(last_cmd) + max_delta,
                )
            )

        if last_cmd is not None and abs(gripper_cmd - last_cmd) < 1e-3:
            return

        try:
            if arm_side == "left":
                self._robot.left_gripper_goto(
                    width=gripper_cmd * self.config.gripper_max_open,
                    force=self._gripper_force,
                )
            else:
                self._robot.right_gripper_goto(
                    width=gripper_cmd * self.config.gripper_max_open,
                    force=self._gripper_force,
                )

            setattr(self, gripper_cmd_attr, gripper_cmd)

        except Exception as e:
            logger.warning(f"[{arm_side.upper()} GRIPPER] zerorpc error: {e}")

    # ---------------------------------------------------------------------
    # Action helpers
    # ---------------------------------------------------------------------

    def _has_dual_arm_delta(self, action: dict[str, Any]) -> bool:
        left_req = [f"left_delta_ee_pose.{a}" for a in ["x", "y", "z", "rx", "ry", "rz"]]
        right_req = [f"right_delta_ee_pose.{a}" for a in ["x", "y", "z", "rx", "ry", "rz"]]

        return all(k in action for k in left_req + right_req)

    def _has_single_right_delta(self, action: dict[str, Any]) -> bool:
        req = [f"delta_ee_pose.{a}" for a in ["x", "y", "z", "rx", "ry", "rz"]]

        return all(k in action for k in req)

    def _get_action_vector7(self, action: dict[str, Any]) -> Optional[np.ndarray]:
        if "action" not in action:
            return None

        vec = action["action"]

        try:
            if hasattr(vec, "detach"):
                vec = vec.detach().cpu().numpy()

            vec = np.asarray(vec, dtype=np.float64).reshape(-1)

            if vec.shape[0] != 7:
                return None

            return vec

        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Send action
    # ---------------------------------------------------------------------

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if action.get("reset_requested", False):
            logger.info("[ROBOT] Reset requested for dual-arm system...")

            self._robot.robot_go_home()

            if self.config.use_gripper:
                self._robot.left_gripper_goto(
                    width=self.config.gripper_max_open,
                    force=self._gripper_force,
                )
                self._robot.right_gripper_goto(
                    width=self.config.gripper_max_open,
                    force=self._gripper_force,
                )

                self._left_gripper_cmd = 1.0
                self._right_gripper_cmd = 1.0

            self._reset_right_umi_reference()
            self.reset()

            return action

        if not self.config.debug:
            try:
                self.send_action_cartesian(action)
            except Exception as e:
                logger.warning(f"[ROBOT] Action failed: {e}")

        if "left_gripper_cmd" in action:
            self.handle_gripper("left", action["left_gripper_cmd"], is_binary=False)

        if "right_gripper_cmd" in action:
            self.handle_gripper("right", action["right_gripper_cmd"], is_binary=False)

        if "gripper_cmd" in action:
            self.handle_gripper("right", action["gripper_cmd"], is_binary=False)

        return action

    def send_action_cartesian(self, action: dict[str, Any]) -> None:
        if not self._should_send_action():
            return

        if self.config.debug:
            return

        try:
            # -------------------------------------------------------------
            # 0) right_umi_14 policy mode
            # -------------------------------------------------------------
            if self._is_right_umi_14_mode():
                vec14 = self._get_action_vector14(action)

                if vec14 is None:
                    logger.warning(
                        "[right_umi_14] Could not parse 14D action. Keys=%s",
                        list(action.keys()),
                    )
                    return

                right7 = vec14[7:14]

                # policy action right7:
                # [dx, dy, dz, drx, dry, drz, gripper_norm_target]
                policy_delta6 = right7[0:6]

                current_right_pose = None

                try:
                    current_right_pose = self._robot.right_robot_get_ee_pose()
                except Exception:
                    current_right_pose = None

                nero_delta6 = self._policy_delta6_to_nero_action6(policy_delta6)
                nero_delta6 = self._limit_policy_delta(
                    nero_delta6,
                    current_right_ee_pose=current_right_pose,
                )

                norm = float(np.linalg.norm(nero_delta6))

                logger.info(
                    "[right_umi_14] raw_right7=%s nero_delta6=%s norm=%.6f grip_norm_raw=%.6f",
                    np.round(right7, 6).tolist(),
                    np.round(nero_delta6, 6).tolist(),
                    norm,
                    float(right7[6]),
                )

                if norm >= 1e-6:
                    self._robot.servo_p_OL(
                        "right_robot",
                        nero_delta6,
                        delta=True,
                    )

                if self.config.use_gripper:
                    grip_cmd = self._policy_gripper_norm_to_cmd(float(right7[6]))
                    action["gripper_cmd"] = grip_cmd

                return

            # -------------------------------------------------------------
            # 1) Original dual-arm delta mode
            # -------------------------------------------------------------
            if self._has_dual_arm_delta(action):
                left_delta = np.array(
                    [
                        action[f"left_delta_ee_pose.{axis}"]
                        for axis in ["x", "y", "z", "rx", "ry", "rz"]
                    ],
                    dtype=np.float64,
                )

                right_delta = np.array(
                    [
                        action[f"right_delta_ee_pose.{axis}"]
                        for axis in ["x", "y", "z", "rx", "ry", "rz"]
                    ],
                    dtype=np.float64,
                )

                if float(np.linalg.norm(left_delta)) >= 0.001:
                    self._robot.servo_p_OL("left_robot", left_delta, delta=True)

                if float(np.linalg.norm(right_delta)) >= 0.001:
                    self._robot.servo_p_OL("right_robot", right_delta, delta=True)

                return

            # -------------------------------------------------------------
            # 2) Single-arm RIGHT explicit delta keys
            # -------------------------------------------------------------
            if self._has_single_right_delta(action):
                right_delta = np.array(
                    [
                        action[f"delta_ee_pose.{axis}"]
                        for axis in ["x", "y", "z", "rx", "ry", "rz"]
                    ],
                    dtype=np.float64,
                )

                if float(np.linalg.norm(right_delta)) >= 0.001:
                    self._robot.servo_p_OL("right_robot", right_delta, delta=True)

                return

            # -------------------------------------------------------------
            # 3) Compatibility: 7D action vector
            # -------------------------------------------------------------
            vec7 = self._get_action_vector7(action)

            if vec7 is not None:
                right_delta = vec7[:6]

                if float(np.linalg.norm(right_delta)) >= 0.001:
                    self._robot.servo_p_OL("right_robot", right_delta, delta=True)

                if self.config.use_gripper:
                    action["gripper_cmd"] = float(vec7[6])

                return

        except Exception as e:
            logger.warning(f"[DUAL ARM] servo_p_OL failed: {e}")

    # ---------------------------------------------------------------------
    # Observation
    # ---------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            # -------------------------------------------------------------
            # right_umi_14 policy mode
            # -------------------------------------------------------------
            if self._is_right_umi_14_mode():
                right_ee_pose = self._robot.right_robot_get_ee_pose()

                obs_dict: dict[str, Any] = {}

                state14 = self._make_right_umi_state14(right_ee_pose)
                names = self._right_umi_names()

                for i, name in enumerate(names):
                    obs_dict[name] = float(state14[i])

                for cam_key, cam in self.cameras.items():
                    obs_dict[cam_key] = cam.read()

                self._prev_observation = obs_dict

                return obs_dict

            # -------------------------------------------------------------
            # Original dual-arm observation mode
            # -------------------------------------------------------------
            left_joint_pos = self._robot.left_robot_get_joint_positions()
            left_ee_pose = self._robot.left_robot_get_ee_pose()

            right_joint_pos = self._robot.right_robot_get_joint_positions()
            right_ee_pose = self._robot.right_robot_get_ee_pose()

        except Exception as e:
            logger.warning(f"[ROBOT] zerorpc error in get_observation: {e}")

            if self._prev_observation is not None:
                return self._prev_observation

            raise

        obs_dict: dict[str, Any] = {}

        for i in range(len(left_joint_pos)):
            obs_dict[f"left_joint_{i+1}.pos"] = float(left_joint_pos[i])

        # Original Nero observation order from your source:
        # [x, y, z, rz, ry, rx]
        for i, axis in enumerate(["x", "y", "z", "rz", "ry", "rx"]):
            obs_dict[f"left_ee_pose.{axis}"] = float(left_ee_pose[i])

        for i in range(len(right_joint_pos)):
            obs_dict[f"right_joint_{i+1}.pos"] = float(right_joint_pos[i])

        for i, axis in enumerate(["x", "y", "z", "rz", "ry", "rx"]):
            obs_dict[f"right_ee_pose.{axis}"] = float(right_ee_pose[i])

        if self.config.use_gripper:
            obs_dict["left_gripper_cmd"] = self._left_gripper_cmd
            obs_dict["right_gripper_cmd"] = self._right_gripper_cmd
        else:
            obs_dict["left_gripper_cmd"] = None
            obs_dict["right_gripper_cmd"] = None

        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.read()

        self._prev_observation = obs_dict

        return obs_dict

    # ---------------------------------------------------------------------
    # Disconnect etc.
    # ---------------------------------------------------------------------

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        for cam in self.cameras.values():
            cam.disconnect()

        if self._robot is not None:
            self._robot.close()

        self.is_connected = False
        logger.info(f"[INFO] ===== {self.name} disconnected =====")

    def calibrate(self) -> None:
        pass

    def is_calibrated(self) -> bool:
        return self.is_connected

    def configure(self) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self._is_connected = value

