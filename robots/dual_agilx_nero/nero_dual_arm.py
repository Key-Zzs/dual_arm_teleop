"""
Nero dual-arm robot implementation.
Each arm has 7 DOF with agx_gripper as end effector.
Uses Oculus Quest for teleoperation control.
"""

import logging
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots.robot import Robot

from .config_nero import NeroDualArmConfig
from .nero_interface_client import NeroDualArmClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NeroDualArm(Robot):
    """
    Dual-arm Nero robot
    Each arm has 7 DOF, total 14 DOF.
    """
    
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
        
        # Gripper settings
        self._gripper_force = config.gripper_force
        self._left_gripper_cmd = 1.0
        self._right_gripper_cmd = 1.0
        # self._last_left_gripper_cmd = 1.0
        # self._last_right_gripper_cmd = 1.0

        # Action smoothing
        # self._smoothing_alpha = 0.4
        # self._left_smoothed_delta = None
        # self._right_smoothed_delta = None

        # 发送频率控制
        self.action_send_freq = 100.0  # 50Hz
        self.action_send_dt = 1.0 / self.action_send_freq
        self.last_action_send_time = 0.0
        self._logged_execution_alignment = False

    def _should_send_action(self) -> bool:
        """检查是否应该发送action（频率限制）"""
        current_time = time.time()
        if current_time - self.last_action_send_time >= self.action_send_dt:
            self.last_action_send_time = current_time
            return True
        return False

    def _get_action_delta_alignment(self) -> str:
        # Execution must not infer semantics from feature names alone:
        # both modes still use the legacy `*_delta_ee_pose.*` keys for compatibility, but the numeric meaning
        # differs once ACT chunk-wise inference is enabled.
        alignment = getattr(self.config, "action_delta_alignment", "step_wise")
        if alignment not in {"step_wise", "chunk_wise"}:
            raise ValueError(
                "`action_delta_alignment` must be either 'step_wise' or 'chunk_wise'. "
                f"Got {alignment}."
            )
        return alignment

    def _execute_cartesian_as_delta(self) -> bool:
        # True only when the action values themselves are executable per-step deltas.
        # In `chunk_wise`, the policy returns absolute target poses; execution converts each target to a
        # one-shot servo delta immediately before calling `servo_p_OL(delta=True)`.
        #
        # 关键语义：
        # - step_wise: action 数值本身就是「这一帧要走多少」。
        # - chunk_wise: action 数值是「这一帧想去哪里」，不是「要走多少」。
        # 所以这个 helper 只回答“action 原始数值能不能直接当 delta 发”，不是回答最终是否走
        # `servo_p_OL(delta=True)`。方案 B 下两个模式最终都会用 delta=True，只是 chunk_wise 需要先转换。
        return self._get_action_delta_alignment() == "step_wise"

    def _log_execution_alignment_once(self) -> None:
        if self._logged_execution_alignment:
            return

        alignment = self._get_action_delta_alignment()
        if alignment == "step_wise":
            logger.info(
                "[EXEC] action_delta_alignment=step_wise | action is already delta ee pose | "
                "servo_p_OL(delta=True)"
            )
        else:
            logger.info(
                "[EXEC] action_delta_alignment=chunk_wise | queued policy actions are absolute target poses | "
                "execution converts against the current reference pose just before send | servo_p_OL(delta=True)"
            )
        self._logged_execution_alignment = True

    def _chunkwise_reference_pose_source(self) -> str:
        source = getattr(self.config, "chunkwise_reference_pose_source", "servo_ol")
        if source not in {"servo_ol", "direct_ee", "observation"}:
            raise ValueError(
                "`chunkwise_reference_pose_source` must be one of 'servo_ol', 'direct_ee', 'observation'. "
                f"Got {source}."
            )
        return source

    @staticmethod
    def _as_finite_pose(pose: Any, *, name: str) -> np.ndarray:
        if pose is None:
            raise ValueError(f"{name} is None.")

        pose_array = np.asarray(pose, dtype=float).reshape(-1)
        if pose_array.size != 6:
            raise ValueError(f"{name} must contain 6 pose values [x, y, z, rx, ry, rz]. Got {pose_array.size}.")
        if not np.isfinite(pose_array).all():
            raise ValueError(f"{name} contains non-finite values: {pose_array.tolist()}.")
        return pose_array

    @staticmethod
    def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
        quaternion = np.asarray(quaternion, dtype=float).reshape(4)
        norm = np.linalg.norm(quaternion)
        if norm < 1e-12:
            raise ValueError("Cannot normalize a near-zero quaternion.")
        return quaternion / norm

    @classmethod
    def _quaternion_multiply(cls, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        # Returns `lhs * rhs` with the same [x, y, z, w] convention used by `servo_p_OL`.
        x1, y1, z1, w1 = cls._normalize_quaternion(lhs)
        x2, y2, z2, w2 = cls._normalize_quaternion(rhs)
        return cls._normalize_quaternion(
            np.array(
                [
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                ],
                dtype=float,
            )
        )

    @classmethod
    def _quaternion_inverse(cls, quaternion: np.ndarray) -> np.ndarray:
        x, y, z, w = cls._normalize_quaternion(quaternion)
        return np.array([-x, -y, -z, w], dtype=float)

    @classmethod
    def _euler_xyz_to_quaternion(cls, euler_xyz: np.ndarray) -> np.ndarray:
        # Match the ACT conversion helpers and the active server: roll/pitch/yaw increments are converted as
        # qz * qy * qx, then left-multiplied onto the current orientation in `servo_p_OL(delta=True)`.
        #
        # 这里不能随便换成 scipy 默认欧拉顺序，也不能简单逐轴相减。训练侧 chunk-wise delta 累积、
        # 推理侧 chunk-wise -> absolute 解码、以及 server 端 `servo_p_OL` 都依赖这一套旋转约定：
        #     target_quat = delta_quat * current_quat
        # 四元数格式保持 server 使用的 [x, y, z, w]。
        roll, pitch, yaw = np.asarray(euler_xyz, dtype=float).reshape(3) * 0.5
        qx = np.array([np.sin(roll), 0.0, 0.0, np.cos(roll)], dtype=float)
        qy = np.array([0.0, np.sin(pitch), 0.0, np.cos(pitch)], dtype=float)
        qz = np.array([0.0, 0.0, np.sin(yaw), np.cos(yaw)], dtype=float)
        return cls._quaternion_multiply(qz, cls._quaternion_multiply(qy, qx))

    @classmethod
    def _quaternion_to_euler_xyz(cls, quaternion: np.ndarray) -> np.ndarray:
        x, y, z, w = cls._normalize_quaternion(quaternion)
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return np.array([roll, pitch, yaw], dtype=float)

    @classmethod
    def _convert_absolute_pose_to_servo_delta(cls, target_pose: Any, current_pose: Any) -> np.ndarray:
        """Convert an absolute target pose into the one-shot delta expected by `servo_p_OL(delta=True)`.

        这是方案 B 的核心逆变换：
        - 输入 target_pose 来自 policy 返回的 chunk_wise absolute target action。
        - 输入 current_pose 必须是“当前发送这一帧时”的执行参考 pose，不是 chunk 起点 pose。
        - 输出 delta_pose 只用于本次 `servo_p_OL(delta=True)`，不写回 action queue。

        平移是普通差值；旋转要按 server 的增量欧拉语义反解：
            server: target_quat = delta_quat * current_quat
            here:   delta_quat = target_quat * inverse(current_quat)
        """
        target_pose = cls._as_finite_pose(target_pose, name="target_pose")
        current_pose = cls._as_finite_pose(current_pose, name="current_pose")

        delta_pose = np.zeros(6, dtype=float)
        # Translation is expressed in the same world/base pose frame as the absolute ee pose.
        delta_pose[:3] = target_pose[:3] - current_pose[:3]

        current_quat = cls._euler_xyz_to_quaternion(current_pose[3:])
        target_quat = cls._euler_xyz_to_quaternion(target_pose[3:])
        # Server-side delta mode computes: target_quat = delta_quat * current_quat.
        # Therefore the inverse conversion is: delta_quat = target_quat * inverse(current_quat).
        delta_quat = cls._quaternion_multiply(target_quat, cls._quaternion_inverse(current_quat))
        delta_pose[3:] = cls._quaternion_to_euler_xyz(delta_quat)
        return delta_pose


    def _get_prev_semantic_ee_pose(self, arm_side: str) -> Optional[np.ndarray]:
        if self._prev_observation is None:
            return None

        prefix = f"{arm_side}_ee_pose"
        canonical_axes = ("x", "y", "z", "rx", "ry", "rz")
        stored_axes = tuple(getattr(self.config, "ee_pose_observation_axis_order", canonical_axes))
        semantic_values: dict[str, float] = {}
        # `get_observation()` may store Nero's compatibility order, e.g. x/y/z/rz/ry/rx, while the execution
        # math below always expects canonical semantic order [x, y, z, rx, ry, rz].
        # This remap keeps old dataset/feature names stable without letting axis-order compatibility leak into
        # the pose conversion math.
        for semantic_axis, stored_axis in zip(canonical_axes, stored_axes, strict=True):
            key = f"{prefix}.{stored_axis}"
            if key not in self._prev_observation:
                return None
            semantic_values[semantic_axis] = float(self._prev_observation[key])

        return np.array([semantic_values[axis] for axis in canonical_axes], dtype=float)

    def _get_current_cartesian_reference_pose(self, arm_side: str) -> np.ndarray:
        """Return the absolute ee pose used to convert a chunk-wise target just before sending it."""
        if arm_side not in {"left", "right"}:
            raise ValueError(f"`arm_side` must be 'left' or 'right'. Got {arm_side}.")

        source = self._chunkwise_reference_pose_source()
        if source == "observation":
            # Compatibility path for setups that cannot query the server-side servo reference directly.
            observation_pose = self._get_prev_semantic_ee_pose(arm_side)
            if observation_pose is None:
                raise RuntimeError(
                    f"Chunk-wise reference source is 'observation', but no latest observation pose is "
                    f"available for {arm_side} arm."
                )
            try:
                return self._as_finite_pose(
                    observation_pose,
                    name=f"{arm_side} current execution reference pose from observation",
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid {arm_side} current execution reference pose in the latest observation."
                ) from exc

        # Default path: use the same open-loop reference pose that server servo_p_OL(delta=True) will compose
        # the incoming delta against.
        if self._robot is None:
            raise RuntimeError(
                f"Chunk-wise absolute action execution requires current execution reference pose for {arm_side} "
                "arm, but no robot client is available."
            )

        if source == "servo_ol":
            getter_name = f"{arm_side}_robot_get_servo_p_ol_reference_pose"
            source_description = "server servo_p_OL(delta=True) IK-state/FK reference pose"
        else:
            getter_name = f"{arm_side}_robot_get_ee_pose"
            source_description = "direct robot ee pose"

        getter = getattr(self._robot, getter_name, None)
        if getter is None:
            raise RuntimeError(
                f"Chunk-wise absolute action execution requires current execution reference pose for {arm_side} "
                f"arm from {source_description}, but robot client has no `{getter_name}` method. "
                "Set chunkwise_reference_pose_source='direct_ee' or 'observation' only when their pose "
                "semantics match the active controller."
            )

        try:
            direct_pose = getter()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to query current execution reference pose for {arm_side} arm via `{getter_name}` "
                f"({source_description})."
            ) from exc

        try:
            return self._as_finite_pose(
                direct_pose,
                name=f"{arm_side} current execution reference pose from `{getter_name}` ({source_description})",
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid {arm_side} current execution reference pose returned by `{getter_name}`."
            ) from exc

    def _execute_stepwise_delta_action(
        self,
        *,
        left_delta_command: np.ndarray,
        right_delta_command: np.ndarray,
    ) -> None:
        # Step-wise actions are already executable delta ee poses, so they keep the legacy direct delta path.
        # 这里故意不读取 current pose，也不做 absolute->delta 转换，避免影响老 checkpoint / 老数据的控制语义。
        left_should_send = np.linalg.norm(left_delta_command) >= 0.001
        right_should_send = np.linalg.norm(right_delta_command) >= 0.001

        if left_should_send:
            # t_servo_start = time.perf_counter()
            self._robot.servo_p_OL("left_robot", left_delta_command, delta=True)
            # t_servo_end = time.perf_counter()
            # logger.info(f"[TIMING] left servo_p_OL: {(t_servo_end-t_servo_start)*1000:.2f}ms")

        if right_should_send:
            # t_servo_start = time.perf_counter()
            self._robot.servo_p_OL("right_robot", right_delta_command, delta=True)
            # t_servo_end = time.perf_counter()
            # logger.info(f"[TIMING] right servo_p_OL: {(t_servo_end-t_servo_start)*1000:.2f}ms")

    def _execute_chunkwise_absolute_action_as_delta(
        self,
        *,
        left_target_abs: np.ndarray,
        right_target_abs: np.ndarray,
    ) -> None:
        # Chunk-wise policy output remains absolute all the way through the action queue. The conversion below
        # happens only at the send boundary, using the current execution reference pose for this frame.
        #
        # 也就是说，queue 里永远不存“预先算好的 delta chunk”：
        #   policy absolute target -> action queue -> send boundary -> current ref -> one-shot delta
        # 这样可以避免把整段 chunk 锚死在 chunk 起点，也避免把 absolute target 直接交给 delta=False 路径。
        left_current_ref = self._get_current_cartesian_reference_pose("left")
        right_current_ref = self._get_current_cartesian_reference_pose("right")

        left_delta_to_send = self._convert_absolute_pose_to_servo_delta(left_target_abs, left_current_ref)
        right_delta_to_send = self._convert_absolute_pose_to_servo_delta(right_target_abs, right_current_ref)

        left_should_send = np.linalg.norm(left_delta_to_send) >= 0.001
        right_should_send = np.linalg.norm(right_delta_to_send) >= 0.001

        # 方案 B 的最终落点：chunk_wise 也复用更稳定的 `servo_p_OL(delta=True)` 控制链路。
        # 这里发送的是刚刚按当前执行参考 pose 算出的 one-shot delta，而不是 policy 返回的 absolute action。
        if left_should_send:
            self._robot.servo_p_OL("left_robot", left_delta_to_send, delta=True)
        if right_should_send:
            self._robot.servo_p_OL("right_robot", right_delta_to_send, delta=True)

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the robot.
        
        Args:
            calibrate: Whether to calibrate the robot after connecting.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected.")
        
        logger.info("\n" + "=" * 60)
        logger.info("[ROBOT] Connecting to Nero Dual-Arm System")
        logger.info("=" * 60)
        
        # Connect to dual-arm server (single port)
        self._robot = self.check_nero_connection()
        # print("Nero dual-arm connected successfully.")
        
        # Connect to gripper server
        if self.config.use_gripper:
            self.initialize_grippers()
        
        # TODO: Connect cameras
        logger.info("\n===== [CAM] Initializing Cameras =====")
        for cam_name, cam in self.cameras.items():
            cam.connect()
            logger.info(f"[CAM] {cam_name} connected successfully.")
        logger.info("===== [CAM] Cameras Initialized Successfully =====\n")
        
        self.is_connected = True
        logger.info(f"[INFO] {self.name} initialization completed successfully.\n")
    
    def check_nero_connection(self) -> NeroDualArmClient:
        """Connect to Nero dual-arm server via zerorpc (single port)."""
        try:
            logger.info("\n===== [ROBOT] Connecting to Nero dual-arm =====")
            
            robot = NeroDualArmClient(
                ip=self.config.robot_ip,
                port=self.config.robot_port
            )
            # print(robot)
            # Get end-effector poses for both arms
            left_ee_pose = robot.left_robot_get_ee_pose()
            right_ee_pose = robot.right_robot_get_ee_pose()
            left_joint_pos = robot.left_robot_get_joint_positions()
            right_joint_pos = robot.right_robot_get_joint_positions()
            # print(left_ee_pose)
            # print(right_ee_pose)
            # print(left_joint_pos)
            # print(right_joint_pos)

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
        """Initialize both grippers."""
        try:
            logger.info("\n===== [GRIPPER] Initializing grippers =====")
            # self._robot.left_gripper_initialize()
            self._robot.left_gripper_goto(
                width=self.config.gripper_max_open,
                force=self._gripper_force
            )
            logger.info("[LEFT GRIPPER] Initialized successfully")
            # self._robot.right_gripper_initialize()
            self._robot.right_gripper_goto(
                width=self.config.gripper_max_open,
                force=self._gripper_force
                )
            self._left_gripper_cmd = 1.0
            self._right_gripper_cmd = 1.0
            logger.info("[RIGHT GRIPPER] Initialized successfully")
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
                force=self._gripper_force
            )
            self._robot.right_gripper_goto(
                width=self.config.gripper_max_open,
                force=self._gripper_force
            )
            self._left_gripper_cmd = 1.0
            self._right_gripper_cmd = 1.0
        
        logger.info("===== [ROBOT] Dual-arm system reset successfully =====\n")
    
    @property
    def motor_features(self) -> dict[str, type]:
        """Motor features for dual-arm system."""
        features = {}
        
        # Left arm joint positions
        for i in range(self._num_joints_per_arm):
            features[f"left_joint_{i+1}.pos"] = float
        
        # Right arm joint positions
        for i in range(self._num_joints_per_arm):
            features[f"right_joint_{i+1}.pos"] = float
        
        # Left arm end effector pose
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"left_ee_pose.{axis}"] = float
        
        # Right arm end effector pose
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"right_ee_pose.{axis}"] = float
        
        # Gripper states
        if self.config.use_gripper:
            # features["left_gripper_state_norm"] = float
            features["left_gripper_cmd"] = float
            # features["right_gripper_state_norm"] = float
            features["right_gripper_cmd"] = float
        
        return features
    
    @property
    def action_features(self) -> dict[str, type]:
        features = {}

        # # Left arm joint positions
        # for i in range(self._num_joints_per_arm):
        #     features[f"left_joint_{i+1}.pos"] = float
        
        # # Right arm joint positions
        # for i in range(self._num_joints_per_arm):
        #     features[f"right_joint_{i+1}.pos"] = float

        # Left arm delta pose
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"left_delta_ee_pose.{axis}"] = float
        # Right arm delta pose
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"right_delta_ee_pose.{axis}"] = float
        if self.config.use_gripper:
            features["left_gripper_cmd"] = float
            features["right_gripper_cmd"] = float
        return features

    @staticmethod
    def _clip_gripper_cmd(value: float) -> float:
        return min(1.0, max(0.0, float(value)))

    def handle_gripper(self, arm_side: str, gripper_value: float, is_binary: bool = False) -> None:
        t_handle_start = time.perf_counter()
        
        if not self.config.use_gripper:
            return
        
        gripper_cmd_attr = f"_{arm_side}_gripper_cmd"
        last_cmd = getattr(self, gripper_cmd_attr)
        
        if is_binary:
            if gripper_value < self.config.close_threshold:
                gripper_cmd = 0.0
            else:
                gripper_cmd = 1.0
        else:
            gripper_cmd = self._clip_gripper_cmd(gripper_value)
            # print(f"gripper_value: {gripper_value}")
        
        if self.config.gripper_reverse:
            gripper_cmd = 1.0 - gripper_cmd

        # Skip redundant command writes to reduce RPC blocking and gripper bus load.
        if last_cmd is not None and abs(gripper_cmd - last_cmd) < 1e-3:
            return
        
        try:
            if arm_side == "left":
                self._robot.left_gripper_goto(
                    width=gripper_cmd * self.config.gripper_max_open,
                    force=self._gripper_force
                )
            else:
                self._robot.right_gripper_goto(
                    width=gripper_cmd * self.config.gripper_max_open,
                    force=self._gripper_force
                )
            # print(f"width: {gripper_cmd * self.config.gripper_max_open}")
            setattr(self, gripper_cmd_attr, gripper_cmd)
        except Exception as e:
            logger.warning(f"[{arm_side.upper()} GRIPPER] zerorpc error: {e}")
        
        # t_handle_end = time.perf_counter()
        # logger.info(f"[TIMING] handle_gripper {arm_side}: {(t_handle_end-t_handle_start)*1000:.2f}ms")
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        t_send_start = time.perf_counter()
        
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Check for reset request
        if action.get("reset_requested", False):
            logger.info("[ROBOT] Reset requested for dual-arm system...")
            self._robot.robot_go_home()
            if self.config.use_gripper:
                self._robot.left_gripper_goto(
                    width=self.config.gripper_max_open,
                    force=self._gripper_force
                )
                self._robot.right_gripper_goto(
                    width=self.config.gripper_max_open,
                    force=self._gripper_force
                )
            self.reset()
            return action

        # Use joint servo control if joint positions are provided
        if not self.config.debug:
            try:
                # Action field names remain `*_delta_ee_pose.*` for backward compatibility with the existing
                # dataset / postprocessing stack. The execution semantics are selected from
                # `action_delta_alignment`, not from the legacy feature-name prefix.
                self.send_action_cartesian(action)
                    
            except Exception as e:
                logger.warning(f"[ROBOT] Action failed: {e}")
        
        # Handle grippers
        if "left_gripper_cmd" in action:
            self.handle_gripper("left", action["left_gripper_cmd"], is_binary=False)
        if "right_gripper_cmd" in action:
            self.handle_gripper("right", action["right_gripper_cmd"], is_binary=False)

        # t_send_end = time.perf_counter()
        # logger.info(f"[TIMING] send_action total: {(t_send_end-t_send_start)*1000:.2f}ms")

        return action

    def send_action_cartesian(self, action: dict[str, Any]) -> None:
        t_cart_start = time.perf_counter()
        
        # 频率限制
        if not self._should_send_action():
            return

        execute_as_delta = self._execute_cartesian_as_delta()
        self._log_execution_alignment_once()

        # Keep the legacy feature names for compatibility. In `chunk_wise` mode these values are absolute
        # target poses returned by ACT inference and stored unchanged in the action queue; only the send
        # boundary converts them to one-shot deltas for `servo_p_OL(delta=True)`.
        left_pose_command = np.array([
            action[f"left_delta_ee_pose.{axis}"] for axis in ["x", "y", "z", "rx", "ry", "rz"]
        ], dtype=float)
        right_pose_command = np.array([
            action[f"right_delta_ee_pose.{axis}"] for axis in ["x", "y", "z", "rx", "ry", "rz"]
        ], dtype=float)
        if not self.config.debug:
            try:
                # 分支语义总览：
                # - step_wise: `left/right_pose_command` 已经是可执行 delta，直接发。
                # - chunk_wise: `left/right_pose_command` 是 absolute target，先在
                #   `_execute_chunkwise_absolute_action_as_delta()` 里用当前执行参考 pose 转成 delta 再发。
                # 两条路径最终都调用 `servo_p_OL(delta=True)`，区别只在 action 原始数值是否需要转换。
                if execute_as_delta:
                    self._execute_stepwise_delta_action(
                        left_delta_command=left_pose_command,
                        right_delta_command=right_pose_command,
                    )
                else:
                    self._execute_chunkwise_absolute_action_as_delta(
                        left_target_abs=left_pose_command,
                        right_target_abs=right_pose_command,
                    )
                    
            except Exception as e:
                logger.warning(f"[DUAL ARM] cartesian action execution failed: {e}")
        
        # t_cart_end = time.perf_counter()
        # logger.info(f"[TIMING] send_action_cartesian total: {(t_cart_end-t_cart_start)*1000:.2f}ms")


    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # t_total_start = time.perf_counter()
        
        try:
            # t_query_start = time.perf_counter()
            left_joint_pos = self._robot.left_robot_get_joint_positions()
            left_ee_pose = self._robot.left_robot_get_ee_pose()
            # t_query_end = time.perf_counter()
            # logger.info(f"[TIMING] left robot query: {(t_query_end-t_query_start)*1000:.2f}ms")
            
            # t_query_start = time.perf_counter()
            right_joint_pos = self._robot.right_robot_get_joint_positions()
            right_ee_pose = self._robot.right_robot_get_ee_pose()
            # t_query_end = time.perf_counter()
            # logger.info(f"[TIMING] right robot query: {(t_query_end-t_query_start)*1000:.2f}ms")
            
        except Exception as e:
            logger.warning(f"[ROBOT] zerorpc error in get_observation: {e}")
            if self._prev_observation is not None:
                return self._prev_observation
            else:
                raise
        
        obs_dict = {}
        
        # Left arm observations
        for i in range(len(left_joint_pos)):
            obs_dict[f"left_joint_{i+1}.pos"] = float(left_joint_pos[i])

        for i, axis in enumerate(self.config.ee_pose_observation_axis_order):
            obs_dict[f"left_ee_pose.{axis}"] = float(left_ee_pose[i])
        
        # Right arm observations
        for i in range(len(right_joint_pos)):
            obs_dict[f"right_joint_{i+1}.pos"] = float(right_joint_pos[i])

        for i, axis in enumerate(self.config.ee_pose_observation_axis_order):
            obs_dict[f"right_ee_pose.{axis}"] = float(right_ee_pose[i])
        
        # Gripper states
        if self.config.use_gripper:
            obs_dict["left_gripper_cmd"] = self._left_gripper_cmd
            obs_dict["right_gripper_cmd"] = self._right_gripper_cmd
        else:
            obs_dict["left_gripper_cmd"] = None
            obs_dict["right_gripper_cmd"] = None

        # TODO: Camera images
        # t_cam_total_start = time.perf_counter()
        for cam_key, cam in self.cameras.items():
            # t_cam_start = time.perf_counter()
            obs_dict[cam_key] = cam.read()
            # t_cam_end = time.perf_counter()
            # logger.info(f"[TIMING] {cam_key} read: {(t_cam_end-t_cam_start)*1000:.2f}ms")
        # t_cam_total_end = time.perf_counter()
        # logger.info(f"[TIMING] camera total: {(t_cam_total_end-t_cam_total_start)*1000:.2f}ms")
        
        self._prev_observation = obs_dict
        # t_total_end = time.perf_counter()
        # logger.info(f"[TIMING] get_observation total: {(t_total_end-t_total_start)*1000:.2f}ms")
        return obs_dict
    
    def disconnect(self) -> None:
        if not self.is_connected:
            return
        
        # TODO: Disconnect cameras
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
    
    @property
    def cameras_features(self) -> dict[str, tuple]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3) 
            for cam in self.cameras
        }
    
    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self.motor_features, **self.cameras_features}
