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
        self._execution_debug_logs_remaining = 5

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
        # `step_wise` preserves the old deployment contract: action values are per-step deltas.
        # `chunk_wise` means ACT inference has already decoded them into absolute target poses.
        return self._get_action_delta_alignment() == "step_wise"

    def _log_execution_alignment_once(self) -> None:
        if self._logged_execution_alignment:
            return

        execute_as_delta = self._execute_cartesian_as_delta()
        action_semantics = "delta ee pose" if execute_as_delta else "absolute target pose"
        logger.info(
            "[EXEC] action_delta_alignment=%s | cartesian actions interpreted as %s | servo_p_OL(delta=%s)",
            self._get_action_delta_alignment(),
            action_semantics,
            execute_as_delta,
        )
        self._logged_execution_alignment = True

    @staticmethod
    def _format_pose_values(values: np.ndarray) -> list[float]:
        return [round(float(value), 4) for value in values.tolist()]

    def _get_prev_semantic_ee_pose(self, arm_side: str) -> Optional[np.ndarray]:
        if self._prev_observation is None:
            return None

        prefix = f"{arm_side}_ee_pose"
        canonical_axes = ("x", "y", "z", "rx", "ry", "rz")
        stored_axes = tuple(getattr(self.config, "ee_pose_observation_axis_order", canonical_axes))
        semantic_values: dict[str, float] = {}
        for semantic_axis, stored_axis in zip(canonical_axes, stored_axes, strict=True):
            key = f"{prefix}.{stored_axis}"
            if key not in self._prev_observation:
                return None
            semantic_values[semantic_axis] = float(self._prev_observation[key])

        return np.array([semantic_values[axis] for axis in canonical_axes], dtype=float)

    def _log_cartesian_command_debug(
        self,
        *,
        execute_as_delta: bool,
        left_pose_command: np.ndarray,
        right_pose_command: np.ndarray,
    ) -> None:
        if self._execution_debug_logs_remaining <= 0:
            return

        debug_index = 6 - self._execution_debug_logs_remaining
        current_left_pose = self._get_prev_semantic_ee_pose("left")
        current_right_pose = self._get_prev_semantic_ee_pose("right")
        interpretation = "delta" if execute_as_delta else "absolute"
        logger.info(
            "[EXEC DEBUG %d/5] mode=%s | current_left=%s | command_left=%s",
            debug_index,
            interpretation,
            None if current_left_pose is None else self._format_pose_values(current_left_pose),
            self._format_pose_values(left_pose_command),
        )
        logger.info(
            "[EXEC DEBUG %d/5] mode=%s | current_right=%s | command_right=%s",
            debug_index,
            interpretation,
            None if current_right_pose is None else self._format_pose_values(current_right_pose),
            self._format_pose_values(right_pose_command),
        )
        self._execution_debug_logs_remaining -= 1

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

        # Keep the legacy feature names for compatibility. In `chunk_wise` mode ACT inference has already
        # decoded these values to absolute target poses, so execution must switch the `delta` flag instead of
        # trying to re-interpret or re-integrate the numeric values here.
        left_pose_command = np.array([
            action[f"left_delta_ee_pose.{axis}"] for axis in ["x", "y", "z", "rx", "ry", "rz"]
        ], dtype=float)
        right_pose_command = np.array([
            action[f"right_delta_ee_pose.{axis}"] for axis in ["x", "y", "z", "rx", "ry", "rz"]
        ], dtype=float)
        self._log_cartesian_command_debug(
            execute_as_delta=execute_as_delta,
            left_pose_command=left_pose_command,
            right_pose_command=right_pose_command,
        )

        if not self.config.debug:
            try:
                # `servo_p_OL` already supports both semantic modes through its `delta` flag, so the execution
                # layer's job is only to choose the correct interpretation path here. We intentionally do not
                # re-implement any delta<->absolute conversion in the robot class.
                # The small-norm suppression only makes sense for per-step delta actions. In absolute mode we
                # should forward the target pose directly, otherwise a valid non-zero absolute target would be
                # interpreted as "always large" and the old delta threshold would become meaningless.
                if execute_as_delta:
                    left_should_send = np.linalg.norm(left_pose_command) >= 0.001
                    right_should_send = np.linalg.norm(right_pose_command) >= 0.001
                else:
                    left_should_send = np.isfinite(left_pose_command).all()
                    right_should_send = np.isfinite(right_pose_command).all()

                if left_should_send:
                    # t_servo_start = time.perf_counter()
                    self._robot.servo_p_OL("left_robot", left_pose_command, delta=execute_as_delta)
                    # t_servo_end = time.perf_counter()
                    # logger.info(f"[TIMING] left servo_p_OL: {(t_servo_end-t_servo_start)*1000:.2f}ms")
                
                if right_should_send:
                    # t_servo_start = time.perf_counter()
                    self._robot.servo_p_OL("right_robot", right_pose_command, delta=execute_as_delta)
                    # t_servo_end = time.perf_counter()
                    # logger.info(f"[TIMING] right servo_p_OL: {(t_servo_end-t_servo_start)*1000:.2f}ms")
                    
            except Exception as e:
                logger.warning(f"[DUAL ARM] servo_p_OL failed: {e}")
        
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
