#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pathlib import Path
from typing import Any, Dict
import webbrowser
import yaml
import placo
import time
from lerobot.teleoperators.teleoperator import Teleoperator
from .config_teleop import VRTeleopConfig
from rtde_receive import RTDEReceiveInterface
from xrobotoolkit_teleop.common.xr_client import XrClient
from placo_utils.visualization import frame_viz, robot_frame_viz, robot_viz
import threading
from pathlib import Path
from xrobotoolkit_teleop.hardware.interface.universal_robots import CONTROLLER_DEADZONE
import numpy as np
import meshcat.transformations as tf
from xrobotoolkit_teleop.utils.geometry import apply_delta_pose, quat_diff_as_angle_axis
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

DEFAULT_MANIPULATOR_CONFIG = {
    "left_arm": {
        "base_link": "left_base",
        "link_name": "left_tcp",
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
        "gripper_trigger": "left_trigger",
    },
    "right_arm": {
        "base_link": "right_base",
        "link_name": "right_tcp",
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_trigger": "right_trigger",
    },
}

ARM_MAP = {
    "left_arm": {
        "last": "_last_left_trigger_val",
        "pos": "left_gripper_pos",
    },
    "right_arm": {
        "last": "_last_right_trigger_val",
        "pos": "right_gripper_pos",
    },
}

class VRTeleop(Teleoperator):
    """
    VR Teleop class for controlling a single robot arm.
    """

    config_class = VRTeleopConfig
    name = "VRTeleop"

    def __init__(self, config: VRTeleopConfig):
        """
        初始化VR遥操作控制器。
        
        Args:
            config: VR遥操作配置对象，包含机器人IP、XR客户端等配置信息
        """
        super().__init__(config)
        self.xr_client = config.xr_client
        self.cfg = config
        self._arm = {}
        self.effector_task = {}
        self.init_ee_xyz = {}
        self.last_target_ee = {}
        self.init_ee_quat = {}
        self.init_controller_xyz = {}
        self.init_controller_quat = {}
        self._is_connected = False
        self._stop_event = threading.Event()
        self._last_left_trigger_val = 1.0
        self._last_right_trigger_val = 1.0
        self.delta_xyz_left_base = np.zeros(3)
        self.delta_rot_angle_axis_left_base = np.array([0.0, 0.0, 0.0])
        self.delta_xyz_right_base = np.zeros(3)
        self.delta_rot_angle_axis_right_base = np.array([0.0, 0.0, 0.0])
        self.manipulator_config = DEFAULT_MANIPULATOR_CONFIG
        self.R_headset_world = R.from_euler('ZYX', self.cfg.R_headset_world, degrees=True).as_matrix()
        self.robot_urdf_path = Path(__file__).parents[2] / self.cfg.robot_urdf_path
    
    @property
    def action_features(self) -> dict:
        """返回动作特征字典，当前为空。"""
        return {}

    @property
    def feedback_features(self) -> dict:
        """返回反馈特征字典，当前为空。"""
        return {}

    @property
    def is_connected(self) -> bool:
        """检查是否已连接到机器人和VR设备。"""
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """检查是否已校准，当前未实现。"""
        pass

    def connect(self) -> None:
        """
        连接到UR机器人和VR设备，初始化所有必要的组件。
        包括：连接机器人、设置Placo运动学求解器、初始化末端执行器、
        启动可视化（如果启用）以及启动关节位置更新线程。
        """
        # Connect to robot
        self._arm['left_rtde_r'] = self._check_ur_connection(self.cfg.left_robot_ip, "left")
        self._arm['right_rtde_r'] = self._check_ur_connection(self.cfg.right_robot_ip, "right")
        
        # Check Placo Setup
        self._check_placo_setup()
        
        # Check End-effector Setup
        self._check_endeffector_setup()

        # Initialize qpos targets
        self._init_qpos()

        # Connect to visualize Placo
        if self.cfg.visualize_placo:
            self._start_placo_visualizer()

        # Start qpos update thread from xrclient
        threading.Thread(target=self._start_qpos_update, daemon=True).start()

        # Check completed
        self._is_connected = True
        logger.info(f"[INFO] {self.name} env initialization completed successfully.\n")

    def _check_endeffector_setup(self):
        """
        检查并设置末端执行器任务。
        为每个机械臂创建帧任务和可操作性任务，并初始化相关位姿变量。
        """
        for name, config in self.manipulator_config.items():
            initial_pose = np.eye(4)
            self.effector_task[name] = self.solver.add_frame_task(config["link_name"], initial_pose)
            self.effector_task[name].configure(f"{name}_frame", "soft", 1.0)
            manipulability = self.solver.add_manipulability_task(config["link_name"], "both", 1.0)
            manipulability.configure(f"{name}_manipulability", "soft", 5e-2)
            self.init_ee_xyz[name] = np.array([0, 0, 0])
            self.init_ee_quat[name] = np.array([1, 0, 0, 0])
            self.init_controller_xyz[name] = np.array([0, 0, 0])
            self.init_controller_quat[name] = np.array([1, 0, 0, 0])

    def _check_placo_setup(self):
        """
        检查并设置Placo运动学求解器。
        加载机器人URDF模型，创建运动学求解器，并配置求解器参数。
        """
        # Placo Setup
        self.placo_robot = placo.RobotWrapper(str(self.robot_urdf_path))
        self.solver = placo.KinematicsSolver(self.placo_robot)
        self.solver.dt = self.cfg.servo_time
        self.solver.mask_fbase(True)
        self.solver.add_kinetic_energy_regularization_task(1e-6)

    def _init_qpos(self):
        """
        初始化关节位置。
        从实际机器人读取当前关节角度，并设置到Placo模型中作为初始目标位置。
        同时初始化夹爪位置为打开状态。
        """
        left_qpos_init = np.array(self._arm["left_rtde_r"].getActualQ())
        right_qpos_init = np.array(self._arm["right_rtde_r"].getActualQ())

        self.placo_robot.state.q[7:13] = left_qpos_init
        self.placo_robot.state.q[13:19] = right_qpos_init

        self.target_left_q = left_qpos_init.copy()
        self.target_right_q = right_qpos_init.copy()
        self.left_gripper_pos = self.cfg.open_position
        self.right_gripper_pos = self.cfg.open_position

    def _check_ur_connection(self, robot_ip: str, arm_name: str):
        """
        检查并建立与UR机器人的连接。
        
        Args:
            robot_ip: 机器人的IP地址
            arm_name: 机械臂名称（用于日志输出）
            
        Returns:
            RTDEReceiveInterface: RTDE接收接口实例
        """
        try:
            logger.info(f"\n===== [ROBOT] Connecting to {arm_name} UR robot =====")
            rtde_r = RTDEReceiveInterface(robot_ip)

            joint_positions = rtde_r.getActualQ()
            if joint_positions is not None and len(joint_positions) == 6:
                formatted_joints = [round(j, 4) for j in joint_positions]
                logger.info(f"[ROBOT] Current {arm_name} UR joint positions: {formatted_joints}")
                logger.info(f"===== [ROBOT] {arm_name} UR connected successfully =====\n")
            else:
                logger.info(f"===== [ERROR] Failed to read {arm_name} UR joint positions. Check connection or remote control mode =====")

        except Exception as e:
            logger.info(f"===== [ERROR] Failed to connect to {arm_name} UR robot =====")
            logger.info(f"Exception: {e}\n")

        return rtde_r
    
    def _start_qpos_update(self):
        """
        启动关节位置更新线程。
        在后台持续运行，按照设定的帧率从XR设备读取数据并更新机器人目标关节位置。
        """
        while not self._stop_event.is_set():
            try:
                start = time.perf_counter()
                self._update_robot_qpos_from_xr()
                elapsed = time.perf_counter() - start
                time.sleep(max(0, 1/self.cfg.fps - elapsed))
            except Exception as e:
                logger.error(f"Error in qpos update thread: {e}")

    def _start_placo_visualizer(self):
        """
        启动Placo可视化工具。
        初始化Meshcat可视化器，并在浏览器中自动打开可视化窗口。
        同时显示机器人模型和末端执行器目标帧。
        """
        left_qpos_init = np.array(self._arm["left_rtde_r"].getActualQ())
        right_qpos_init = np.array(self._arm["right_rtde_r"].getActualQ())

        self.placo_robot.update_kinematics()
        self.placo_vis = robot_viz(self.placo_robot)
        # 7 (base) + 6 (left) + 6 (right)
        self.placo_robot.state.q[7:13] = left_qpos_init
        self.placo_robot.state.q[13:19] = right_qpos_init

        # Automatically open browser window
        time.sleep(0.5)  # Small delay to ensure server is ready
        meshcat_url = self.placo_vis.viewer.url()
        print(f"Automatically opening meshcat at: {meshcat_url}")
        webbrowser.open(meshcat_url)

        self.placo_vis.display(self.placo_robot.state.q)
        for name, config in self.manipulator_config.items():
            robot_frame_viz(self.placo_robot, config["link_name"])
            frame_viz(
                f"vis_target_{name}",
                self.effector_task[name].T_world_frame,
            )

    def _get_base_to_world_transform(self, base_link_name: str = "base_link"):
        """
        获取从指定基座链接到世界坐标系的变换矩阵。
        
        Args:
            base_link_name: 基座链接名称，默认为"base_link"
            
        Returns:
            4x4变换矩阵，如果获取失败则返回None
        """
        try:
            T_world_base = self.placo_robot.get_T_world_frame(base_link_name)
            return T_world_base
        except Exception as e:
            logger.error(f"Failed to get transformation for {base_link_name}: {e}")
            return None
        
    def _reset_ee_pose(self):
        """
        重置末端执行器的增量位姿。
        将左右手臂的位置增量和旋转增量都重置为零。
        """
        self.delta_xyz_left_base = np.zeros(3)
        self.delta_rot_angle_axis_left_base = np.array([0.0, 0.0, 0.0])
        self.delta_xyz_right_base = np.zeros(3)
        self.delta_rot_angle_axis_right_base = np.array([0.0, 0.0, 0.0])

    def _update_arm_deltas(self, arm_name, target_xyz, target_quat):
        """
        更新机械臂的位姿增量。
        计算当前目标位姿与上一次目标位姿之间的差值，并转换到基座坐标系下。
        
        Args:
            arm_name: 机械臂名称（"left_arm"或"right_arm"）
            target_xyz: 目标位置
            target_quat: 目标四元数姿态
        """
        delta_xyz = target_xyz - self.last_target_ee[arm_name][:3]
        delta_rot_angle_axis = quat_diff_as_angle_axis(self.last_target_ee[arm_name][3:], target_quat)
        self.last_target_ee[arm_name] = np.concatenate([target_xyz, target_quat])

        base_name = self.manipulator_config[arm_name]["base_link"]
        delta_xyz_attr = f"delta_xyz_{arm_name.split('_')[0]}_base"
        delta_rot_attr = f"delta_rot_angle_axis_{arm_name.split('_')[0]}_base"

        T_world_base = self._get_base_to_world_transform(base_name)
        R_inv = T_world_base[:3, :3].T

        setattr(self, delta_xyz_attr, R_inv @ delta_xyz)
        setattr(self, delta_rot_attr, R_inv @ delta_rot_angle_axis)

    def _update_robot_qpos_from_xr(self):
        """
        从XR控制器更新机器人关节位置。
        读取当前机器人实际关节角度，根据XR控制器输入计算目标末端位姿，
        通过逆运动学求解器计算目标关节角度，并更新夹爪状态。
        """
        current_q_left_actual = self._arm["left_rtde_r"].getActualQ()
        current_q_right_actual = self._arm["right_rtde_r"].getActualQ()
        self.placo_robot.state.q[7:13] = np.array(current_q_left_actual)
        self.placo_robot.state.q[13:19] = np.array(current_q_right_actual)

        self.placo_robot.update_kinematics()
        for arm_name, config in self.manipulator_config.items():
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            active = xr_grip_val > (1.0 - CONTROLLER_DEADZONE)
            trigger_val = self.xr_client.get_key_value_by_name(config["gripper_trigger"])
            if self.cfg.trigger_reverse:
                trigger_val = self.cfg.open_position - trigger_val
            if trigger_val < self.cfg.trigger_threshold:
                trigger_val = self.cfg.close_position
            else:
                trigger_val = self.cfg.open_position

            last_attr = ARM_MAP[arm_name]["last"]
            pos_attr = ARM_MAP[arm_name]["pos"]

            last_trigger = getattr(self, last_attr)
            gripper_pos = getattr(self, pos_attr)

            if last_trigger == 1 and trigger_val == 0:
                gripper_pos = (
                    self.cfg.close_position
                    if gripper_pos == self.cfg.open_position
                    else self.cfg.open_position
                )

            setattr(self, pos_attr, gripper_pos)
            setattr(self, last_attr, trigger_val)

            if active:
                if self.init_ee_xyz[arm_name] is None:
                    # First activation: store current EE pose as initial
                    # Get current EE pose from Placo model based on actual joint angles
                    T_world_ee_current = self.placo_robot.get_T_world_frame(config["link_name"])
                    self.init_ee_xyz[arm_name] = T_world_ee_current[:3, 3].copy()
                    self.init_ee_quat[arm_name] = tf.quaternion_from_matrix(T_world_ee_current)
                    # print(
                    #     f"{arm_name} activated. Current EE xyz: {self.init_ee_xyz[arm_name]}, quat: {self.init_ee_quat[arm_name]}."
                    # )

                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                delta_xyz, delta_rot_angle_axis = self._process_xr_pose(xr_pose, arm_name)
                target_xyz, target_quat = apply_delta_pose(
                    self.init_ee_xyz[arm_name],
                    self.init_ee_quat[arm_name],
                    delta_xyz,
                    delta_rot_angle_axis,
                )

                target_transform = tf.quaternion_matrix(target_quat)
                target_transform[:3, 3] = target_xyz
                self.effector_task[arm_name].T_world_frame = target_transform

                if self.last_target_ee[arm_name] is None:
                    self.last_target_ee[arm_name] = np.concatenate([target_xyz, target_quat])

                if arm_name in ["left_arm", "right_arm"]:
                    self._update_arm_deltas(arm_name, target_xyz, target_quat)

            else:  # Not active
                if self.init_ee_xyz[arm_name] is not None:
                    # print(f"{arm_name} deactivated.")
                    self.init_ee_xyz[arm_name] = None
                    self.init_ee_quat[arm_name] = None
                    self.init_controller_xyz[arm_name] = None
                    self.last_target_ee[arm_name] = None
                    self.init_controller_quat[arm_name] = None
                    self._reset_ee_pose()
                    T_world_ee_current = self.placo_robot.get_T_world_frame(config["link_name"])
                    self.effector_task[arm_name].T_world_frame = T_world_ee_current

        try:
            self.solver.solve(True)

            self.target_left_q = self.placo_robot.state.q[7:13].copy()
            self.target_right_q = self.placo_robot.state.q[13:19].copy()
            if self.cfg.visualize_placo and hasattr(self, "placo_vis"):
                self.placo_vis.display(self.placo_robot.state.q)
                for name, config in self.manipulator_config.items():
                    robot_frame_viz(self.placo_robot, config["link_name"])
                    frame_viz(
                        f"vis_target_{name}",
                        self.effector_task[name].T_world_frame,
                    )

        except RuntimeError as e:
            print(f"IK solver failed: {e}. Returning last known good joint positions.")
        except Exception as e:
            print(f"An unexpected error occurred in IK: {e}. Returning last known good joint positions.")

    def reset_placo_effector(self):
        """
        重置Placo末端执行器任务。
        将所有末端执行器任务的目标位姿设置为当前实际位姿。
        """
        for name, config in self.manipulator_config.items():
            T_world_ee_current = self.placo_robot.get_T_world_frame(config["link_name"])
            self.effector_task[name].T_world_frame = T_world_ee_current
            
    def _process_xr_pose(self, xr_pose, arm_name: str):
        """
        处理XR控制器的当前位姿。
        将XR控制器的位姿转换到世界坐标系，并计算相对于初始位姿的增量。
        
        Args:
            xr_pose: XR控制器位姿 [tx, ty, tz, qx, qy, qz, qw]
            arm_name: 机械臂名称
            
        Returns:
            delta_xyz: 位置增量
            delta_rot: 旋转增量（角度轴表示）
        """
        # xr_pose is typically [tx, ty, tz, qx, qy, qz, qw]
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
        controller_quat = np.array(
            [
                xr_pose[6],  # w
                xr_pose[3],  # x
                xr_pose[4],  # y
                xr_pose[5],  # z
            ]
        )
        controller_xyz = self.R_headset_world @ controller_xyz
        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)
        controller_quat = tf.quaternion_multiply(
            tf.quaternion_multiply(R_quat, controller_quat),
            tf.quaternion_conjugate(R_quat),
        )

        if self.init_controller_xyz[arm_name] is None:
            self.init_controller_xyz[arm_name] = controller_xyz.copy()
            self.init_controller_quat[arm_name] = controller_quat.copy()
            delta_xyz = np.zeros(3)
            delta_rot = np.array([0.0, 0.0, 0.0])  # Angle-axis
        else:
            delta_xyz = (controller_xyz - self.init_controller_xyz[arm_name]) * self.cfg.scale_factor
            delta_rot = quat_diff_as_angle_axis(self.init_controller_quat[arm_name], controller_quat)
        return delta_xyz, delta_rot
    
    def calibrate(self) -> None:
        """校准方法，当前未实现。"""
        pass

    def configure(self):
        """配置方法，当前未实现。"""
        pass

    def get_action(self) -> dict[str, Any]:
        """
        获取当前动作字典。
        返回左右手臂的目标关节角度、末端执行器位姿增量和夹爪位置。
        
        Returns:
            dict: 包含所有动作数据的字典，键格式为"{left/right}_joint_{i}.pos"、
                  "{left/right}_ee_delta_{x/y/z}"、"{left/right}_ee_delta_{rx/ry/rz}"、
                  "{left/right}_gripper_position"
        """
        action = {}
        for name in ["left", "right"]:
            for i in range(6):
                action[f"{name}_joint_{i+1}.pos"] = self.target_left_q[i] if name == "left" else self.target_right_q[i]
            for i, axis in enumerate(["x", "y", "z"]):
                action[f"{name}_ee_delta_{axis}"] = self.delta_xyz_left_base[i] if name == "left" else self.delta_xyz_right_base[i]
            for i, axis in enumerate(["rx", "ry", "rz"]):
                action[f"{name}_ee_delta_{axis}"] = self.delta_rot_angle_axis_left_base[i] if name == "left" else self.delta_rot_angle_axis_right_base[i]
            action[f"{name}_gripper_position"] = self.left_gripper_pos if name == "left" else self.right_gripper_pos
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """发送反馈方法，当前未实现。"""
        pass

    def disconnect(self) -> None:
        """
        断开与机器人和VR设备的连接。
        断开UR机器人的RTDE连接，并停止所有后台线程。
        """
        if not self.is_connected:
            return
        
        # Disconnect UR robots
        self._arm["left_rtde_r"].disconnect()
        self._arm["right_rtde_r"].disconnect()

        # Disconnect all threads
        self._stop_event.set()

        logger.info(f"[INFO] ===== All {self.name} connections have been closed =====")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    class RecordConfig:
        """记录配置类，用于解析和存储配置文件中的参数。"""
        
        def __init__(self, cfg: Dict[str, Any]):
            """
            初始化记录配置。
            
            Args:
                cfg: 配置字典，包含遥操作、机器人、Placo和夹爪的配置信息
            """
            teleop = cfg["teleop"]
            robot_cfg = teleop["robot"]
            placo_cfg = teleop["placo"]
            gripper_cfg = teleop["gripper"]            
            
            # global config
            self.fps = cfg["fps"]
            self.scale_factor = teleop["scale_factor"]
            self.R_headset_world = teleop["R_headset_world"]
            # robot config
            self.left_robot_ip = robot_cfg["left_robot_ip"]
            self.right_robot_ip = robot_cfg["right_robot_ip"]
            # placo config
            self.robot_urdf_path = placo_cfg["robot_urdf_path"]
            self.visualize_placo = placo_cfg.get("visualize_placo", False)
            self.servo_time = placo_cfg.get("servo_time", 0.017)
            # gripper config
            self.trigger_reverse = gripper_cfg["trigger_reverse"]
            self.trigger_threshold = gripper_cfg["trigger_threshold"]
            self.close_position = gripper_cfg["close_position"]
            self.open_position = gripper_cfg["open_position"]

    with open(Path(__file__).parent / "config" / "cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    record_cfg = RecordConfig(cfg["record"])
    xr_client = XrClient()
    teleop_config = VRTeleopConfig(
        left_robot_ip=record_cfg.left_robot_ip,
        right_robot_ip=record_cfg.right_robot_ip,
        xr_client=xr_client,
        trigger_reverse=record_cfg.trigger_reverse,
        trigger_threshold=record_cfg.trigger_threshold,
        close_position=record_cfg.close_position,
        open_position=record_cfg.open_position,
        servo_time=record_cfg.servo_time,
        fps=record_cfg.fps,
        scale_factor=record_cfg.scale_factor,
        R_headset_world=record_cfg.R_headset_world,
        robot_urdf_path=record_cfg.robot_urdf_path,
        visualize_placo=record_cfg.visualize_placo,
        control_mode="vrteleop",
    )
    teleop = VRTeleop(teleop_config)
    teleop.connect()

    try:
        while True:
            print(teleop.get_action())
            time.sleep(0.01)
    except KeyboardInterrupt:
        teleop.disconnect()