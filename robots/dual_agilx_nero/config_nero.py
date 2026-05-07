"""
Configuration for Nero dual-arm robot system.
Each arm has 7 DOF
"""
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("nero_dual_arm")
@dataclass
class NeroDualArmConfig(RobotConfig):
    """Configuration for Nero dual-arm robot with agx_grippers."""
    
    # Robot identification
    name: str = "nero_dual_arm"
    
    # Network configuration - single port for dual-arm control
    robot_ip: str = "192.168.110.114"  # Nero server ip
    robot_port: int = 4242  # dual-arm zerorpc port (single port for both arms)
    
    # Gripper configuration (agx_grippers)
    gripper_ip: str = "192.168.110.114"  # gripper zerorpc ip, if different from robot_ip, set to robot_ip
    gripper_port: int = 4243  # gripper zerorpc port (single port for both arms)
    use_gripper: bool = True
    gripper_max_open: float = 0.1  # agx_gripper max opening: 10mm
    gripper_force: float = 2.0  # Gripping force in N
    gripper_speed: float = 0.1  # Speed in m/s
    gripper_reverse: bool = False  # Whether to reverse gripper command
    close_threshold: float = 0.05  # Threshold for binary gripper control
    
    # Control configuration
    control_mode: str = "oculus"
    debug: bool = True
    # Execution-side interpretation of cartesian action values.
    # `step_wise`: actions are delta ee poses -> call servo_p_OL(..., delta=True)
    # `chunk_wise`: ACT inference already decoded actions to absolute targets; execution converts each target
    # to a one-shot delta against the current reference pose, then calls servo_p_OL(..., delta=True)
    action_delta_alignment: Literal["step_wise", "chunk_wise"] = "step_wise"
    # Reference pose source used only by chunk-wise execution-side absolute->delta conversion.
    # `servo_ol`: query the server's own servo_p_OL(delta=True) reference pose, i.e. IK-state q_prev FK.
    # `direct_ee`: query the robot TCP pose directly through *_robot_get_ee_pose().
    # `observation`: use the most recent local observation cache when it matches the active controller semantics.
    chunkwise_reference_pose_source: Literal["servo_ol", "direct_ee", "observation"] = "servo_ol"
    # Legacy Nero datasets keep the `left/right_ee_pose.rx/ry/rz` feature names, but the values recorded in
    # `observation.state` follow this compatibility axis order. Chunk-wise ACT deployment uses this hint when
    # interpreting the current absolute ee pose as the chunk reference pose.
    ee_pose_observation_axis_order: list[str] = field(
        default_factory=lambda: ["x", "y", "z", "rz", "ry", "rx"]
    )
    
    # Joint configuration (7 DOF per arm)
    num_joints_per_arm: int = 7
    
    # Cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    
    # Safety limits
    max_joint_velocity: float = 2.0  # rad/s
    max_ee_velocity: float = 0.5  # m/s
    max_joint_delta: float = 0.3  # rad - max joint change per step
