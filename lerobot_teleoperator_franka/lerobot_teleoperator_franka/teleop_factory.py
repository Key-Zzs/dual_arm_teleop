#!/usr/bin/env python

"""
Factory for creating teleoperation instances.
"""

from typing import Union

from .base_teleop import BaseTeleop
from .config_teleop import (
    BaseTeleopConfig,
    DynamixelTeleopConfig,
    SpacemouseTeleopConfig,
    OculusTeleopConfig,
)
from .dynamixel_teleop import DynamixelTeleop
from .spacemouse_teleop import SpacemouseTeleop
from .oculus_teleop import OculusTeleop


def create_teleop(config: BaseTeleopConfig) -> BaseTeleop:
    """
    Create a teleoperation instance based on the configuration.
    
    Args:
        config: Teleoperation configuration (DynamixelTeleopConfig, SpacemouseTeleopConfig, or OculusTeleopConfig)
    
    Returns:
        A teleoperation instance (DynamixelTeleop, SpacemouseTeleop, or OculusTeleop)
    
    Raises:
        ValueError: If the control mode is not supported
    """
    if isinstance(config, DynamixelTeleopConfig) or config.control_mode == "isoteleop":
        return DynamixelTeleop(config if isinstance(config, DynamixelTeleopConfig) else DynamixelTeleopConfig())
    
    elif isinstance(config, SpacemouseTeleopConfig) or config.control_mode == "spacemouse":
        return SpacemouseTeleop(config if isinstance(config, SpacemouseTeleopConfig) else SpacemouseTeleopConfig())
    
    elif isinstance(config, OculusTeleopConfig) or config.control_mode == "oculus":
        return OculusTeleop(config if isinstance(config, OculusTeleopConfig) else OculusTeleopConfig())
    
    else:
        raise ValueError(f"Unsupported control mode: {config.control_mode}. "
                        f"Supported modes: isoteleop, spacemouse, oculus")


def create_teleop_config(control_mode: str, **kwargs) -> BaseTeleopConfig:
    """
    Create a teleoperation configuration based on the control mode.
    
    Args:
        control_mode: The teleoperation mode ("isoteleop", "spacemouse", or "oculus")
        **kwargs: Configuration parameters specific to each mode
    
    Returns:
        A teleoperation configuration instance
    
    Raises:
        ValueError: If the control mode is not supported
    """
    if control_mode == "isoteleop":
        return DynamixelTeleopConfig(**kwargs)
    elif control_mode == "spacemouse":
        return SpacemouseTeleopConfig(**kwargs)
    elif control_mode == "oculus":
        return OculusTeleopConfig(**kwargs)
    else:
        raise ValueError(f"Unsupported control mode: {control_mode}. "
                        f"Supported modes: isoteleop, spacemouse, oculus")


# Convenience function to get action features for a control mode
def get_action_features(control_mode: str, use_gripper: bool = True) -> dict:
    """
    Get the action features for a given control mode.
    
    Args:
        control_mode: The teleoperation mode ("isoteleop", "spacemouse", or "oculus")
        use_gripper: Whether gripper is used
    
    Returns:
        Dictionary of action features
    """
    if control_mode == "isoteleop":
        features = {}
        for i in range(7):
            features[f"joint_{i+1}.pos"] = float
        if use_gripper:
            features["gripper_position"] = float
        return features
    
    elif control_mode in ["spacemouse", "oculus"]:
        features = {}
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"delta_ee_pose.{axis}"] = float
        if use_gripper:
            features["gripper_cmd_bin"] = float
        return features
    
    else:
        raise ValueError(f"Unsupported control mode: {control_mode}")
