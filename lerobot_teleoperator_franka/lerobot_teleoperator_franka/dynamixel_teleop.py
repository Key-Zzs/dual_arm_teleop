#!/usr/bin/env python

"""
Dynamixel-based isomorphic teleoperation implementation.
"""

import logging
from typing import Any, Dict

from .base_teleop import BaseTeleop
from .config_teleop import DynamixelTeleopConfig
from .dynamixel.dynamixel_robot import DynamixelRobot

logger = logging.getLogger(__name__)


class DynamixelTeleop(BaseTeleop):
    """
    Isomorphic teleoperation using Dynamixel servos.
    
    This teleoperation mode uses a master arm (Dynamixel servos) to control
    a slave robot arm. The joint positions are directly mapped from master to slave.
    """
    
    config_class = DynamixelTeleopConfig
    name = "IsoTeleop"
    
    def __init__(self, config: DynamixelTeleopConfig):
        super().__init__(config)
        self.dynamixel_robot: DynamixelRobot = None
    
    def _get_teleop_name(self) -> str:
        return "IsoTeleop"
    
    @property
    def action_features(self) -> dict:
        """Return action features for isoteleop mode (joint positions)."""
        features = {}
        for i in range(7):
            features[f"joint_{i+1}.pos"] = float
        features["gripper_position"] = float
        return features
    
    def _connect_impl(self) -> None:
        """Connect to Dynamixel robot."""
        self.dynamixel_robot = DynamixelRobot(
            hardware_offsets=self.cfg.hardware_offsets,
            joint_ids=self.cfg.joint_ids,
            joint_offsets=self.cfg.joint_offsets,
            joint_signs=self.cfg.joint_signs,
            port=self.cfg.port,
            use_gripper=self.cfg.use_gripper,
            gripper_config=self.cfg.gripper_config,
            real=True
        )
        joint_positions = self.dynamixel_robot.get_joint_state()
        formatted_joints = [round(float(j), 4) for j in joint_positions]
        logger.info(f"[TELEOP] Current joint positions: {formatted_joints}")
    
    def _disconnect_impl(self) -> None:
        """Disconnect from Dynamixel robot."""
        if self.dynamixel_robot is not None:
            self.dynamixel_robot._driver.close()
    
    def _get_action_impl(self) -> Dict[str, Any]:
        """Get joint positions from Dynamixel robot."""
        return self.dynamixel_robot.get_observations()
