#!/usr/bin/env python

"""
SpaceMouse teleoperation implementation.
"""

import logging
from typing import Any, Dict

from .base_teleop import BaseTeleop
from .config_teleop import SpacemouseTeleopConfig
from .spacemouse.spacemouse_robot import SpaceMouseRobot

logger = logging.getLogger(__name__)


class SpacemouseTeleop(BaseTeleop):
    """
    Teleoperation using SpaceMouse.
    
    This teleoperation mode uses a SpaceMouse to control the robot's end-effector
    in Cartesian space. The output is delta pose (position and orientation changes).
    """
    
    config_class = SpacemouseTeleopConfig
    name = "SpacemouseTeleop"
    
    def __init__(self, config: SpacemouseTeleopConfig):
        super().__init__(config)
        self.spacemouse_robot: SpaceMouseRobot = None
    
    def _get_teleop_name(self) -> str:
        return "SpacemouseTeleop"
    
    @property
    def action_features(self) -> dict:
        """Return action features for spacemouse mode (delta ee pose)."""
        features = {}
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"delta_ee_pose.{axis}"] = float
        features["gripper_cmd_bin"] = float
        return features
    
    def _connect_impl(self) -> None:
        """Connect to SpaceMouse."""
        self.spacemouse_robot = SpaceMouseRobot(
            use_gripper=self.cfg.use_gripper,
            pose_scaler=self.cfg.pose_scaler,
            channel_signs=self.cfg.channel_signs,
        )
        actions = self.spacemouse_robot.get_action()
        formatted_actions = [round(float(j), 4) for j in actions]
        logger.info(f"[TELEOP] Current ee pose actions: {formatted_actions}")
    
    def _disconnect_impl(self) -> None:
        """Disconnect from SpaceMouse."""
        if self.spacemouse_robot is not None:
            self.spacemouse_robot._expert.close()
    
    def _get_action_impl(self) -> Dict[str, Any]:
        """Get delta pose from SpaceMouse."""
        return self.spacemouse_robot.get_observations()
