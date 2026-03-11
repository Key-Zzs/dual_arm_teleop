#!/usr/bin/env python
"""
Placo visualization utilities using meshcat.
Based on VRTeleop implementation.
"""

import numpy as np
import webbrowser
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False
    logger.warning("[VIS] meshcat not available, visualization disabled")


def create_coordinate_frame(length: float = 0.1, radius: float = 0.005):
    """Create a coordinate frame with RGB axes (X=Red, Y=Green, Z=Blue)."""
    # Create cylinders for each axis
    frame = {}
    
    # X axis (Red)
    frame["x"] = g.Cylinder(length, radius)
    # Y axis (Green)
    frame["y"] = g.Cylinder(length, radius)
    # Z axis (Blue)
    frame["z"] = g.Cylinder(length, radius)
    
    return frame


class PlacoVisualizer:
    """
    Meshcat-based visualizer for placo robot with full robot model.
    """
    
    def __init__(self, robot, auto_open: bool = True):
        """
        Initialize the visualizer.
        
        Args:
            robot: placo.RobotWrapper instance
            auto_open: Whether to automatically open browser
        """
        if not MESHCAT_AVAILABLE:
            raise RuntimeError("meshcat is required for visualization")
        
        self.robot = robot
        self.vis = meshcat.Visualizer()
        self.auto_open = auto_open
        
        # Define link names for visualization
        self.left_arm_links = [
            "left_base_link",
            "left_Link1", "left_Link2", "left_Link3", 
            "left_Link4", "left_Link5", "left_Link6"
        ]
        self.right_arm_links = [
            "right_base_link",
            "right_Link1", "right_Link2", "right_Link3",
            "right_Link4", "right_Link5", "right_Link6"
        ]
        
        # Initialize visualization
        self._setup_visualization()
        
        if auto_open:
            self.open_browser()
    
    def _setup_visualization(self):
        """Set up the meshcat visualization with full robot model."""
        # Create base platform
        self.vis["robot/base"].set_object(
            g.Box([0.3, 0.3, 0.05]),
            g.MeshLambertMaterial(color=0x808080)
        )
        
        # Create left arm links
        for i, link_name in enumerate(self.left_arm_links):
            # Use different colors for each link
            color = 0x4A90D9 if i % 2 == 0 else 0x6BA3E0
            size = 0.04 if i == 0 else 0.03
            
            self.vis[f"robot/left_arm/{link_name}"].set_object(
                g.Box([size, size, size * 2]),
                g.MeshLambertMaterial(color=color)
            )
        
        # Create right arm links
        for i, link_name in enumerate(self.right_arm_links):
            # Use different colors for each link
            color = 0xD94A4A if i % 2 == 0 else 0xE06B6B
            size = 0.04 if i == 0 else 0.03
            
            self.vis[f"robot/right_arm/{link_name}"].set_object(
                g.Box([size, size, size * 2]),
                g.MeshLambertMaterial(color=color)
            )
        
        # Create end-effector frames (larger, more visible)
        self.vis["ee/left"].set_object(
            g.Box([0.025, 0.025, 0.025]),
            g.MeshLambertMaterial(color=0x00FF00)
        )
        self.vis["ee/right"].set_object(
            g.Box([0.025, 0.025, 0.025]),
            g.MeshLambertMaterial(color=0x00FF00)
        )
        
        # Create target frames (transparent)
        self.vis["target/left"].set_object(
            g.Box([0.03, 0.03, 0.03]),
            g.MeshLambertMaterial(color=0xFF0000, opacity=0.5)
        )
        self.vis["target/right"].set_object(
            g.Box([0.03, 0.03, 0.03]),
            g.MeshLambertMaterial(color=0xFF0000, opacity=0.5)
        )
        
        # Create coordinate frame at world origin
        self._create_world_frame()
    
    def _create_world_frame(self):
        """Create a coordinate frame at world origin."""
        # X axis (Red)
        self.vis["world_frame/x"].set_object(
            g.Cylinder(0.1, 0.005),
            g.MeshLambertMaterial(color=0xFF0000)
        )
        self.vis["world_frame/x"].set_transform(
            tf.rotation_matrix(np.pi/2, [0, 0, 1]) @ tf.translation_matrix([0.05, 0, 0])
        )
        
        # Y axis (Green)
        self.vis["world_frame/y"].set_object(
            g.Cylinder(0.1, 0.005),
            g.MeshLambertMaterial(color=0x00FF00)
        )
        self.vis["world_frame/y"].set_transform(
            tf.rotation_matrix(np.pi/2, [1, 0, 0]) @ tf.translation_matrix([0, 0.05, 0])
        )
        
        # Z axis (Blue)
        self.vis["world_frame/z"].set_object(
            g.Cylinder(0.1, 0.005),
            g.MeshLambertMaterial(color=0x0000FF)
        )
        self.vis["world_frame/z"].set_transform(
            tf.translation_matrix([0, 0, 0.05])
        )
    
    def open_browser(self):
        """Open the meshcat visualization in a browser."""
        time.sleep(0.5)  # Small delay to ensure server is ready
        url = self.vis.url()
        logger.info(f"[VIS] Opening meshcat at: {url}")
        webbrowser.open(url)
    
    def display(self, q: np.ndarray = None):
        """
        Display the robot at a given configuration.
        
        Args:
            q: Joint configuration vector
        """
        if q is not None:
            self.robot.state.q = q
            self.robot.update_kinematics()
        
        # Update all link positions
        try:
            # Update left arm links
            for link_name in self.left_arm_links:
                try:
                    T = self.robot.get_T_world_frame(link_name)
                    self.vis[f"robot/left_arm/{link_name}"].set_transform(T)
                except Exception:
                    pass
            
            # Update right arm links
            for link_name in self.right_arm_links:
                try:
                    T = self.robot.get_T_world_frame(link_name)
                    self.vis[f"robot/right_arm/{link_name}"].set_transform(T)
                except Exception:
                    pass
            
            # Update end-effectors
            T_left = self.robot.get_T_world_frame("left_Link6")
            T_right = self.robot.get_T_world_frame("right_Link6")
            
            self.vis["ee/left"].set_transform(T_left)
            self.vis["ee/right"].set_transform(T_right)
            
        except Exception as e:
            logger.warning(f"[VIS] Failed to update visualization: {e}")
    
    def update_target(self, left_target: np.ndarray = None, right_target: np.ndarray = None):
        """
        Update target frame visualization.
        
        Args:
            left_target: 4x4 transformation matrix for left target
            right_target: 4x4 transformation matrix for right target
        """
        if left_target is not None:
            self.vis["target/left"].set_transform(left_target)
        if right_target is not None:
            self.vis["target/right"].set_transform(right_target)
    
    def url(self) -> str:
        """Get the meshcat server URL."""
        return self.vis.url()


# Keep backward compatibility
def robot_viz(robot, url: str = None):
    """Create a meshcat visualizer for a placo robot."""
    return PlacoVisualizer(robot, auto_open=False)


def frame_viz(name: str, T_world_frame: np.ndarray, length: float = 0.1):
    """Visualize a coordinate frame in meshcat."""
    pass


def robot_frame_viz(robot, frame_name: str):
    """Visualize a robot frame in meshcat."""
    pass
