#!/usr/bin/env python

"""
Oculus Quest dual-arm teleoperation implementation.
Uses both Oculus controllers to control a dual-arm robot system.
Left controller -> Left arm, Right controller -> Right arm.
Integrates IK solver for joint position tracking and robot client for state feedback.
"""

import logging
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional

from .base_teleop import BaseTeleop
from .config_teleop import OculusDualArmTeleopConfig
from .oculus.oculus_dual_arm_robot import OculusDualArmRobot
from .ik_solver import DualArmIKSolver

logger = logging.getLogger(__name__)

# Import robot client (try multiple paths)
try:
    # Try relative import from lerobot_robot package
    from lerobot_robot.dobot_interface_client import DobotDualArmClient
    logger.debug("[TELEOP] Imported DobotDualArmClient from lerobot_robot package")
except ImportError:
    try:
        # Try direct import if in path
        from dobot_interface_client import DobotDualArmClient
        logger.debug("[TELEOP] Imported DobotDualArmClient directly")
    except ImportError:
        # Fallback: add path and import
        import sys
        robot_path = Path(__file__).parent.parent.parent / "lerobot_robot/lerobot_robot"
        if robot_path.exists():
            sys.path.insert(0, str(robot_path))
            from dobot_interface_client import DobotDualArmClient
            logger.debug(f"[TELEOP] Imported DobotDualArmClient from {robot_path}")
        else:
            logger.warning(f"[TELEOP] Robot client path not found: {robot_path}")
            DobotDualArmClient = None


class OculusDualArmTeleop(BaseTeleop):
    """
    Dual-arm teleoperation using both Oculus Quest controllers.
    
    This teleoperation mode uses both Oculus Quest controllers to simultaneously
    control two robot arms in Cartesian space. The output includes both delta pose
    and joint positions (via IK solver).
    
    Automatically connects to robot to get current state for IK computation.
    
    Controls:
    - LG (Left Grip):    Must be pressed to enable left arm action recording
    - RG (Right Grip):   Must be pressed to enable right arm action recording
    - LTr (Left Trigger):  Controls left gripper  (0.0 = open, 1.0 = closed)
    - RTr (Right Trigger): Controls right gripper (0.0 = open, 1.0 = closed)
    - Left controller pose:  Controls left arm end-effector delta pose
    - Right controller pose: Controls right arm end-effector delta pose
    - A button: Request robot reset
    """
    
    config_class = OculusDualArmTeleopConfig
    name = "OculusDualArmTeleop"
    
    def __init__(self, config: OculusDualArmTeleopConfig):
        super().__init__(config)
        self.oculus_robot: OculusDualArmRobot = None
        self.ik_solver: Optional[DualArmIKSolver] = None
        self.robot_client: Optional[DobotDualArmClient] = None
        
        # Current robot state (for IK)
        self.left_current_pose: Optional[np.ndarray] = None
        self.right_current_pose: Optional[np.ndarray] = None
        self.left_current_joints: np.ndarray = np.zeros(6)
        self.right_current_joints: np.ndarray = np.zeros(6)
    
    def _get_teleop_name(self) -> str:
        return "OculusDualArmTeleop"
    
    @property
    def action_features(self) -> dict:
        """Return action features for dual-arm oculus mode (includes both EE pose and joints)."""
        features = {}
        # Left arm delta pose
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"left_delta_ee_pose.{axis}"] = float
        # Right arm delta pose
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"right_delta_ee_pose.{axis}"] = float
        
        # Left arm joint positions (from IK)
        for i in range(6):
            features[f"left_joint_{i+1}.pos"] = float
        # Right arm joint positions (from IK)
        for i in range(6):
            features[f"right_joint_{i+1}.pos"] = float
        
        # Gripper commands
        if self.cfg.use_gripper:
            features["left_gripper_cmd_bin"] = float
            features["right_gripper_cmd_bin"] = float
        return features
    
    def _connect_impl(self) -> None:
        """Connect to Oculus Quest for dual-arm control and initialize IK solver and robot client."""
        self.oculus_robot = OculusDualArmRobot(
            ip=self.cfg.ip,
            use_gripper=self.cfg.use_gripper,
            left_pose_scaler=self.cfg.left_pose_scaler,
            left_channel_signs=self.cfg.left_channel_signs,
            right_pose_scaler=self.cfg.right_pose_scaler,
            right_channel_signs=self.cfg.right_channel_signs,
        )
        logger.info(f"[TELEOP] Oculus dual-arm connected at IP: {self.cfg.ip}")
        logger.info(f"[TELEOP]   Left arm  - pose_scaler: {self.cfg.left_pose_scaler}, "
                    f"channel_signs: {self.cfg.left_channel_signs}")
        logger.info(f"[TELEOP]   Right arm - pose_scaler: {self.cfg.right_pose_scaler}, "
                    f"channel_signs: {self.cfg.right_channel_signs}")
        
        # Initialize robot client for state feedback
        try:
            # Get robot IP and port from config (with defaults)
            robot_ip = getattr(self.cfg, 'robot_ip', '127.0.0.1')
            robot_port = getattr(self.cfg, 'robot_port', 4242)
            
            self.robot_client = DobotDualArmClient(ip=robot_ip, port=robot_port)
            logger.info(f"[TELEOP] Robot client connected at {robot_ip}:{robot_port}")
        except Exception as e:
            logger.warning(f"[TELEOP] Failed to connect to robot client: {e}")
            self.robot_client = None
        
        # Initialize IK solver
        try:
            urdf_path = Path(__file__).parent.parent.parent / "assets/dobot_description/urdf/dual_nova5_robot.urdf"
            if urdf_path.exists():
                self.ik_solver = DualArmIKSolver(
                    urdf_path=str(urdf_path),
                    left_arm_prefix="left_",
                    right_arm_prefix="right_",
                    left_ee_link="left_Link6",
                    right_ee_link="right_Link6",
                )
                logger.info(f"[TELEOP] IK solver initialized with URDF: {urdf_path}")
            else:
                logger.warning(f"[TELEOP] URDF not found at {urdf_path}, IK solver disabled")
                self.ik_solver = None
        except Exception as e:
            logger.warning(f"[TELEOP] Failed to initialize IK solver: {e}")
            self.ik_solver = None
    
    def _disconnect_impl(self) -> None:
        """Disconnect from Oculus Quest and robot client."""
        if self.robot_client is not None:
            try:
                self.robot_client.close()
                logger.info("[TELEOP] Robot client disconnected")
            except:
                pass
    
    def update_robot_state(
        self,
        left_ee_pose: Optional[np.ndarray] = None,
        right_ee_pose: Optional[np.ndarray] = None,
        left_joints: Optional[np.ndarray] = None,
        right_joints: Optional[np.ndarray] = None,
    ):
        """
        Update current robot state for IK computation.
        This should be called by the robot with its current state.
        
        Args:
            left_ee_pose: Current left arm end-effector pose [x, y, z, rx, ry, rz]
            right_ee_pose: Current right arm end-effector pose [x, y, z, rx, ry, rz]
            left_joints: Current left arm joint positions (6 values)
            right_joints: Current right arm joint positions (6 values)
        """
        if left_ee_pose is not None:
            self.left_current_pose = left_ee_pose
        if right_ee_pose is not None:
            self.right_current_pose = right_ee_pose
        if left_joints is not None:
            self.left_current_joints = left_joints
            if self.ik_solver is not None:
                self.ik_solver.update_joint_positions(left_joints=left_joints)
        if right_joints is not None:
            self.right_current_joints = right_joints
            if self.ik_solver is not None:
                self.ik_solver.update_joint_positions(right_joints=right_joints)
    
    def _get_action_impl(self) -> Dict[str, Any]:
        """
        Get delta pose from both Oculus controllers and compute joint positions via IK.
        
        Automatically fetches current robot state from robot_client if available.
        Uses Dobot's built-in IK solver for accurate joint computation.
        
        Returns dict with:
            - left_delta_ee_pose.{x,y,z,rx,ry,rz}
            - right_delta_ee_pose.{x,y,z,rx,ry,rz}
            - left_joint_{1-6}.pos (IK computed, in radians)
            - right_joint_{1-6}.pos (IK computed, in radians)
            - left_gripper_cmd_bin
            - right_gripper_cmd_bin
            - reset_requested
        """
        # Automatically update robot state from client if available
        if self.robot_client is not None:
            try:
                # Get current EE poses (in meters and radians)
                left_ee = self.robot_client.left_robot_get_ee_pose()
                right_ee = self.robot_client.right_robot_get_ee_pose()
                
                # Get current joint positions (in radians)
                left_joints = self.robot_client.left_robot_get_joint_positions()
                right_joints = self.robot_client.right_robot_get_joint_positions()
                
                # Update internal state
                self.left_current_pose = left_ee
                self.right_current_pose = right_ee
                self.left_current_joints = left_joints
                self.right_current_joints = right_joints
                    
            except Exception as e:
                logger.warning(f"[TELEOP] Failed to get robot state: {e}")
        
        # Get delta pose from Oculus
        oculus_obs = self.oculus_robot.get_observations()
        
        # Extract delta poses
        left_delta = np.array([
            oculus_obs[f"left_delta_ee_pose.{axis}"] 
            for axis in ["x", "y", "z", "rx", "ry", "rz"]
        ])
        right_delta = np.array([
            oculus_obs[f"right_delta_ee_pose.{axis}"] 
            for axis in ["x", "y", "z", "rx", "ry", "rz"]
        ])
        
        # Build action dict
        action = {}
        
        # Add delta EE poses
        for i, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
            action[f"left_delta_ee_pose.{axis}"] = float(left_delta[i])
            action[f"right_delta_ee_pose.{axis}"] = float(right_delta[i])
        
        # Compute joint positions via IK using Dobot's built-in IK
        if self.robot_client is not None and self.left_current_pose is not None:
            try:
                # Compute target poses with proper rotation handling
                # Position: can be added directly
                left_target_pos = self.left_current_pose[:3] + left_delta[:3]
                right_target_pos = self.right_current_pose[:3] + right_delta[:3]
                
                # Orientation: need to use rotation matrices
                from scipy.spatial.transform import Rotation
                
                # Current orientation as rotation matrix
                R_current_left = Rotation.from_euler("xyz", self.left_current_pose[3:])
                R_current_right = Rotation.from_euler("xyz", self.right_current_pose[3:])
                
                # Delta orientation as rotation matrix
                R_delta_left = Rotation.from_euler("xyz", left_delta[3:])
                R_delta_right = Rotation.from_euler("xyz", right_delta[3:])
                
                # Target orientation: R_target = R_delta * R_current
                R_target_left = R_delta_left * R_current_left
                R_target_right = R_delta_right * R_current_right
                
                # Convert back to Euler angles
                left_target_rot = R_target_left.as_euler("xyz")
                right_target_rot = R_target_right.as_euler("xyz")
                
                # Combine position and orientation (in meters and radians)
                left_target_pose = np.concatenate([left_target_pos, left_target_rot])
                right_target_pose = np.concatenate([right_target_pos, right_target_rot])
                
                # Solve IK using Dobot controller (expects meters and radians)
                left_joints_result = self.robot_client.inverse_kinematics(
                    'left', left_target_pose, self.left_current_joints
                )
                right_joints_result = self.robot_client.inverse_kinematics(
                    'right', right_target_pose, self.right_current_joints
                )
                
                # IK returns radians
                if left_joints_result is not None:
                    left_target_joints = np.array(left_joints_result)
                else:
                    logger.warning("[TELEOP] Left arm IK failed, using current joints")
                    left_target_joints = self.left_current_joints
                
                if right_joints_result is not None:
                    right_target_joints = np.array(right_joints_result)
                else:
                    logger.warning("[TELEOP] Right arm IK failed, using current joints")
                    right_target_joints = self.right_current_joints
                
                # Add joint positions to action (in radians)
                for i in range(6):
                    action[f"left_joint_{i+1}.pos"] = float(left_target_joints[i])
                    action[f"right_joint_{i+1}.pos"] = float(right_target_joints[i])
                
            except Exception as e:
                logger.warning(f"[TELEOP] IK computation failed: {e}")
                # Use current joints as fallback
                for i in range(6):
                    action[f"left_joint_{i+1}.pos"] = float(self.left_current_joints[i])
                    action[f"right_joint_{i+1}.pos"] = float(self.right_current_joints[i])
        else:
            # No robot client, use zeros
            for i in range(6):
                action[f"left_joint_{i+1}.pos"] = 0.0
                action[f"right_joint_{i+1}.pos"] = 0.0
        
        # Add gripper commands
        if self.cfg.use_gripper:
            action["left_gripper_cmd_bin"] = oculus_obs.get("left_gripper_cmd_bin", 0.0)
            action["right_gripper_cmd_bin"] = oculus_obs.get("right_gripper_cmd_bin", 0.0)
        
        # Add reset flag
        action["reset_requested"] = oculus_obs.get("reset_requested", False)
        
        return action


def main() -> None:
    """Simple CLI test entrypoint for OculusDualArmTeleop output."""
    parser = argparse.ArgumentParser(description="Test OculusDualArmTeleop action output")
    parser.add_argument("--ip", type=str, default="192.168.110.62", help="Oculus Quest IP")
    parser.add_argument("--hz", type=float, default=10.0, help="Print frequency")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N steps; 0 means run forever")
    parser.add_argument("--no-gripper", action="store_true", help="Disable gripper fields")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = OculusDualArmTeleopConfig(
        ip=args.ip,
        use_gripper=not args.no_gripper,
    )
    teleop = OculusDualArmTeleop(cfg)

    sleep_s = 1.0 / max(args.hz, 1e-6)
    step = 0
    try:
        teleop.connect()
        while True:
            action = teleop.get_action()
            print(f"[step={step}] {action}")
            step += 1
            if args.max_steps > 0 and step >= args.max_steps:
                break
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        teleop.disconnect()


if __name__ == "__main__":
    main()


