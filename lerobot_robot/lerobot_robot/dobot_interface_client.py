'''
Dobot Nova5 dual-arm robot interface client.
Connects to dobot_interface_server via zerorpc.
'''

import logging
import numpy as np
import zerorpc
from typing import Optional, Dict, Any

log = logging.getLogger(__name__)


class DobotDualArmClient:
    """Client for dual-arm Dobot Nova5 robot."""
    
    def __init__(self, ip: str = '127.0.0.1', port: int = 4242):
        self.ip = ip
        self.port = port
        
        try:
            self.server = zerorpc.Client(heartbeat=20)
            self.server.connect(f"tcp://{ip}:{port}")
            log.info(f"[CLIENT] Connected to {ip}:{port}")
        except Exception as e:
            log.error(f"[CLIENT] Connection failed: {e}")
            self.server = None
    
    # ==================== State Query ====================
    
    def left_robot_get_joint_positions(self) -> np.ndarray:
        """Get left arm joint positions (radians)."""
        if self.server is None:
            return np.zeros(6)
        # Server returns radians
        return np.array(self.server.left_robot_get_joint_positions())
    
    def right_robot_get_joint_positions(self) -> np.ndarray:
        """Get right arm joint positions (radians)."""
        if self.server is None:
            return np.zeros(6)
        # Server returns radians
        return np.array(self.server.right_robot_get_joint_positions())
    
    def left_robot_get_ee_pose(self) -> np.ndarray:
        """Get left arm EE pose [x, y, z, rx, ry, rz] (m, radians)."""
        if self.server is None:
            return np.zeros(6)
        # Server returns meters and radians
        return np.array(self.server.left_robot_get_ee_pose())
    
    def right_robot_get_ee_pose(self) -> np.ndarray:
        """Get right arm EE pose [x, y, z, rx, ry, rz] (m, radians)."""
        if self.server is None:
            return np.zeros(6)
        # Server returns meters and radians
        return np.array(self.server.right_robot_get_ee_pose())
    
    # ==================== MoveIt Motion ====================
    
    def left_robot_move_to_joint_positions(self, positions: np.ndarray, delta: bool = False):
        """Move left arm to joint positions (radians)."""
        if self.server is None:
            return
        self.server.left_robot_move_to_joint_positions(positions.tolist(), delta)
    
    def right_robot_move_to_joint_positions(self, positions: np.ndarray, delta: bool = False):
        """Move right arm to joint positions (radians)."""
        if self.server is None:
            return
        self.server.right_robot_move_to_joint_positions(positions.tolist(), delta)
    
    def left_robot_move_to_ee_pose(self, pose: np.ndarray, delta: bool = False):
        """Move left arm to EE pose [x, y, z, rx, ry, rz] (m, radians)."""
        if self.server is None:
            return
        self.server.left_robot_move_to_ee_pose(pose.tolist(), delta)
    
    def right_robot_move_to_ee_pose(self, pose: np.ndarray, delta: bool = False):
        """Move right arm to EE pose [x, y, z, rx, ry, rz] (m, radians)."""
        if self.server is None:
            return
        self.server.right_robot_move_to_ee_pose(pose.tolist(), delta)
    
    def dual_robot_move_to_ee_pose(self, left_pose: np.ndarray, right_pose: np.ndarray, delta: bool = False):
        """Move both arms to EE poses simultaneously."""
        if self.server is None:
            return
        self.server.dual_robot_move_to_ee_pose(left_pose.tolist(), right_pose.tolist(), delta)
    
    # ==================== Go Home ====================
    
    def left_robot_go_home(self):
        """Move left arm to home position."""
        if self.server is None:
            return
        self.server.left_robot_go_home()
    
    def right_robot_go_home(self):
        """Move right arm to home position."""
        if self.server is None:
            return
        self.server.right_robot_go_home()
    
    def robot_go_home(self):
        """Move both arms to home position."""
        if self.server is None:
            return
        self.server.robot_go_home()
    
    # ==================== ServoJ Control (Joint Servo) ====================
    
    def servo_j(self, arm_name: str, joints: np.ndarray, t: float = 0.1,
                lookahead_time: float = 0.05, gain: float = 300) -> bool:
        """
        Send ServoJ with ABSOLUTE joint angles (radians).
        Args:
            joints: Joint angles in RADIANS
        """
        if self.server is None:
            return True
        # Server expects radians
        return self.server.servo_j(arm_name, joints.tolist(), t, lookahead_time, gain)
    
    def servo_j_delta(self, arm_name: str, delta_joints: np.ndarray, t: float = 0.1,
                      lookahead_time: float = 0.05, gain: float = 300) -> bool:
        """
        Send ServoJ with RELATIVE joint increments (radians).
        Args:
            delta_joints: Joint increments in RADIANS
        """
        if self.server is None:
            return True
        # Server expects radians
        return self.server.servo_j_delta(arm_name, delta_joints.tolist(), t, lookahead_time, gain)
    
    # ==================== ServoP Control (Pose Servo) ====================
    
    def servo_p(self, arm_name: str, pose: np.ndarray) -> bool:
        """
        Send ServoP with target pose [x, y, z, rx, ry, rz] (m, radians).
        Args:
            pose: Target pose in METERS and RADIANS
        """
        if self.server is None:
            return True
        # Server expects meters and radians
        return self.server.servo_p(arm_name, pose.tolist())
    
    def servo_p_delta(self, arm_name: str, delta_pose: np.ndarray) -> bool:
        """
        Send ServoP with RELATIVE pose increments (m, radians).
        Args:
            delta_pose: Pose increments in METERS and RADIANS
        """
        if self.server is None:
            return True
        # Server expects meters and radians
        return self.server.servo_p_delta(arm_name, delta_pose.tolist())
    
    # ==================== Inverse Kinematics ====================
    
    def inverse_kinematics(self, arm_name: str, pose: np.ndarray, 
                          current_joints: np.ndarray = None) -> Optional[list]:
        """
        Solve IK using Dobot controller.
        Args:
            arm_name: 'left' or 'right'
            pose: Target pose [x, y, z, rx, ry, rz] (m, radians)
            current_joints: Current joints for reference (radians)
        Returns:
            Joint angles (radians) or None if failed
        """
        if self.server is None:
            return [0.0] * 6
        # Server expects meters and radians, returns radians
        pose_list = pose.tolist() if hasattr(pose, 'tolist') else list(pose)
        joints_list = current_joints.tolist() if current_joints is not None and hasattr(current_joints, 'tolist') else current_joints
        return self.server.inverse_kinematics(arm_name, pose_list, joints_list)
    
    # ==================== Gripper ====================
    
    def left_gripper_initialize(self):
        if self.server is None:
            return
        self.server.left_gripper_initialize()
    
    def left_gripper_goto(self, width: float, speed: float, force: float):
        if self.server is None:
            return
        self.server.left_gripper_goto(width, speed, force)
    
    def left_gripper_get_state(self) -> dict:
        if self.server is None:
            return {"width": 0.04, "is_moving": False, "is_grasped": False}
        return self.server.left_gripper_get_state()
    
    def right_gripper_initialize(self):
        if self.server is None:
            return
        self.server.right_gripper_initialize()
    
    def right_gripper_goto(self, width: float, speed: float, force: float):
        if self.server is None:
            return
        self.server.right_gripper_goto(width, speed, force)
    
    def right_gripper_get_state(self) -> dict:
        if self.server is None:
            return {"width": 0.04, "is_moving": False, "is_grasped": False}
        return self.server.right_gripper_get_state()
    
    def gripper_initialize(self):
        self.left_gripper_initialize()
        self.right_gripper_initialize()
    
    # ==================== Utility ====================
    
    def stop(self, arm_name: str):
        """Stop specified arm."""
        if self.server is None:
            return
        self.server.stop(arm_name)
    
    def close(self):
        """Close connection."""
        if self.server is not None:
            try:
                self.server.close()
            except:
                pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    client = DobotDualArmClient()
    
    # Test connection
    print("Testing connection...")
    left_joints = client.left_robot_get_joint_positions()
    right_joints = client.right_robot_get_joint_positions()
    print(f"Left joints (rad): {left_joints}")
    print(f"Right joints (rad): {right_joints}")
    
    left_pose = client.left_robot_get_ee_pose()
    print(f"Left pose (m, rad): {left_pose}")
    
    client.close()
    print("Done!")
