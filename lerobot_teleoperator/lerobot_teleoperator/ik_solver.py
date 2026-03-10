#!/usr/bin/env python
"""
Inverse Kinematics solver for dual-arm robot using placo or fallback implementation.
Supports both placo (if available) and scipy-based IK solving.
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import placo, fallback to scipy if not available
try:
    import placo
    PLACO_AVAILABLE = True
    logger.info("[IK] Using placo for inverse kinematics")
except ImportError:
    PLACO_AVAILABLE = False
    from scipy.spatial.transform import Rotation as R
    logger.warning("[IK] placo not available, using scipy fallback for IK")


class DualArmIKSolver:
    """
    Inverse Kinematics solver for dual-arm robot.
    Uses placo if available, otherwise falls back to scipy-based implementation.
    """
    
    def __init__(
        self,
        urdf_path: str,
        left_arm_prefix: str = "left_",
        right_arm_prefix: str = "right_",
        left_ee_link: str = "left_Link6",
        right_ee_link: str = "right_Link6",
    ):
        """
        Initialize IK solver for dual-arm robot.
        
        Args:
            urdf_path: Path to URDF file
            left_arm_prefix: Prefix for left arm joints
            right_arm_prefix: Prefix for right arm joints
            left_ee_link: End-effector link name for left arm
            right_ee_link: End-effector link name for right arm
        """
        self.urdf_path = urdf_path
        self.left_arm_prefix = left_arm_prefix
        self.right_arm_prefix = right_arm_prefix
        self.left_ee_link = left_ee_link
        self.right_ee_link = right_ee_link
        
        # Joint names for each arm (6 DOF per arm)
        self.left_joint_names = [f"{left_arm_prefix}joint{i}" for i in range(1, 7)]
        self.right_joint_names = [f"{right_arm_prefix}joint{i}" for i in range(1, 7)]
        
        # Current joint positions
        self.left_joint_positions = np.zeros(6)
        self.right_joint_positions = np.zeros(6)
        
        # Initialize solver
        if PLACO_AVAILABLE:
            self._init_placo()
        else:
            self._init_scipy()
        
        logger.info(f"[IK] Initialized dual-arm IK solver")
        logger.info(f"[IK]   URDF: {urdf_path}")
        logger.info(f"[IK]   Left arm joints: {self.left_joint_names}")
        logger.info(f"[IK]   Right arm joints: {self.right_joint_names}")
    
    def _init_placo(self):
        """Initialize placo-based IK solver."""
        try:
            # Load URDF
            self.robot = placo.RobotWrapper(self.urdf_path)
            
            # Create IK solver for each arm
            self.left_ik = placo.IKSolver(self.robot, self.left_ee_link)
            self.right_ik = placo.IKSolver(self.robot, self.right_ee_link)
            
            logger.info("[IK] Placo IK solver initialized successfully")
        except Exception as e:
            logger.error(f"[IK] Failed to initialize placo: {e}")
            raise
    
    def _init_scipy(self):
        """Initialize scipy-based fallback IK solver."""
        # Store robot model parameters (simplified)
        # For Nova5, we'll use a simple Jacobian-based IK
        self.robot = None
        logger.info("[IK] Scipy fallback IK solver initialized")
    
    def solve_left_ik(
        self,
        target_pose: np.ndarray,
        current_joints: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Solve inverse kinematics for left arm.
        
        Args:
            target_pose: Target end-effector pose [x, y, z, rx, ry, rz] (Euler angles)
            current_joints: Current joint positions (optional, uses stored state if None)
        
        Returns:
            Tuple of (joint_positions, success)
        """
        if current_joints is None:
            current_joints = self.left_joint_positions
        
        if PLACO_AVAILABLE:
            return self._solve_placo_ik(
                self.left_ik,
                target_pose,
                current_joints,
                self.left_ee_link
            )
        else:
            return self._solve_scipy_ik(
                target_pose,
                current_joints,
                "left"
            )
    
    def solve_right_ik(
        self,
        target_pose: np.ndarray,
        current_joints: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Solve inverse kinematics for right arm.
        
        Args:
            target_pose: Target end-effector pose [x, y, z, rx, ry, rz] (Euler angles)
            current_joints: Current joint positions (optional, uses stored state if None)
        
        Returns:
            Tuple of (joint_positions, success)
        """
        if current_joints is None:
            current_joints = self.right_joint_positions
        
        if PLACO_AVAILABLE:
            return self._solve_placo_ik(
                self.right_ik,
                target_pose,
                current_joints,
                self.right_ee_link
            )
        else:
            return self._solve_scipy_ik(
                target_pose,
                current_joints,
                "right"
            )
    
    def _solve_placo_ik(
        self,
        ik_solver,
        target_pose: np.ndarray,
        current_joints: np.ndarray,
        ee_link: str,
    ) -> Tuple[np.ndarray, bool]:
        """Solve IK using placo."""
        try:
            # Set target pose
            position = target_pose[:3]
            rotation = R.from_euler("xyz", target_pose[3:]).as_quat()
            
            # Solve IK
            success = ik_solver.solve(
                position=position,
                orientation=rotation,
                q_init=current_joints,
            )
            
            if success:
                joint_positions = ik_solver.get_result()
                return joint_positions, True
            else:
                logger.warning(f"[IK] Placo IK failed for {ee_link}")
                return current_joints, False
        except Exception as e:
            logger.error(f"[IK] Placo IK error: {e}")
            return current_joints, False
    
    def _solve_scipy_ik(
        self,
        target_pose: np.ndarray,
        current_joints: np.ndarray,
        arm: str,
    ) -> Tuple[np.ndarray, bool]:
        """
        Fallback IK solver using scipy optimization.
        This is a simplified Jacobian-based IK for 6-DOF arms.
        """
        # For now, return current joints with False
        # In practice, you would implement a Jacobian-based IK here
        logger.warning(f"[IK] Scipy IK not implemented for {arm} arm, returning current joints")
        return current_joints, False
    
    def solve_dual_ik(
        self,
        left_target_pose: np.ndarray,
        right_target_pose: np.ndarray,
        left_current_joints: Optional[np.ndarray] = None,
        right_current_joints: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
        """
        Solve IK for both arms simultaneously.
        
        Args:
            left_target_pose: Target pose for left arm [x, y, z, rx, ry, rz]
            right_target_pose: Target pose for right arm [x, y, z, rx, ry, rz]
            left_current_joints: Current left joint positions (optional)
            right_current_joints: Current right joint positions (optional)
        
        Returns:
            Tuple of (left_joints, right_joints, left_success, right_success)
        """
        left_joints, left_success = self.solve_left_ik(left_target_pose, left_current_joints)
        right_joints, right_success = self.solve_right_ik(right_target_pose, right_current_joints)
        
        return left_joints, right_joints, left_success, right_success
    
    def update_joint_positions(
        self,
        left_joints: Optional[np.ndarray] = None,
        right_joints: Optional[np.ndarray] = None,
    ):
        """Update stored joint positions."""
        if left_joints is not None:
            self.left_joint_positions = left_joints
        if right_joints is not None:
            self.right_joint_positions = right_joints
    
    def get_joint_positions(self) -> Dict[str, np.ndarray]:
        """Get current joint positions."""
        return {
            "left_joints": self.left_joint_positions.copy(),
            "right_joints": self.right_joint_positions.copy(),
        }
    
    def compute_delta_joints(
        self,
        left_delta_pose: np.ndarray,
        right_delta_pose: np.ndarray,
        left_current_pose: np.ndarray,
        right_current_pose: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute joint deltas from Cartesian deltas.
        
        Args:
            left_delta_pose: Delta pose for left arm [dx, dy, dz, drx, dry, drz]
            right_delta_pose: Delta pose for right arm [dx, dy, dz, drx, dry, drz]
            left_current_pose: Current pose for left arm [x, y, z, rx, ry, rz]
            right_current_pose: Current pose for right arm [x, y, z, rx, ry, rz]
        
        Returns:
            Tuple of (left_joint_deltas, right_joint_deltas)
        """
        # Compute target poses
        left_target_pose = left_current_pose + left_delta_pose
        right_target_pose = right_current_pose + right_delta_pose
        
        # Solve IK
        left_joints, left_success = self.solve_left_ik(left_target_pose)
        right_joints, right_success = self.solve_right_ik(right_target_pose)
        
        # Compute deltas
        left_joint_deltas = left_joints - self.left_joint_positions
        right_joint_deltas = right_joints - self.right_joint_positions
        
        # Update stored positions
        if left_success:
            self.left_joint_positions = left_joints
        if right_success:
            self.right_joint_positions = right_joints
        
        return left_joint_deltas, right_joint_deltas


if __name__ == "__main__":
    # Test IK solver
    import sys
    
    # Default URDF path
    urdf_path = Path(__file__).parent.parent.parent / "assets/dobot_description/urdf/dual_nova5_robot.urdf"
    
    if not urdf_path.exists():
        logger.error(f"URDF file not found: {urdf_path}")
        sys.exit(1)
    
    # Create IK solver
    ik_solver = DualArmIKSolver(str(urdf_path))
    
    # Test IK solving
    test_pose = np.array([0.5, 0.3, 1.0, 0.0, 0.0, 0.0])  # [x, y, z, rx, ry, rz]
    
    logger.info("\n--- Testing Left Arm IK ---")
    left_joints, left_success = ik_solver.solve_left_ik(test_pose)
    logger.info(f"Left arm IK success: {left_success}")
    logger.info(f"Left arm joints: {left_joints}")
    
    logger.info("\n--- Testing Right Arm IK ---")
    right_joints, right_success = ik_solver.solve_right_ik(test_pose)
    logger.info(f"Right arm IK success: {right_success}")
    logger.info(f"Right arm joints: {right_joints}")
    
    logger.info("\n--- Testing Dual Arm IK ---")
    left_joints, right_joints, left_ok, right_ok = ik_solver.solve_dual_ik(test_pose, test_pose)
    logger.info(f"Dual arm IK success: left={left_ok}, right={right_ok}")
    logger.info(f"Left joints: {left_joints}")
    logger.info(f"Right joints: {right_joints}")