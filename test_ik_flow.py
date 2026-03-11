#!/usr/bin/env python
"""
Test script for IK flow:
1. Get current joint positions from robot
2. Compute current EE pose via placo FK
3. Set a target EE pose (small offset from current)
4. Solve IK to get target joint positions
5. Send action to robot
6. Repeat 100 steps
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import placo

# Import robot client
from lerobot_robot.dobot_interface_client import DobotDualArmClient

# URDF path
URDF_PATH = "/home/geist/lerobot/dual_arm_data_collection/lerobot_dual_arm_teleop/assets/dual_dobot/dual_nova5_robot.urdf"

# Joint indices in placo model
LEFT_ARM_Q_IDX = slice(7, 13)   # q[7:13]
RIGHT_ARM_Q_IDX = slice(25, 31) # q[25:31]


def create_ik_solver(urdf_path: str):
    """Create and configure placo IK solver."""
    robot = placo.RobotWrapper(urdf_path)
    solver = placo.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.mask_fbase(True)
    
    # Add regularization
    solver.add_kinetic_energy_regularization_task(1e-6)
    joint_reg = solver.add_regularization_task(0.1)
    joint_reg.configure("joint_regularization", "soft", 0.1)
    
    # Enable joint limits
    solver.enable_joint_limits(True)
    
    return robot, solver


def create_frame_task(solver, robot, ee_link: str, initial_pose: np.ndarray):
    """Create a frame task for end-effector."""
    task = solver.add_frame_task(ee_link, initial_pose)
    task.configure(f"{ee_link}_frame", "soft", 1.0)
    return task


def get_current_ee_pose(robot, ee_link: str) -> tuple:
    """Get current end-effector pose from placo FK."""
    T_ee = robot.get_T_world_frame(ee_link)
    pos = T_ee[:3, 3].copy()
    rot = R.from_matrix(T_ee[:3, :3]).as_euler("xyz")
    return pos, rot, T_ee


def compute_target_pose(current_pos: np.ndarray, current_rot: np.ndarray, 
                        delta_pos: np.ndarray, delta_rot: np.ndarray) -> tuple:
    """Compute target pose by adding delta to current pose."""
    target_pos = current_pos + delta_pos
    
    R_current = R.from_euler("xyz", current_rot)
    R_delta = R.from_euler("xyz", delta_rot)
    R_target = R_delta * R_current
    target_rot = R_target.as_euler("xyz")
    
    # Build transform matrix
    T_target = np.eye(4)
    T_target[:3, 3] = target_pos
    T_target[:3, :3] = R_target.as_matrix()
    
    return target_pos, target_rot, T_target


def main():
    print("=" * 60)
    print("IK Flow Test")
    print("=" * 60)
    
    # Connect to robot
    print("\n[1] Connecting to robot...")
    robot_client = DobotDualArmClient(ip='127.0.0.1', port=4242)
    
    # Create IK solver
    print("[2] Creating IK solver...")
    placo_robot, solver = create_ik_solver(URDF_PATH)
    
    # Create frame tasks
    print("[3] Creating frame tasks...")
    initial_pose = np.eye(4)
    left_task = create_frame_task(solver, placo_robot, "left_Link6", initial_pose)
    right_task = create_frame_task(solver, placo_robot, "right_Link6", initial_pose)
    
    # Get initial joint positions
    print("[4] Getting initial joint positions...")
    left_joints = robot_client.left_robot_get_joint_positions()
    right_joints = robot_client.right_robot_get_joint_positions()
    print(f"    Left joints:  {np.rad2deg(left_joints).round(2)} deg")
    print(f"    Right joints: {np.rad2deg(right_joints).round(2)} deg")
    
    # Set initial joints in placo
    placo_robot.state.q[LEFT_ARM_Q_IDX] = left_joints
    placo_robot.state.q[RIGHT_ARM_Q_IDX] = right_joints
    placo_robot.update_kinematics()
    
    # Get initial EE poses
    left_pos, left_rot, _ = get_current_ee_pose(placo_robot, "left_Link6")
    right_pos, right_rot, _ = get_current_ee_pose(placo_robot, "right_Link6")
    print(f"\n    Left EE:  pos={left_pos.round(4)}, rot={np.rad2deg(left_rot).round(2)} deg")
    print(f"    Right EE: pos={right_pos.round(4)}, rot={np.rad2deg(right_rot).round(2)} deg")
    
    # Define small delta for testing (move forward 1cm, rotate 1 deg)
    delta_pos = np.array([0.01, 0.0, 0.0])  # 1cm forward
    delta_rot = np.array([0.0, 0.0, np.deg2rad(1.0)])  # 1 deg yaw
    
    print(f"\n[5] Delta: pos={delta_pos}m, rot={np.rad2deg(delta_rot)} deg")
    
    # Main loop
    print("\n[6] Starting test loop (100 steps)...")
    print("-" * 60)
    
    for step in range(100):
        t_start = time.perf_counter()
        
        # --- Step 1: Get current joint positions from robot ---
        t1 = time.perf_counter()
        left_joints = robot_client.left_robot_get_joint_positions()
        right_joints = robot_client.right_robot_get_joint_positions()
        t_get_joints = (time.perf_counter() - t1) * 1000
        
        # --- Step 2: Update placo model and get current EE pose ---
        t2 = time.perf_counter()
        placo_robot.state.q[LEFT_ARM_Q_IDX] = left_joints
        placo_robot.state.q[RIGHT_ARM_Q_IDX] = right_joints
        placo_robot.update_kinematics()
        
        left_pos, left_rot, _ = get_current_ee_pose(placo_robot, "left_Link6")
        right_pos, right_rot, _ = get_current_ee_pose(placo_robot, "right_Link6")
        t_fk = (time.perf_counter() - t2) * 1000
        
        # --- Step 3: Compute target pose ---
        t3 = time.perf_counter()
        _, _, T_left_target = compute_target_pose(left_pos, left_rot, delta_pos, delta_rot)
        _, _, T_right_target = compute_target_pose(right_pos, right_rot, delta_pos, delta_rot)
        t_target = (time.perf_counter() - t3) * 1000
        
        # --- Step 4: Solve IK ---
        t4 = time.perf_counter()
        left_task.T_world_frame = T_left_target
        right_task.T_world_frame = T_right_target
        solver.solve(True)
        
        target_left_q = placo_robot.state.q[LEFT_ARM_Q_IDX].copy()
        target_right_q = placo_robot.state.q[RIGHT_ARM_Q_IDX].copy()
        t_ik = (time.perf_counter() - t4) * 1000
        
        # --- Step 5: Send action to robot ---
        t5 = time.perf_counter()
        robot_client.left_robot_move_to_joint_positions(target_left_q)
        robot_client.right_robot_move_to_joint_positions(target_right_q)
        t_send = (time.perf_counter() - t5) * 1000
        
        # --- Timing summary ---
        t_total = (time.perf_counter() - t_start) * 1000
        
        if step % 10 == 0:
            print(f"Step {step:3d}: "
                  f"get_joints={t_get_joints:5.1f}ms, "
                  f"FK={t_fk:5.1f}ms, "
                  f"target={t_target:5.1f}ms, "
                  f"IK={t_ik:5.1f}ms, "
                  f"send={t_send:5.1f}ms, "
                  f"total={t_total:5.1f}ms")
        
        # Small delay to not overwhelm the robot
        time.sleep(0.01)
    
    print("-" * 60)
    print("\n[7] Test completed!")
    
    # Get final positions
    left_joints = robot_client.left_robot_get_joint_positions()
    right_joints = robot_client.right_robot_get_joint_positions()
    placo_robot.state.q[LEFT_ARM_Q_IDX] = left_joints
    placo_robot.state.q[RIGHT_ARM_Q_IDX] = right_joints
    placo_robot.update_kinematics()
    
    left_pos, left_rot, _ = get_current_ee_pose(placo_robot, "left_Link6")
    right_pos, right_rot, _ = get_current_ee_pose(placo_robot, "right_Link6")
    
    print(f"\nFinal Left EE:  pos={left_pos.round(4)}, rot={np.rad2deg(left_rot).round(2)} deg")
    print(f"Final Right EE: pos={right_pos.round(4)}, rot={np.rad2deg(right_rot).round(2)} deg")


if __name__ == "__main__":
    main()
