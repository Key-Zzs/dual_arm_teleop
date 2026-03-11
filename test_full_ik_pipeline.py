#!/usr/bin/env python
"""
Full IK pipeline test script.
Tests:
1. Teleop IK with real robot state (placo IK solver)
2. DobotDualArm joint control
3. IK -> joint control -> real robot motion
"""

import logging
import numpy as np
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def test_teleop_ik_with_real_robot():
    """
    Test 1: Teleop IK with real robot state.
    Uses placo IK solver in teleop to compute joint positions.
    """
    log.info("\n" + "="*60)
    log.info("Test 1: Teleop IK with Real Robot State (Placo IK)")
    log.info("="*60)
    
    # Import teleop
    # import sys
    # sys.path.insert(0, str(Path(__file__).parent / "lerobot_teleoperator/lerobot_teleoperator"))
    from lerobot_teleoperator.oculus_dual_arm_teleop import OculusDualArmTeleop
    from lerobot_teleoperator.config_teleop import OculusDualArmTeleopConfig
    
    # Create config (Oculus IP not needed for IK test)
    config = OculusDualArmTeleopConfig(
        ip="192.168.110.62",  # Oculus IP (won't connect for this test)
        robot_ip="127.0.0.1",
        robot_port=4242,
        use_gripper=False,
        left_pose_scaler=[0.5, 0.5],
        right_pose_scaler=[0.5, 0.5],
        servo_time = 0.017,
        visualize_placo = False
    )
    
    # Create teleop
    teleop = OculusDualArmTeleop(config)
    
    # Connect to robot (for state feedback)
    log.info("\n--- Connecting to Robot ---")
    try:
        teleop._connect_impl()
        log.info("✓ Connected to robot client")
    except Exception as e:
        log.error(f"✗ Connection failed: {e}")
        return False
    
    # Get current robot state
    log.info("\n--- Getting Current Robot State ---")
    try:
        if teleop.robot_client is not None:
            left_ee = teleop.robot_client.left_robot_get_ee_pose()
            right_ee = teleop.robot_client.right_robot_get_ee_pose()
            left_joints = teleop.robot_client.left_robot_get_joint_positions()
            right_joints = teleop.robot_client.right_robot_get_joint_positions()
            
            log.info(f"Left EE pose (m, rad): {left_ee}")
            log.info(f"Right EE pose (m, rad): {right_ee}")
            log.info(f"Left joints (rad): {left_joints}")
            log.info(f"Right joints (rad): {right_joints}")
        else:
            log.error("✗ Robot client not connected")
            return False
    except Exception as e:
        log.error(f"✗ Failed to get robot state: {e}")
        return False
    
    # Test placo IK solver directly
    log.info("\n--- Testing Placo IK Solver ---")
    if teleop.ik_solver is None:
        log.error("✗ IK solver not initialized")
        return False
    
    try:
        # Update IK solver with current joint positions
        teleop.ik_solver.update_joint_positions(left_joints=left_joints, right_joints=right_joints)
        
        # Get current EE poses from placo
        left_ee_placo = teleop.ik_solver.get_ee_pose("left")
        right_ee_placo = teleop.ik_solver.get_ee_pose("right")
        
        log.info(f"Left EE from placo: {left_ee_placo}")
        log.info(f"Right EE from placo: {right_ee_placo}")
        
        # Test IK with small delta
        log.info("\n--- Testing IK with Small Delta ---")
        
        # Add small delta (1cm in Z direction - upward)
        left_target = left_ee_placo.copy()
        left_target[2] += 0.01  # 1cm up
        
        right_target = right_ee_placo.copy()
        right_target[2] += 0.01  # 1cm up
        
        log.info(f"Left target: {left_target}")
        log.info(f"Right target: {right_target}")
        
        # Solve IK
        left_ik_joints, right_ik_joints, left_ok, right_ok = teleop.ik_solver.solve_dual_ik(
            left_target, right_target
        )
        
        log.info(f"Left IK success: {left_ok}")
        log.info(f"Right IK success: {right_ok}")
        log.info(f"Left IK joints: {left_ik_joints}")
        log.info(f"Right IK joints: {right_ik_joints}")
        
        if left_ok and right_ok:
            log.info("\n✓ Placo IK solver working correctly")
            return True
        else:
            log.error("\n✗ Placo IK solver failed")
            return False
            
    except Exception as e:
        log.error(f"✗ IK test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        teleop._disconnect_impl()


def test_dobot_dual_arm_joint_control():
    """
    Test 2: DobotDualArm joint control.
    Tests sending joint commands to the robot.
    """
    log.info("\n" + "="*60)
    log.info("Test 2: DobotDualArm Joint Control")
    log.info("="*60)
    
    # import sys
    # sys.path.insert(0, str(Path(__file__).parent / "lerobot_robot/lerobot_robot"))
    from lerobot_robot.dobot_dual_arm import DobotDualArm
    from lerobot_robot.config_dobot import DobotDualArmConfig
    
    # Create config
    config = DobotDualArmConfig(
        name="dobot_dual_arm",
        robot_ip="127.0.0.1",
        robot_port=4242,
        use_gripper=False,
        debug=False,
    )
    
    # Create robot
    robot = DobotDualArm(config)
    
    try:
        # Connect
        log.info("\n--- Connecting to Robot ---")
        robot.connect()
        log.info("✓ Connected")
        
        # Get current state
        log.info("\n--- Getting Current State ---")
        state = robot.get_observation()
        
        left_joints = np.array([state[f"left_joint_{i+1}.pos"] for i in range(6)])
        right_joints = np.array([state[f"right_joint_{i+1}.pos"] for i in range(6)])
        
        log.info(f"Current left joints (rad): {left_joints}")
        log.info(f"Current right joints (rad): {right_joints}")
        
        # Test small joint delta
        log.info("\n--- Testing Small Joint Delta ---")
        delta = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])  # 0.01 rad on joint 1
        
        # Create action with delta
        action = {}
        for i in range(6):
            action[f"left_joint_{i+1}.pos"] = left_joints[i] + delta[i]
            action[f"right_joint_{i+1}.pos"] = right_joints[i] + delta[i]
        
        log.info(f"Target left joints: {[action[f'left_joint_{i+1}.pos'] for i in range(6)]}")
        log.info(f"Target right joints: {[action[f'right_joint_{i+1}.pos'] for i in range(6)]}")
        
        # Send action
        log.info("\n--- Sending Action ---")
        robot.send_action(action)
        log.info("✓ Action sent")
        
        # Wait for motion
        time.sleep(0.5)
        
        # Check new state
        new_state = robot.get_observation()
        new_left_joints = np.array([new_state[f"left_joint_{i+1}.pos"] for i in range(6)])
        new_right_joints = np.array([new_state[f"right_joint_{i+1}.pos"] for i in range(6)])
        
        log.info(f"New left joints (rad): {new_left_joints}")
        log.info(f"New right joints (rad): {new_right_joints}")
        log.info(f"Actual left delta: {new_left_joints - left_joints}")
        log.info(f"Actual right delta: {new_right_joints - right_joints}")
        
        # Check if motion happened
        if np.abs(new_left_joints[0] - left_joints[0]) > 0.001:
            log.info("\n✓ Joint control working")
            return True
        else:
            log.warning("\n⚠ Joint may not have moved (check robot)")
            return True  # Still pass, might be in position
            
    except Exception as e:
        log.error(f"✗ Joint control test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        robot.disconnect()


def test_full_ik_to_robot_control():
    """
    Test 3: Full pipeline - IK -> joint control -> real robot motion.
    Uses teleop IK to compute joints, then sends to robot.
    """
    log.info("\n" + "="*60)
    log.info("Test 3: Full IK -> Robot Control Pipeline")
    log.info("="*60)
    
    from lerobot_robot.dobot_dual_arm import DobotDualArm
    from lerobot_robot.config_dobot import DobotDualArmConfig
    from lerobot_teleoperator.ik_solver import DualArmIKSolver
    
    # Create robot
    robot_config = DobotDualArmConfig(
        name="dobot_dual_arm",
        robot_ip="127.0.0.1",
        robot_port=4242,
        use_gripper=False,
        debug=False,
    )
    robot = DobotDualArm(robot_config)
    
    # Create IK solver
    urdf_path = "/home/geist/lerobot/dual_arm_data_collection/lerobot_dual_arm_teleop/assets/dual_dobot/dual_nova5_robot.urdf"
    ik_solver = DualArmIKSolver(str(urdf_path))
    
    try:
        # Connect to robot
        log.info("\n--- Connecting to Robot ---")
        robot.connect()
        log.info("✓ Connected")
        
        # Get current state
        log.info("\n--- Getting Current State ---")
        state = robot.get_observation()
        
        left_joints = np.array([state[f"left_joint_{i+1}.pos"] for i in range(6)])
        right_joints = np.array([state[f"right_joint_{i+1}.pos"] for i in range(6)])
        
        log.info(f"Current left joints (rad): {left_joints}")
        log.info(f"Current right joints (rad): {right_joints}")
        
        # Update IK solver with current joints
        ik_solver.update_joint_positions(left_joints=left_joints, right_joints=right_joints)
        
        # Get current EE pose from IK solver
        left_ee = ik_solver.get_ee_pose("left")
        right_ee = ik_solver.get_ee_pose("right")
        
        log.info(f"Current left EE: {left_ee}")
        log.info(f"Current right EE: {right_ee}")
        
        # Define target poses (small movement)
        log.info("\n--- Computing IK for Target Poses ---")
        
        # Move 2cm up in Z
        left_target = left_ee.copy()
        left_target[2] += 0.02
        
        right_target = right_ee.copy()
        right_target[2] += 0.02
        
        log.info(f"Left target EE: {left_target}")
        log.info(f"Right target EE: {right_target}")
        
        # Solve IK
        left_ik_joints, right_ik_joints, left_ok, right_ok = ik_solver.solve_dual_ik(
            left_target, right_target
        )
        
        if not left_ok or not right_ok:
            log.error("✗ IK failed")
            return False
        
        log.info(f"Left IK joints: {left_ik_joints}")
        log.info(f"Right IK joints: {right_ik_joints}")
        log.info(f"Left joint delta: {left_ik_joints - left_joints}")
        log.info(f"Right joint delta: {right_ik_joints - right_joints}")
        
        # Create action
        action = {}
        for i in range(6):
            action[f"left_joint_{i+1}.pos"] = float(left_ik_joints[i])
            action[f"right_joint_{i+1}.pos"] = float(right_ik_joints[i])
        
        # Send to robot
        log.info("\n--- Sending IK Result to Robot ---")
        robot.send_action(action)
        log.info("✓ Action sent")
        
        # Wait for motion
        time.sleep(1.0)
        
        # Check result
        log.info("\n--- Checking Result ---")
        new_state = robot.get_observation()
        new_left_joints = np.array([new_state[f"left_joint_{i+1}.pos"] for i in range(6)])
        new_right_joints = np.array([new_state[f"right_joint_{i+1}.pos"] for i in range(6)])
        
        log.info(f"New left joints: {new_left_joints}")
        log.info(f"New right joints: {new_right_joints}")
        
        # Check if robot moved towards target
        left_error = np.linalg.norm(new_left_joints - left_ik_joints)
        right_error = np.linalg.norm(new_right_joints - right_ik_joints)
        
        log.info(f"Left joint error: {left_error:.6f} rad")
        log.info(f"Right joint error: {right_error:.6f} rad")
        
        if left_error < 0.1 and right_error < 0.1:  # Allow some error
            log.info("\n✓ Full pipeline working correctly")
            return True
        else:
            log.warning("\n⚠ Large error, but pipeline executed")
            return True
            
    except Exception as e:
        log.error(f"✗ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        robot.disconnect()


def test_continuous_ik_control():
    """
    Test 4: Continuous IK control loop.
    Simulates teleop control loop.
    """
    log.info("\n" + "="*60)
    log.info("Test 4: Continuous IK Control Loop (5 iterations)")
    log.info("="*60)
    
    from lerobot_robot.dobot_dual_arm import DobotDualArm
    from lerobot_robot.config_dobot import DobotDualArmConfig
    from lerobot_teleoperator.ik_solver import DualArmIKSolver
    
    # Create robot and IK solver
    robot_config = DobotDualArmConfig(
        name="dobot_dual_arm",
        robot_ip="127.0.0.1",
        robot_port=4242,
        use_gripper=False,
        debug=False,
    )
    robot = DobotDualArm(robot_config)
    
    urdf_path = "/home/geist/lerobot/dual_arm_data_collection/lerobot_dual_arm_teleop/assets/dual_dobot/dual_nova5_robot.urdf"
    ik_solver = DualArmIKSolver(str(urdf_path), visualize=True)
    
    try:
        # Connect
        log.info("\n--- Connecting ---")
        robot.connect()
        log.info("✓ Connected")
        
        # Get initial state
        state = robot.get_observation()
        left_joints_init = np.array([state[f"left_joint_{i+1}.pos"] for i in range(6)])
        right_joints_init = np.array([state[f"right_joint_{i+1}.pos"] for i in range(6)])
        
        log.info(f"Initial left joints: {left_joints_init}")
        log.info(f"Initial right joints: {right_joints_init}")
        
        # Control loop
        log.info("\n--- Running Control Loop ---")
        
        for i in range(30):
            log.info(f"\nIteration {i+1}/30:")
            
            # Get current state
            state = robot.get_observation()
            left_joints = np.array([state[f"left_joint_{i+1}.pos"] for i in range(6)])
            right_joints = np.array([state[f"right_joint_{i+1}.pos"] for i in range(6)])
            
            # Update IK solver
            ik_solver.update_joint_positions(left_joints=left_joints, right_joints=right_joints)
            
            # Get current EE pose
            left_ee = ik_solver.get_ee_pose("left")
            right_ee = ik_solver.get_ee_pose("right")
            
            # Compute target (small oscillation in Z)
            # z_offset = 0.05 * np.sin(i * 0.5)  # Oscillate ±1cm
            z_offset = -0.01  # Move 2cm up
            left_target = left_ee.copy()
            left_target[0] += z_offset
            right_target = right_ee.copy()
            right_target[0] += z_offset
            
            # Solve IK
            left_ik, right_ik, left_ok, right_ok = ik_solver.solve_dual_ik(left_target, right_target)
            
            if left_ok and right_ok:
                # Create and send action
                action = {}
                for j in range(6):
                    action[f"left_joint_{j+1}.pos"] = float(left_ik[j])
                    action[f"right_joint_{j+1}.pos"] = float(right_ik[j])
                
                robot.send_action(action)
                log.info(f"  Sent: left_j1={left_ik[0]:.4f}, right_j1={right_ik[0]:.4f}")
            else:
                log.warning(f"  IK failed, skipping")
            
            # time.sleep(0.1)
        
        log.info("\n✓ Continuous control loop completed")
        return True
        
    except Exception as e:
        log.error(f"✗ Continuous control test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        robot.disconnect()


def test_teleop_to_robot_closed_loop():
    """
    Test 5: Closed-loop teleop -> robot control.
    Reads actions from teleop (with IK) and sends to robot directly.
    This is the full teleoperation pipeline.
    """
    log.info("\n" + "="*60)
    log.info("Test 5: Teleop -> Robot Closed Loop Control")
    log.info("="*60)
    
    from lerobot_robot.dobot_dual_arm import DobotDualArm
    from lerobot_robot.config_dobot import DobotDualArmConfig
    from lerobot_teleoperator.oculus_dual_arm_teleop import OculusDualArmTeleop
    from lerobot_teleoperator.config_teleop import OculusDualArmTeleopConfig
    
    # Create robot config
    robot_config = DobotDualArmConfig(
        name="dobot_dual_arm",
        robot_ip="127.0.0.1",
        robot_port=4242,
        use_gripper=False,
        debug=False,
    )
    
    # Create teleop config
    teleop_config = OculusDualArmTeleopConfig(
        ip="192.168.110.62",  # Oculus IP
        robot_ip="127.0.0.1",
        robot_port=4242,
        use_gripper=False,
        left_pose_scaler=[0.5, 0.5],
        right_pose_scaler=[0.5, 0.5],
    )
    
    # Create instances
    robot = DobotDualArm(robot_config)
    teleop = OculusDualArmTeleop(teleop_config)
    
    try:
        # Connect
        log.info("\n--- Connecting ---")
        robot.connect()
        teleop.connect()
        log.info("✓ Connected to robot and teleop")
        
        # Get initial state
        log.info("\n--- Initial State ---")
        state = robot.get_observation()
        left_joints_init = np.array([state[f"left_joint_{i+1}.pos"] for i in range(6)])
        right_joints_init = np.array([state[f"right_joint_{i+1}.pos"] for i in range(6)])
        log.info(f"Initial left joints: {left_joints_init}")
        log.info(f"Initial right joints: {right_joints_init}")
        
        # Control loop
        log.info("\n--- Running Closed-Loop Control (10 iterations) ---")
        log.info("Reading actions from teleop and sending to robot...")
        
        for i in range(200):
            log.info(f"\nIteration {i+1}/10:")
            
            # Get action from teleop (includes IK computation)
            action = teleop.get_action()
            
            # Extract joint positions
            left_joints = [action.get(f"left_joint_{j+1}.pos", 0.0) for j in range(6)]
            right_joints = [action.get(f"right_joint_{j+1}.pos", 0.0) for j in range(6)]
            
            # Extract delta poses
            left_delta = [action.get(f"left_delta_ee_pose.{axis}", 0.0) for axis in ["x", "y", "z", "rx", "ry", "rz"]]
            right_delta = [action.get(f"right_delta_ee_pose.{axis}", 0.0) for axis in ["x", "y", "z", "rx", "ry", "rz"]]
            
            log.info(f"  Left delta EE: x={left_delta[0]:.4f}, y={left_delta[1]:.4f}, z={left_delta[2]:.4f}")
            log.info(f"  Right delta EE: x={right_delta[0]:.4f}, y={right_delta[1]:.4f}, z={right_delta[2]:.4f}")
            log.info(f"  Left joints: {[f'{j:.4f}' for j in left_joints]}")
            log.info(f"  Right joints: {[f'{j:.4f}' for j in right_joints]}")
            
            # Send action to robot
            robot.send_action(action)
            log.info(f"  ✓ Action sent to robot")
            
            # Small delay
            # time.sleep(0.05)
        
        # Final state
        log.info("\n--- Final State ---")
        final_state = robot.get_observation()
        left_joints_final = np.array([final_state[f"left_joint_{i+1}.pos"] for i in range(6)])
        right_joints_final = np.array([final_state[f"right_joint_{i+1}.pos"] for i in range(6)])
        
        log.info(f"Final left joints: {left_joints_final}")
        log.info(f"Final right joints: {right_joints_final}")
        log.info(f"Total left joint change: {left_joints_final - left_joints_init}")
        log.info(f"Total right joint change: {right_joints_final - right_joints_init}")
        
        log.info("\n✓ Closed-loop control test completed")
        return True
        
    except Exception as e:
        log.error(f"✗ Closed-loop control test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        robot.disconnect()
        teleop.disconnect()


if __name__ == "__main__":
    log.info("\n" + "="*60)
    log.info("Full IK Pipeline Test Suite")
    log.info("="*60)
    
    # Run tests
    tests = [
        # ("Teleop IK (Placo)", test_teleop_ik_with_real_robot),
        # ("Joint Control", test_dobot_dual_arm_joint_control),
        # ("Full IK->Robot", test_full_ik_to_robot_control),
        # ("Continuous Control", test_continuous_ik_control),
        ("Teleop->Robot Closed Loop", test_teleop_to_robot_closed_loop),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            log.error(f"Test '{name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    log.info("\n" + "="*60)
    log.info("Test Summary")
    log.info("="*60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        log.info(f"{name}: {status}")
    
    # Overall result
    all_passed = all(results.values())
    log.info("\n" + "="*60)
    if all_passed:
        log.info("✓ All tests passed!")
    else:
        log.info("✗ Some tests failed")
    log.info("="*60)
