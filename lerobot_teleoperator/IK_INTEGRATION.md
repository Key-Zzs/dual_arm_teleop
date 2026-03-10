# IK Solver Integration for Dual-Arm Teleoperation

## Overview

This implementation adds inverse kinematics (IK) support to the dual-arm teleoperation system, allowing simultaneous tracking of both end-effector pose and joint positions.

**New Feature**: Automatically fetches robot state from `DobotDualArmClient` for real-time IK computation!

## Features

1. **Dual IK Solver**: Supports both placo (preferred) and scipy fallback
2. **Real-time Joint Tracking**: Computes joint positions from Cartesian delta poses
3. **Synchronized Recording**: Records both EE pose and joint positions simultaneously
4. **URDF-based**: Uses the robot's URDF for accurate kinematics
5. **Automatic State Feedback**: Connects to robot server to get current state automatically

## Installation

### Option 1: Install placo (Recommended)

```bash
pip install placo
```

### Option 2: Use scipy fallback

If placo is not available, the system will automatically fall back to a scipy-based implementation (currently returns current joints).

## Usage

### 1. Basic Usage (Automatic State Feedback)

```python
from lerobot_teleoperator import OculusDualArmTeleop
from lerobot_teleoperator.config_teleop import OculusDualArmTeleopConfig

# Create config with robot connection
config = OculusDualArmTeleopConfig(
    ip="192.168.110.62",          # Oculus Quest IP
    robot_ip="127.0.0.1",         # Robot server IP
    robot_port=4242,              # Robot server port
    use_gripper=True,
    left_pose_scaler=[1.0, 1.0],
    right_pose_scaler=[1.0, 1.0],
)

# Create teleop
teleop = OculusDualArmTeleop(config)
teleop.connect()

# Get action (automatically fetches robot state and computes IK)
action = teleop.get_action()

# Action contains:
# - left_delta_ee_pose.{x,y,z,rx,ry,rz}
# - right_delta_ee_pose.{x,y,z,rx,ry,rz}
# - left_joint_{1-6}.pos (computed via IK)
# - right_joint_{1-6}.pos (computed via IK)
# - left_gripper_cmd_bin
# - right_gripper_cmd_bin
# - reset_requested
```

### 2. Manual State Update (Optional)

If you want to manually control state updates:

```python
# Disable automatic state fetch by not connecting robot_client
config = OculusDualArmTeleopConfig(
    ip="192.168.110.62",
    robot_ip=None,  # Disable robot client
)

teleop = OculusDualArmTeleop(config)
teleop.connect()

# Manually update state
left_ee_pose = np.array([0.5, 0.3, 1.0, 0.0, 0.0, 0.0])
right_ee_pose = np.array([0.5, 0.3, 1.0, 0.0, 0.0, 0.0])
left_joints = np.zeros(6)
right_joints = np.zeros(6)

teleop.update_robot_state(
    left_ee_pose=left_ee_pose,
    right_ee_pose=right_ee_pose,
    left_joints=left_joints,
    right_joints=right_joints,
)

# Get action
action = teleop.get_action()
```

### 3. Integration with Robot Control Loop

```python
# In your robot control loop:
while True:
    # Get action (automatically fetches current state from robot)
    action = teleop.get_action()
    
    # Send action to robot
    robot.send_action(action)
    
    # The teleop already has the latest state, no need to update manually
```

## Architecture

```
OculusDualArmTeleop
├── OculusDualArmRobot (Cartesian delta pose)
│   ├── Left controller -> left_delta_ee_pose
│   └── Right controller -> right_delta_ee_pose
├── DobotDualArmClient (Robot state feedback)
│   ├── left_robot_get_ee_pose() -> current EE pose
│   ├── right_robot_get_ee_pose() -> current EE pose
│   ├── left_robot_get_joint_positions() -> current joints
│   └── right_robot_get_joint_positions() -> current joints
└── DualArmIKSolver (Joint positions)
    ├── Left arm IK: delta_pose -> joint_positions
    └── Right arm IK: delta_pose -> joint_positions
```

## Data Flow

```
1. Oculus Controller -> delta_pose
2. Robot Client -> current_pose, current_joints
3. IK Solver: current_pose + delta_pose -> target_pose -> target_joints
4. Output: {delta_pose, target_joints, gripper_cmd}
```

## Configuration

### URDF Path

The IK solver automatically loads the URDF from:
```
assets/dobot_description/urdf/dual_nova5_robot.urdf
```

### Robot Connection

Add to your config:
```python
config = OculusDualArmTeleopConfig(
    robot_ip="127.0.0.1",  # Robot server IP
    robot_port=4242,       # Robot server port
)
```

### Joint Names

Default joint names (can be customized):
- Left arm: `left_joint1` to `left_joint6`
- Right arm: `right_joint1` to `right_joint6`

### End-Effector Links

Default EE links:
- Left arm: `left_Link6`
- Right arm: `right_Link6`

## Troubleshooting

### Issue: Robot client not connected

**Solution**: Check that the robot server is running:
```bash
# Start robot server
python3 lerobot_robot/lerobot_robot/dobot_interface_server.py
```

### Issue: IK solver not initialized

**Solution**: Check that the URDF file exists at the expected path:
```bash
ls assets/dobot_description/urdf/dual_nova5_robot.urdf
```

### Issue: IK solution fails

**Possible causes**:
1. Target pose is outside workspace
2. Target pose is in singularity
3. Current robot state not updated

**Solution**: 
- Ensure robot server is running and accessible
- Check joint limits in URDF
- Use smaller delta poses

### Issue: placo not available

**Solution**: Install placo or use scipy fallback:
```bash
pip install placo
```

## Testing

Run the IK solver test:
```bash
python3 lerobot_teleoperator/lerobot_teleoperator/ik_solver.py
```

Run the teleop test:
```bash
python3 lerobot_teleoperator/lerobot_teleoperator/oculus_dual_arm_teleop.py --ip 192.168.110.62
```

Run the integration test:
```bash
python3 lerobot_teleoperator/lerobot_teleoperator/test_ik_integration.py
```

## Benefits of Automatic State Feedback

1. **Simplified Integration**: No need to manually call `update_robot_state()`
2. **Always Up-to-Date**: IK solver always has the latest robot state
3. **Reduced Latency**: Direct connection to robot server
4. **Error Resilient**: Falls back gracefully if robot client unavailable

## Future Improvements

1. **Jacobian-based IK**: Implement scipy fallback using Jacobian pseudo-inverse
2. **Singularity handling**: Add singularity detection and avoidance
3. **Joint limit checking**: Validate IK solutions against joint limits
4. **Collision checking**: Add self-collision avoidance
5. **Redundancy resolution**: Use null-space optimization for 7-DOF arms