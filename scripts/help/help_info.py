def main():
    print("""
==================================================
Dual-Arm Teleoperation - Command Reference
==================================================

Core Commands:
  robot-record           Record teleoperation dataset
  robot-replay           Replay a recorded dataset
  robot-visualize        Visualize recorded dataset
  robot-reset            Reset the robot to initial state
  robot-train            Train a policy on the recorded dataset

Utility Commands:
  utils-joint-offsets    Compute joint offsets for teleoperation

Tool Commands:
  tools-check-dataset    Check local dataset information
  tools-check-rs         Retrieve connected RealSense camera serial numbers
  tools-check-robotiq    Check Robotiq gripper serial ports

Shell Tools:
  check_robotiq_ports.sh  Get Robotiq gripper serial ports

Test Commands:
  test-gripper-ctrl      Run gripper control command (operate the gripper)

--------------------------------------------------
 Tip: Use 'robot-help' anytime to see this summary.
==================================================
""")
