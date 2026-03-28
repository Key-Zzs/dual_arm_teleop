#!/usr/bin/env python

"""
Oculus Quest dual-arm teleoperation implementation.
Uses both Oculus controllers to control a dual-arm robot system.
Left controller -> Left arm, Right controller -> Right arm.
Integrates placo IK solver for joint position tracking and robot client for state feedback.
"""

import logging
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
import threading
import webbrowser

from .base_teleop import BaseTeleop
from .config_teleop import OculusDualArmTeleopConfig
from .oculus.oculus_dual_arm_robot import OculusDualArmRobot
from .placo_visualization import PlacoVisualizer

logger = logging.getLogger(__name__)

# Import robot client
try:
    from lerobot_robot.dobot_interface_client import DobotDualArmClient
except ImportError:
    DobotDualArmClient = None

# Import placo
try:
    import placo
    from scipy.spatial.transform import Rotation as R
    PLACO_AVAILABLE = True
except ImportError:
    PLACO_AVAILABLE = False
    logger.warning("[TELEOP] placo not available")


class OculusDualArmTeleop(BaseTeleop):
    """
    Dual-arm teleoperation using both Oculus Quest controllers.
    
    This teleoperation mode uses both Oculus Quest controllers to simultaneously
    control two robot arms in Cartesian space. The output includes both delta pose
    and joint positions (via placo IK solver).
    
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
    
    # End-effector link configuration
    MANIPULATOR_CONFIG = {
        "left_arm": {
            "ee_link": "left_Link6",
            "base_link": "left_base_fixed_link",
        },
        "right_arm": {
            "ee_link": "right_Link6",
            "base_link": "right_base_fixed_link",
        },
    }
    
    def __init__(self, config: OculusDualArmTeleopConfig):
        super().__init__(config)
        self.oculus_robot: OculusDualArmRobot = None
        self.robot_client: Optional[Any] = None
        
        # Placo IK solver components
        self.placo_robot = None
        self.solver = None
        self.effector_task = {}
        self.placo_vis = None
        
        # Current robot state
        self.left_current_joints: np.ndarray = np.zeros(6)
        self.right_current_joints: np.ndarray = np.zeros(6)
        self.target_left_q: np.ndarray = np.zeros(6)
        self.target_right_q: np.ndarray = np.zeros(6)
        
        # URDF path: configurable via config, with a default relative to this file
        default_urdf_path = (
            Path(__file__).resolve().parent
            / "assets"
            / "dual_dobot"
            / "dual_nova5_robot.urdf"
        )
        self.robot_urdf_path = str(
            getattr(config, "robot_urdf_path", default_urdf_path)
        )
        
        # Threading
        self._stop_event = threading.Event()
        self._qpos_lock = threading.Lock()  # Lock for target joint positions (IK output)
        self._joint_lock = threading.Lock()  # Lock for current joint positions (from robot)
        
        # Visualization flag
        self.visualize = getattr(config, 'visualize_placo', False)
        
        # Joint reading frequency (used only when background polling is explicitly enabled)
        self.joint_read_fps = getattr(config, 'joint_read_fps', 100)
        # zerorpc/gevent client is not reliably safe in a native Python thread.
        # Keep this disabled by default to avoid "This operation would block forever".
        self.enable_joint_read_thread = getattr(config, 'enable_joint_read_thread', False)
    
    def _get_teleop_name(self) -> str:
        return "OculusDualArmTeleop"
    
    @property
    def action_features(self) -> dict:
        """Return action features for dual-arm oculus mode."""
        features = {}
        # Delta EE poses
        for arm in ["left", "right"]:
            for axis in ["x", "y", "z", "rx", "ry", "rz"]:
                features[f"{arm}_delta_ee_pose.{axis}"] = float
        
        # Joint positions (from IK)
        for arm in ["left", "right"]:
            for i in range(6):
                features[f"{arm}_joint_{i+1}.pos"] = float
        
        # Gripper commands
        if self.cfg.use_gripper:
            features["left_gripper_cmd_bin"] = float
            features["right_gripper_cmd_bin"] = float
        
        return features
    
    def _connect_impl(self) -> None:
        """Connect to Oculus Quest and initialize placo IK solver."""
        # Connect to Oculus
        self.oculus_robot = OculusDualArmRobot(
            ip=self.cfg.ip,
            use_gripper=self.cfg.use_gripper,
            left_pose_scaler=self.cfg.left_pose_scaler,
            left_channel_signs=self.cfg.left_channel_signs,
            right_pose_scaler=self.cfg.right_pose_scaler,
            right_channel_signs=self.cfg.right_channel_signs,
        )
        logger.info(f"[TELEOP] Oculus dual-arm connected at IP: {self.cfg.ip}")
        
        # Connect to robot client
        try:
            robot_ip = getattr(self.cfg, 'robot_ip', '127.0.0.1')
            robot_port = getattr(self.cfg, 'robot_port', 4242)
            self.robot_client = DobotDualArmClient(ip=robot_ip, port=robot_port)
            logger.info(f"[TELEOP] Robot client connected at {robot_ip}:{robot_port}")
        except Exception as e:
            logger.warning(f"[TELEOP] Failed to connect to robot client: {e}")
            self.robot_client = None
        
        # Initialize placo IK solver
        self._init_placo_solver()
        
        # Initialize joint positions from robot
        self._init_joint_positions()
        
        # Start visualization if enabled
        if self.visualize:
            self._start_placo_visualizer()
        
        # Optional background thread for reading robot state.
        # Disabled by default because zerorpc/gevent can fail in native threads.
        if self.enable_joint_read_thread:
            threading.Thread(target=self._start_joint_reading, daemon=True).start()
            logger.info("[TELEOP] Started background joint reading thread")
        else:
            logger.info("[TELEOP] Background joint reading thread disabled")
    
    def _init_placo_solver(self):
        """Initialize placo IK solver with frame tasks."""
        if not PLACO_AVAILABLE:
            raise RuntimeError("placo is required for IK solver")
        
        # Load robot model
        self.placo_robot = placo.RobotWrapper(str(self.robot_urdf_path))
        
        # Create IK solver
        self.solver = placo.KinematicsSolver(self.placo_robot)
        self.solver.dt = self.cfg.servo_time  # ~60Hz
        self.solver.mask_fbase(True)
        
        # # Add regularization for smooth motion
        # self.solver.add_kinetic_energy_regularization_task(1e-6)
        
        # # Add joint regularization to prevent jumping between IK solutions
        # joint_reg = self.solver.add_regularization_task(0.1)
        # joint_reg.configure("joint_regularization", "soft", 0.1)
        
        # # Enable joint limits
        # self.solver.enable_joint_limits(True)
        
        # Create frame tasks for both end-effectors
        initial_pose = np.eye(4)
        
        for arm_name, config in self.MANIPULATOR_CONFIG.items():
            self.effector_task[arm_name] = self.solver.add_frame_task(
                config["ee_link"], initial_pose
            )
            self.effector_task[arm_name].configure(f"{arm_name}_frame", "soft", 1.0)
        
        logger.info("[TELEOP] Placo IK solver initialized")
    
    def _init_joint_positions(self):
        """Initialize joint positions from robot."""
        if self.robot_client is not None:
            self.left_current_joints = self.robot_client.left_robot_get_joint_positions()
            self.right_current_joints = self.robot_client.right_robot_get_joint_positions()
        else:
            self.left_current_joints = np.zeros(6)
            self.right_current_joints = np.zeros(6)
        
        # Set initial joint positions in placo model
        # Joint indices: left arm q[7:13], right arm q[25:31]
        self.placo_robot.state.q[7:13] = self.left_current_joints
        self.placo_robot.state.q[25:31] = self.right_current_joints
        self.placo_robot.update_kinematics()
        
        # Initialize target joints
        self.target_left_q = self.left_current_joints.copy()
        self.target_right_q = self.right_current_joints.copy()
        
        logger.info(f"[TELEOP] Initial left joints: {self.left_current_joints}")
        logger.info(f"[TELEOP] Initial right joints: {self.right_current_joints}")
    
    def _sync_placo_to_robot_state(self):
        """
        Synchronize placo model state to current robot state.
        Called when reset is requested to ensure placo starts from the robot's reset position.
        
        IMPORTANT: This method triggers the robot reset FIRST, then syncs placo state.
        This ensures placo is synchronized to the POST-RESET joint positions.
        """
        if self.robot_client is not None:
            # First, trigger robot reset and wait for completion
            logger.info("[TELEOP] Triggering robot reset...")
            try:
                self.robot_client.robot_go_home()
                logger.info("[TELEOP] Robot reset completed")
            except Exception as e:
                logger.warning(f"[TELEOP] Robot reset failed: {e}")
            
            # Small delay to ensure robot state is updated
            time.sleep(0.1)
            
            # Get current joint positions from robot (after reset)
            self.left_current_joints = self.robot_client.left_robot_get_joint_positions()
            self.right_current_joints = self.robot_client.right_robot_get_joint_positions()
        else:
            # If no robot client, reset to zero
            self.left_current_joints = np.zeros(6)
            self.right_current_joints = np.zeros(6)
        
        # Update placo model state
        self.placo_robot.state.q[7:13] = self.left_current_joints
        self.placo_robot.state.q[25:31] = self.right_current_joints
        self.placo_robot.update_kinematics()
        
        # Update target joints to match
        with self._qpos_lock:
            self.target_left_q = self.left_current_joints.copy()
            self.target_right_q = self.right_current_joints.copy()
        
        logger.info(f"[TELEOP] Synced placo to robot state - left: {self.left_current_joints}")
        logger.info(f"[TELEOP] Synced placo to robot state - right: {self.right_current_joints}")
    
    def _start_placo_visualizer(self):
        """Start meshcat visualization."""
        try:
            self.placo_vis = PlacoVisualizer(self.placo_robot, auto_open=True)
            logger.info(f"[TELEOP] Visualization started at: {self.placo_vis.url()}")
        except Exception as e:
            logger.warning(f"[TELEOP] Failed to start visualization: {e}")
            self.placo_vis = None
    
    def _update_visualization(self):
        """Update visualization with current robot state."""
        if self.placo_vis is not None:
            try:
                self.placo_vis.display()
                # Update target frames
                self.placo_vis.update_target(
                    left_target=self.effector_task["left_arm"].T_world_frame,
                    right_target=self.effector_task["right_arm"].T_world_frame
                )
            except Exception as e:
                logger.warning(f"[TELEOP] Visualization update failed: {e}")
    
    def _start_joint_reading(self):
        """
        Background thread for continuously reading robot joint positions via zerorpc.
        
        NOTE: gevent/zerorpc can raise "This operation would block forever"
        when a client created in the main thread is used from a native thread.
        Keep this optional and disabled by default.
        """
        while not self._stop_event.is_set():
            try:
                start = time.perf_counter()
                
                if self.robot_client is not None:
                    # Read joint positions from robot (zerorpc call)
                    left_joints = self.robot_client.left_robot_get_joint_positions()
                    right_joints = self.robot_client.right_robot_get_joint_positions()
                    
                    # Thread-safe update
                    with self._joint_lock:
                        self.left_current_joints = left_joints
                        self.right_current_joints = right_joints
                
                elapsed = time.perf_counter() - start
                time.sleep(max(0, 1/self.joint_read_fps - elapsed))
            except Exception as e:
                logger.error(f"[TELEOP] Error in joint reading thread: {e}")
    
    def _update_robot_qpos(self, oculus_obs: Dict[str, Any]):
        """
        Update robot joint positions from Oculus input via IK solver.
        
        The placo model acts as a **virtual target** that accumulates deltas independently
        of the real robot. We do NOT reset placo's q to real joints each cycle — doing so
        would discard previously accumulated deltas and cause the robot to snap back to its
        initial position.
        
        Args:
            oculus_obs: Pre-fetched Oculus observations to avoid duplicate calls
        """
        # Check if reset was requested - sync placo state to robot's reset position
        if oculus_obs.get("reset_requested", False):
            self._sync_placo_to_robot_state()
            logger.info("[TELEOP] Reset requested - synced placo state to robot reset position")
            # Skip IK update this cycle since we just reset
            return
        
        # Extract delta poses from pre-fetched observations
        left_delta = np.array([
            oculus_obs[f"left_delta_ee_pose.{axis}"] 
            for axis in ["y", "x", "z", "ry", "rx", "rz"]
        ])
        right_delta = np.array([
            oculus_obs[f"right_delta_ee_pose.{axis}"] 
            for axis in ["y", "x", "z", "ry", "rx", "rz"]
        ])
        
        # Compute joint positions via placo IK
        try:
            # Get current EE poses from placo's own accumulated state (NOT real robot)
            T_left_ee = self.placo_robot.get_T_world_frame("left_Link6")
            T_right_ee = self.placo_robot.get_T_world_frame("right_Link6")
            
            # Compute target positions: current + delta
            left_target_pos = T_left_ee[:3, 3] + left_delta[:3]
            right_target_pos = T_right_ee[:3, 3] + right_delta[:3]
            
            # Apply rotation deltas directly on rotation matrices (no euler conversion)
            R_delta_left = R.from_rotvec(left_delta[3:])
            R_delta_right = R.from_rotvec(right_delta[3:])
            R_target_left = R_delta_left * R.from_matrix(T_left_ee[:3, :3])
            R_target_right = R_delta_right * R.from_matrix(T_right_ee[:3, :3])
            
            # Build target transforms
            T_left_target = np.eye(4)
            T_left_target[:3, 3] = left_target_pos
            T_left_target[:3, :3] = R_target_left.as_matrix()
            
            T_right_target = np.eye(4)
            T_right_target[:3, 3] = right_target_pos
            T_right_target[:3, :3] = R_target_right.as_matrix()
            
            # Set effector tasks
            self.effector_task["left_arm"].T_world_frame = T_left_target
            self.effector_task["right_arm"].T_world_frame = T_right_target
            
            # Solve IK — solve(True) updates placo_robot.state.q in-place,
            # so the accumulated state persists to the next cycle
            self.solver.solve(True)
            self.placo_robot.update_kinematics()
            
            # Extract joint positions
            with self._qpos_lock:
                self.target_left_q = self.placo_robot.state.q[7:13].copy()
                self.target_right_q = self.placo_robot.state.q[25:31].copy()
            
            # Update visualization
            if self.visualize:
                self._update_visualization()
            
        except RuntimeError as e:
            logger.warning(f"[TELEOP] IK solver failed: {e}")
        except Exception as e:
            logger.warning(f"[TELEOP] IK computation failed: {e}")
    
    def _disconnect_impl(self) -> None:
        """Disconnect from Oculus Quest and robot client."""
        self._stop_event.set()
        if self.robot_client is not None:
            try:
                self.robot_client.close()
                logger.info("[TELEOP] Robot client disconnected")
            except:
                pass
    
    def _get_action_impl(self) -> Dict[str, Any]:
        """
        Get current action from Oculus controllers.
        Updates robot state and computes IK synchronously because DobotDualArmClient
        uses gevent which is not thread-safe.
        
        Returns dict with:
            - left/right_delta_ee_pose.{x,y,z,rx,ry,rz}
            - left/right_joint_{1-6}.pos (IK computed, in radians)
            - left/right_gripper_cmd_bin
            - reset_requested
        """
        # Get delta pose from Oculus FIRST (only once!)
        oculus_obs = self.oculus_robot.get_observations()
        
        # Update robot qpos and compute IK (pass pre-fetched observations)
        self._update_robot_qpos(oculus_obs)
        
        # Extract delta poses for action
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
        
        # Add joint positions (thread-safe read)
        with self._qpos_lock:
            target_left_q = self.target_left_q.copy()
            target_right_q = self.target_right_q.copy()
        
        for i in range(6):
            action[f"left_joint_{i+1}.pos"] = float(target_left_q[i])
            action[f"right_joint_{i+1}.pos"] = float(target_right_q[i])
        
        # Add gripper commands
        if self.cfg.use_gripper:
            action["left_gripper_cmd_bin"] = oculus_obs.get("left_gripper_cmd_bin", 0.0)
            action["right_gripper_cmd_bin"] = oculus_obs.get("right_gripper_cmd_bin", 0.0)
        
        # Add reset flag
        action["reset_requested"] = oculus_obs.get("reset_requested", False)
        
        return action


def test_action_output(ip: str, hz: float, max_steps: int, use_gripper: bool, visualize: bool) -> None:
    """Test the action output from OculusDualArmTeleop."""
    logger.info("\n" + "="*60)
    logger.info("Testing Action Output with Placo IK")
    logger.info("="*60)
    
    cfg = OculusDualArmTeleopConfig(
        ip=ip,
        use_gripper=use_gripper,
        visualize_placo=True,
    )
    teleop = OculusDualArmTeleop(cfg)

    sleep_s = 1.0 / max(hz, 1e-6)
    step = 0
    try:
        teleop.connect()
        logger.info(f"[TELEOP] Connected to Oculus at {ip}")
        logger.info(f"[TELEOP] Running at {hz} Hz, max_steps={max_steps}")
        logger.info("")
        
        while True:
            action = teleop.get_action()
            
            # Format action output nicely
            print(f"\n--- Step {step} ---")
            print("Delta EE Pose (left):  ", end="")
            for axis in ["x", "y", "z", "rx", "ry", "rz"]:
                print(f"{axis}={action[f'left_delta_ee_pose.{axis}']:.4f} ", end="")
            print()
            print("Delta EE Pose (right): ", end="")
            for axis in ["x", "y", "z", "rx", "ry", "rz"]:
                print(f"{axis}={action[f'right_delta_ee_pose.{axis}']:.4f} ", end="")
            print()
            print("Joint Positions (left):  [", end="")
            for i in range(6):
                print(f"{action[f'left_joint_{i+1}.pos']:.4f}", end=", " if i < 5 else "")
            print("]")
            print("Joint Positions (right): [", end="")
            for i in range(6):
                print(f"{action[f'right_joint_{i+1}.pos']:.4f}", end=", " if i < 5 else "")
            print("]")
            if use_gripper:
                print(f"Gripper (L/R): {action['left_gripper_cmd_bin']:.2f} / {action['right_gripper_cmd_bin']:.2f}")
            print(f"Reset requested: {action['reset_requested']}")
            
            step += 1
            if max_steps > 0 and step >= max_steps:
                break
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        teleop.disconnect()


def main() -> None:
    """CLI test entrypoint for OculusDualArmTeleop."""
    parser = argparse.ArgumentParser(description="Test OculusDualArmTeleop with Placo IK")
    parser.add_argument("--ip", type=str, default="192.168.110.62", help="Oculus Quest IP")
    parser.add_argument("--hz", type=float, default=10.0, help="Print frequency")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N steps; 0 means run forever")
    parser.add_argument("--no-gripper", action="store_true", help="Disable gripper fields")
    parser.add_argument("--visualize", action="store_true", help="Enable placo visualization")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    
    test_action_output(args.ip, args.hz, args.max_steps, not args.no_gripper, args.visualize)


if __name__ == "__main__":
    main()
