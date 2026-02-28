from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R

from oculus_reader import OculusReader
from robot import Robot


class OculusRobot(Robot):
    """
    A class representing a Oculus Quest 3/3s robot controller.
    
    Controls:
    - RG (Right Grip): Must be pressed to enable action recording
    - RTr (Right Trigger): Controls gripper (0.0 = open, 1.0 = closed)
    - Right controller pose: Controls end-effector delta pose
    """

    def __init__(
        self,
        ip: str = '192.168.110.62',
        use_gripper: bool = True,
        pose_scaler: Sequence[float] = [1.0, 1.0],
        channel_signs: Sequence[int] = [1, 1, 1, 1, 1, 1],
    ):  
        self._oculus_reader = OculusReader(ip_address=ip)
        self._use_gripper = use_gripper
        self._pose_scaler = pose_scaler
        self._channel_signs = channel_signs
        self._last_gripper_position = 1.0  # 默认夹爪张开状态
        self._last_valid_action = np.zeros(7 if use_gripper else 6)
        self._prev_pose = None  # 用于计算增量

        
    def num_dofs(self) -> int:
        if self._use_gripper:
            return 7
        else:
            return 6

    def _rotation_matrix_to_rotvec(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to rotation vector (axis-angle)."""
        rot = R.from_matrix(rotation_matrix)
        return rot.as_rotvec()

    def _compute_delta_pose(self, current_transform: np.ndarray) -> np.ndarray:
        """
        Compute delta pose from current transform.
        Returns [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz]
        """
        # Extract position (translation)
        position = current_transform[:3, 3]
        
        # Extract rotation matrix and convert to rotation vector
        rotation_matrix = current_transform[:3, :3]
        rotvec = self._rotation_matrix_to_rotvec(rotation_matrix)
        
        # If we have a previous pose, compute delta
        if self._prev_pose is not None:
            delta_position = position - self._prev_pose[:3]
            
            # Compute delta rotation: delta_rot = current_rot * prev_rot^-1
            prev_rot = R.from_rotvec(self._prev_pose[3:])
            curr_rot = R.from_rotvec(rotvec)
            delta_rot = curr_rot * prev_rot.inv()
            delta_rotvec = delta_rot.as_rotvec()
            
            delta_pose = np.concatenate([delta_position, delta_rotvec])
        else:
            # First frame, no delta
            delta_pose = np.zeros(6)
        
        # Update previous pose
        self._prev_pose = np.concatenate([position, rotvec])
        
        return delta_pose

    def get_action(self) -> np.ndarray:
        """
        Return the current robot actions including gripper control.
        
        Only returns valid action when RG button is pressed.
        Otherwise returns zero action (no movement).
        """
        transforms, buttons = self._oculus_reader.get_transformations_and_buttons()
        
        # Check if RG (Right Grip) is pressed - this enables action recording
        rg_pressed = buttons.get('RG', False)
        
        # Initialize delta pose
        delta_ee_pose = np.zeros(6)
        
        if 'r' in transforms:
            current_transform = transforms['r']  # 4x4 transformation matrix
            
            if rg_pressed:
                # Compute delta pose from the transform
                delta_ee_pose = self._compute_delta_pose(current_transform)
                
                # Apply scaling and signs
                if len(self._pose_scaler) >= 2:
                    position_scale = self._pose_scaler[0]  
                    orientation_scale = self._pose_scaler[1] 
                    
                    # Position scaling
                    delta_ee_pose[0] *= position_scale * self._channel_signs[0]  # x
                    delta_ee_pose[1] *= position_scale * self._channel_signs[1]  # y
                    delta_ee_pose[2] *= position_scale * self._channel_signs[2]  # z
                    
                    # Orientation scaling
                    delta_ee_pose[3] *= orientation_scale * self._channel_signs[3]  # rx
                    delta_ee_pose[4] *= orientation_scale * self._channel_signs[4]  # ry
                    delta_ee_pose[5] *= orientation_scale * self._channel_signs[5]  # rz
                
                # Store as last valid action
                self._last_valid_action[:6] = delta_ee_pose
            else:
                # RG not pressed, reset previous pose to avoid large jumps
                self._prev_pose = None
        else:
            # No right controller detected
            self._prev_pose = None

        # Handle gripper control with RTr (Right Trigger)
        if self._use_gripper:
            # RTr is a tuple like (0.0,) where 0.0 = not pressed, 1.0 = fully pressed
            right_trigger = buttons.get('rightTrig', (0.0,))
            if isinstance(right_trigger, tuple) and len(right_trigger) > 0:
                trigger_value = right_trigger[0]
            else:
                trigger_value = 0.0
            
            # Map trigger value to gripper position: 0.0 (not pressed) = open, 1.0 (pressed) = closed
            gripper_position = 1.0 - trigger_value  # Invert: trigger pressed = closed (0.0)
            
            self._last_gripper_position = gripper_position
            self._last_valid_action[6] = gripper_position
            
            return self._last_valid_action.copy()
        else:
            return delta_ee_pose

    def get_observations(self) -> Dict[str, np.ndarray]:
        """
        Return the current robot observations by formatting the action data.
        """
        action_data = self.get_action()
        
        obs_dict = {}
        axes = ["x", "y", "z", "rx", "ry", "rz"]
        
        if len(action_data) >= 6:
            for i, axis in enumerate(axes):
                obs_dict[f"delta_ee_pose.{axis}"] = float(action_data[i])
        else:
            for axis in axes:
                obs_dict[f"delta_ee_pose.{axis}"] = float(0.0)
        
        if self._use_gripper and len(action_data) >= 7:
            obs_dict["gripper_cmd_bin"] = float(action_data[6])
        else:
            obs_dict["gripper_cmd_bin"] = None
        
        return obs_dict


if __name__ == "__main__":
    import time
    
    # 创建 OculusRobot 实例
    oculus = OculusRobot(
        ip='192.168.110.62',  # 修改为你的 Oculus IP
        use_gripper=True,
        pose_scaler=[0.5, 0.5],  # 缩放因子
        channel_signs=[1, 1, 1, 1, 1, 1]
    )
    
    print("===== Oculus Robot Test =====")
    print("Controls:")
    print("  - RG (Right Grip): Press to enable action recording")
    print("  - RTr (Right Trigger): Control gripper (press = close)")
    print("  - Right controller: Move to control end-effector")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            action = oculus.get_action()
            obs = oculus.get_observations()
            
            # 打印 action
            print(f"\rAction: x={action[0]:+.4f} y={action[1]:+.4f} z={action[2]:+.4f} "
                  f"rx={action[3]:+.4f} ry={action[4]:+.4f} rz={action[5]:+.4f} "
                  f"gripper={action[6]:.2f}", end="")
            
            time.sleep(0.05)  # 20 Hz
            
    except KeyboardInterrupt:
        print("\n\n===== Test Ended =====")


