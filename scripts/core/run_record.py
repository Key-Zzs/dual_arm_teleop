import yaml
from pathlib import Path
from typing import Dict, Any
import numpy as np
from scripts.utils.dataset_utils import generate_dataset_name, update_dataset_info
from robots import (
    SUPPORTED_ROBOTS,
    create_robot_config,
    create_robot,
)
from teleoperators import (
    OculusTeleopConfig,
    OculusTeleop,
)
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors
from lerobot.utils.visualization_utils import init_rerun
from lerobot.utils.control_utils import init_keyboard_listener
# from send2trash import send2trash
import shutil
import termios, sys
import time
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame
from lerobot.utils.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.action_delta import servo_delta_flag_for_action_delta_alignment
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.visualization_utils import log_rerun_data
from dataclasses import field

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

RUN_MIX_MOVEMENT_EPS = 1e-4
RUN_MIX_CHANGE_EPS = 5e-3
RUN_MIX_GRIPPER_SOFT_TAKEOVER_EPS = 0.1


def _annotate_policy_action_delta_execution(action: dict[str, Any], policy) -> None:
    action_delta_alignment = getattr(policy.config, "action_delta_alignment", "step_wise")
    action["action_delta_alignment"] = action_delta_alignment
    action["servo_delta"] = servo_delta_flag_for_action_delta_alignment(action_delta_alignment)


class RecordConfig:
    """Configuration class for recording sessions."""
    
    def __init__(self, cfg: Dict[str, Any]):
        storage = cfg["storage"]
        task = cfg["task"]
        time = cfg["time"]
        cam = cfg["cameras"]
        robot = cfg["robot"]
        policy = cfg["policy"]
        teleop = cfg["teleop"]
        
        # Global config
        self.repo_id: str = cfg["repo_id"]
        self.debug: bool = cfg.get("debug", True)
        self.fps: str = cfg.get("fps", 15)
        self.dataset_path: str = HF_LEROBOT_HOME / self.repo_id
        self.user_info: str = cfg.get("user_notes", None)
        self.run_mode: str = cfg.get("run_mode", "run_record")
        self.rename_map: dict[str, str] = field(default_factory=dict)
        # Finish behavior: by default reset to home and keep connection to avoid server stop on close.
        self.reset_on_finish: bool = cfg.get("reset_on_finish", True)
        self.disconnect_on_finish: bool = cfg.get("disconnect_on_finish", False)
        # Finish behavior: by default reset to home and keep connection to avoid server stop on close.
        self.reset_on_finish: bool = cfg.get("reset_on_finish", True)
        self.disconnect_on_finish: bool = cfg.get("disconnect_on_finish", False)
        
        # Robot type selection
        self.robot_type: str = cfg.get("robot_type", "dobot_dual_arm")
        if self.robot_type not in SUPPORTED_ROBOTS:
            raise ValueError(
                f"Unsupported robot type: {self.robot_type}. "
                f"Supported types: {SUPPORTED_ROBOTS}"
            )
        
        # Teleop config - parse based on control mode
        self.control_mode = teleop.get("control_mode", "oculus")
        self.dual_arm = teleop.get("dual_arm", True)
        self._parse_teleop_config(teleop)
        
        # Policy config
        self._parse_policy_config(policy)
        
        # Robot config
        self.robot_ip: str = robot.get("robot_ip", "localhost")
        self.robot_port: int = robot.get("robot_port", 4242)
        self.use_gripper: bool = robot["use_gripper"]
        self.close_threshold = robot.get("close_threshold", 0.5)
        self.gripper_reverse: bool = robot.get("gripper_reverse", False)
        self.gripper_max_open: float = robot.get("gripper_max_open", 0.085)
        self.gripper_force: float = robot.get("gripper_force", 10.0)
        self.gripper_speed: float = robot.get("gripper_speed", 0.1)
        
        # Task config
        self.num_episodes: int = task.get("num_episodes", 1)
        self.display: bool = task.get("display", True)
        self.task_description: str = task.get("description", "default task")
        self.resume: bool = task.get("resume", False)
        self.resume_dataset: str = task.get("resume_dataset", "")
        
        # Time config
        self.episode_time_sec: int = time.get("episode_time_sec", 60)
        self.reset_time_sec: int = time.get("reset_time_sec", 10)
        # save metadata period (number of episodes between metadata writes)
        # YAML uses `save_meta_period` — use the same name here.
        self.save_meta_period: int = time.get("save_meta_period", 1)
        
        # Cameras config (3 RealSense cameras: left wrist, right wrist, head)
        self.left_wrist_cam_serial: str = cam["left_wrist_cam_serial"]
        self.right_wrist_cam_serial: str = cam["right_wrist_cam_serial"]
        self.head_cam_serial: str = cam["head_cam_serial"]
        self.cam_width: int = cam["width"]
        self.cam_height: int = cam["height"]
        
        # Storage config
        self.push_to_hub: bool = storage.get("push_to_hub", False)
    
    def _parse_teleop_config(self, teleop: Dict[str, Any]) -> None:
        """Parse teleoperation configuration based on control mode."""
        if self.control_mode == "oculus":
            oculus_cfg = teleop.get("oculus_config", {})
            self.use_gripper = oculus_cfg.get("use_gripper", True)
            self.oculus_ip = oculus_cfg.get("ip", "192.168.110.62")
            self.pose_scaler = oculus_cfg.get("pose_scaler", [1.0, 1.0])
            self.channel_signs = oculus_cfg.get("channel_signs", [1, 1, 1, 1, 1, 1])
            self.visualize_placo = oculus_cfg.get("visualize_placo", False)
            self.action_smoothing_alpha = oculus_cfg.get("action_smoothing_alpha", 0.35)
            if self.dual_arm:
                self.left_pose_scaler = oculus_cfg.get("left_pose_scaler", self.pose_scaler)
                self.right_pose_scaler = oculus_cfg.get("right_pose_scaler", self.pose_scaler)
                self.left_channel_signs = oculus_cfg.get("left_channel_signs", self.channel_signs)
                self.right_channel_signs = oculus_cfg.get("right_channel_signs", self.channel_signs)
        
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}. Supported: oculus")
    
    def _parse_policy_config(self, policy: Dict[str, Any]) -> None:
        """Parse policy configuration."""
        def normalize_temporal_ensemble_coeff(value: Any) -> float | None:
            """Treat non-positive and None-like values as disabled temporal ensembling."""
            if value is None:
                return None

            if isinstance(value, str):
                text = value.strip().lower()
                if text in {"", "none", "null", "~"}:
                    return None
                try:
                    value = float(text)
                except ValueError as exc:
                    raise ValueError(
                        "`policy.temporal_ensemble_coeff` must be a number, null, or None-like string. "
                        f"Got: {value!r}"
                    ) from exc

            if isinstance(value, (int, float)):
                return float(value) if value > 0 else None

            raise ValueError(
                "`policy.temporal_ensemble_coeff` must be numeric or null-like. "
                f"Got type: {type(value).__name__}"
            )

        policy_type = policy["type"]
        # NOTE:
        # - "act" and "act_dagger" share the same ACT architecture at inference time.
        # - DAgger only changes data collection/training loop, not policy forward/select_action API.
        if policy_type in {"act", "act_dagger"}:
            from lerobot.policies import ACTConfig

            temporal_ensemble_coeff = normalize_temporal_ensemble_coeff(
                policy.get("temporal_ensemble_coeff")
            )
            self.policy = ACTConfig(
                device=policy["device"],
                push_to_hub=policy["push_to_hub"],
                temporal_ensemble_coeff=temporal_ensemble_coeff,
                chunk_size=policy.get("chunk_size", 100),
                n_action_steps=policy.get("n_action_steps", 100),
                action_delta_alignment=policy.get("action_delta_alignment", "step_wise"),
            )
        elif policy_type == "diffusion":
            from lerobot.policies import DiffusionConfig
            self.policy = DiffusionConfig(
                device=policy["device"],
                push_to_hub=policy["push_to_hub"],
            )
        else:
            raise ValueError(f"No config for policy type: {policy_type}")
        
        if policy.get("pretrained_path"):
            self.policy.pretrained_path = policy["pretrained_path"]
    
    def create_teleop_config(self):
        """Create teleoperation configuration object."""
        if self.control_mode == "oculus":
            if self.dual_arm:
                return OculusTeleopConfig(
                    use_gripper=self.use_gripper,
                    ip=self.oculus_ip,
                    left_pose_scaler=self.left_pose_scaler,
                    right_pose_scaler=self.right_pose_scaler,
                    left_channel_signs=self.left_channel_signs,
                    right_channel_signs=self.right_channel_signs,
                    action_smoothing_alpha=self.action_smoothing_alpha,
                    visualize_placo=self.visualize_placo,
                )
            return OculusTeleopConfig(
                use_gripper=self.use_gripper,
                ip=self.oculus_ip,
                pose_scaler=self.pose_scaler,
                channel_signs=self.channel_signs,
            )
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}. Supported: oculus")


def handle_incomplete_dataset(dataset_path):
    if dataset_path.exists():
        print(f"====== [WARNING] Detected an incomplete dataset folder: {dataset_path} ======")
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
        ans = input("Do you want to delete it? (y/n): ").strip().lower()
        if ans == "y":
            print(f"====== [DELETE] Removing folder: {dataset_path} ======")
            # Send to trash
            # send2trash(dataset_path)
            shutil.rmtree(dataset_path)
            print("====== [DONE] Incomplete dataset folder deleted successfully. ======")
        else:
            print("====== [KEEP] Incomplete dataset folder retained, please check manually. ======")


def _resolve_record_dataset_root(dataset_name: str, run_mode: str) -> Path:
    base = Path(HF_LEROBOT_HOME) / dataset_name
    return base


def _is_arm_override_active(
    teleop_raw_action: dict[str, Any],
    movement_eps: float = RUN_MIX_MOVEMENT_EPS,
) -> tuple[bool, str]:
    """Detect if teleop currently provides an arm/body override signal."""

    if bool(teleop_raw_action.get("reset_requested", False)):
        return True, "reset_requested"

    if bool(teleop_raw_action.get("left_grip_pressed", False)):
        return True, "left_grip_pressed"
    if bool(teleop_raw_action.get("right_grip_pressed", False)):
        return True, "right_grip_pressed"
    if bool(teleop_raw_action.get("is_expert_override", False)):
        return True, "is_expert_override"

    for key, value in teleop_raw_action.items():
        if key == "reset_requested":
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue

        lower_key = key.lower()
        if "delta" in lower_key or "pose" in lower_key:
            if abs(num) > movement_eps:
                return True, f"{key} motion={num:.4f}"

    return False, "no_arm_override_signal"


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip_gripper_cmd(value: float) -> float:
    return min(1.0, max(0.0, value))


def _reset_gripper_soft_takeover(state: dict[str, dict[str, Any]]) -> None:
    for arm_state in state.values():
        arm_state["active"] = False
        arm_state["hold"] = None
        arm_state["manual"] = False
        arm_state["ignore_until_released"] = False
        arm_state["waiting_logged"] = False


def _current_gripper_cmd(
    arm: str,
    raw_obs: dict[str, Any],
    last_exec_action: dict[str, Any] | None,
    fallback_action: dict[str, Any],
) -> float | None:
    key = f"{arm}_gripper_cmd"
    for source in (last_exec_action, raw_obs, fallback_action):
        if source is None or key not in source:
            continue
        value = _float_or_none(source[key])
        if value is not None:
            return _clip_gripper_cmd(value)
    return None


def _gripper_request_reason(
    arm: str,
    teleop_raw_action: dict[str, Any],
    last_teleop_raw_action: dict[str, Any] | None,
    state: dict[str, dict[str, Any]],
    change_eps: float = RUN_MIX_CHANGE_EPS,
) -> str | None:
    arm_state = state[arm]
    if bool(teleop_raw_action.get(f"{arm}_gripper_release_requested", False)):
        should_log_release = (
            arm_state["active"]
            or arm_state["manual"]
            or not arm_state.get("ignore_until_released", False)
        )
        arm_state["active"] = False
        arm_state["hold"] = None
        arm_state["manual"] = False
        arm_state["ignore_until_released"] = True
        arm_state["waiting_logged"] = False
        if should_log_release:
            logging.info(
                "[run_mix] %s gripper released to policy. waiting for trigger release before reacquire.",
                arm,
            )
        return None

    if arm_state.get("ignore_until_released", False):
        if bool(teleop_raw_action.get(f"{arm}_trigger_pressed", False)):
            return None
        arm_state["ignore_until_released"] = False

    if bool(teleop_raw_action.get(f"{arm}_trigger_pressed", False)):
        return f"{arm}_trigger_pressed"

    key = f"{arm}_gripper_cmd"
    if last_teleop_raw_action is None or key not in teleop_raw_action:
        return None

    current = _float_or_none(teleop_raw_action.get(key))
    previous = _float_or_none(last_teleop_raw_action.get(key))
    if current is None or previous is None:
        return None

    delta = current - previous
    if abs(delta) > change_eps:
        return f"{arm}_gripper_trigger_changed"
    return None


def _copy_arm_channels(target_action: dict[str, Any], expert_action: dict[str, Any]) -> None:
    for arm in ("left", "right"):
        for axis in ("x", "y", "z", "rx", "ry", "rz"):
            key = f"{arm}_delta_ee_pose.{axis}"
            if key in expert_action:
                target_action[key] = expert_action[key]

    if expert_action.get("reset_requested", False):
        target_action["reset_requested"] = True


def _clip_gripper_channels(action: dict[str, Any]) -> None:
    for arm in ("left", "right"):
        key = f"{arm}_gripper_cmd"
        value = _float_or_none(action.get(key))
        if value is not None:
            action[key] = _clip_gripper_cmd(value)


def _apply_gripper_channel_control(
    arm: str,
    target_action: dict[str, Any],
    expert_action: dict[str, Any],
    raw_obs: dict[str, Any],
    last_exec_action: dict[str, Any] | None,
    last_teleop_raw_action: dict[str, Any] | None,
    fallback_action: dict[str, Any],
    state: dict[str, dict[str, Any]],
    gripper_request_reason: str | None,
    hold_without_manual: bool,
) -> tuple[bool, str | None]:
    """Apply per-gripper teleop control without letting policy fight that gripper."""
    key = f"{arm}_gripper_cmd"
    if key not in target_action or key not in expert_action:
        return False, None

    arm_state = state[arm]
    if gripper_request_reason is not None:
        arm_state["manual"] = True

    if not arm_state["manual"]:
        if hold_without_manual:
            hold = _current_gripper_cmd(arm, raw_obs, last_exec_action, fallback_action)
            if hold is not None:
                target_action[key] = hold
            return False, None

        arm_state["hold"] = None
        arm_state["active"] = False
        arm_state["waiting_logged"] = False
        return False, None

    teleop_cmd = _float_or_none(expert_action[key])
    if teleop_cmd is None:
        return False, None
    teleop_cmd = _clip_gripper_cmd(teleop_cmd)

    if arm_state["hold"] is None:
        hold = _current_gripper_cmd(arm, raw_obs, last_exec_action, fallback_action)
        arm_state["hold"] = teleop_cmd if hold is None else hold

    hold = arm_state["hold"]
    if not arm_state["active"]:
        previous_teleop_cmd = None
        if last_teleop_raw_action is not None:
            previous_teleop_cmd = _float_or_none(last_teleop_raw_action.get(key))
            if previous_teleop_cmd is not None:
                previous_teleop_cmd = _clip_gripper_cmd(previous_teleop_cmd)

        takeover_matched = abs(teleop_cmd - hold) <= RUN_MIX_GRIPPER_SOFT_TAKEOVER_EPS
        if previous_teleop_cmd is not None:
            takeover_matched = takeover_matched or (
                abs(previous_teleop_cmd - hold) <= RUN_MIX_GRIPPER_SOFT_TAKEOVER_EPS
            )

        if takeover_matched:
            arm_state["active"] = True
            if arm_state["waiting_logged"]:
                logging.info(
                    "[run_mix] %s gripper soft takeover acquired. hold=%.3f teleop=%.3f",
                    arm,
                    hold,
                    teleop_cmd,
                )
        else:
            target_action[key] = hold
            if not arm_state["waiting_logged"]:
                logging.info(
                    "[run_mix] %s gripper soft takeover waiting. hold=%.3f teleop=%.3f",
                    arm,
                    hold,
                    teleop_cmd,
                )
                arm_state["waiting_logged"] = True
            return True, f"{arm}_gripper_waiting"

    target_action[key] = teleop_cmd
    return True, gripper_request_reason or f"{arm}_gripper_active"


def run_mix_record_loop(
    robot,
    teleop,
    policy,
    preprocessor,
    postprocessor,
    dataset: LeRobotDataset | None,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    events: dict,
    fps: int,
    control_time_s: int | float,
    single_task: str,
    display_data: bool,
) -> dict[str, int]:
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    start_episode_t = time.perf_counter()
    timestamp_s = 0.0
    last_teleop_raw_action: dict[str, Any] | None = None

    expert_exec_steps = 0
    policy_exec_steps = 0
    saved_expert_steps = 0

    last_action_source = "policy"
    last_exec_action: dict[str, Any] | None = None
    gripper_soft_takeover = {
        "left": {
            "active": False,
            "hold": None,
            "manual": False,
            "ignore_until_released": False,
            "waiting_logged": False,
        },
        "right": {
            "active": False,
            "hold": None,
            "manual": False,
            "ignore_until_released": False,
            "waiting_logged": False,
        },
    }
    while timestamp_s < control_time_s:
        loop_start_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        raw_obs = robot.get_observation()
        obs_processed = robot_observation_processor(raw_obs)
        observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # (1) Policy inference.
        policy_action = predict_action(
            observation=observation_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=single_task,
            robot_type=robot.robot_type,
        )
        policy_action_processed = make_robot_action(policy_action, dataset.features)
        _annotate_policy_action_delta_execution(policy_action_processed, policy)

        # (2) Default execute policy action. Teleop may override selected channels below.
        exec_action = dict(policy_action_processed)
        action_source = "policy"
        is_expert = False

        # (3) Expert override is split by channel: arm/body and grippers are independent.
        teleop_raw_action = teleop.get_action()
        is_arm_override, arm_override_reason = _is_arm_override_active(teleop_raw_action)
        gripper_request_reasons = {
            arm: _gripper_request_reason(
                arm,
                teleop_raw_action,
                last_teleop_raw_action,
                gripper_soft_takeover,
            )
            for arm in ("left", "right")
        }
        needs_teleop_action = (
            is_arm_override
            or any(reason is not None for reason in gripper_request_reasons.values())
            or any(arm_state["manual"] for arm_state in gripper_soft_takeover.values())
        )

        override_reasons: list[str] = []
        if needs_teleop_action:
            expert_action = dict(teleop_action_processor((teleop_raw_action, raw_obs)))
        else:
            expert_action = {}

        if is_arm_override:
            _copy_arm_channels(exec_action, expert_action)
            exec_action["action_delta_alignment"] = "step_wise"
            exec_action["servo_delta"] = True
            override_reasons.append(arm_override_reason)
            # Flush ACT chunk cache only for arm/body interventions. Gripper-only
            # control should not disturb policy arm motion.
            policy.reset()

        for arm, gripper_reason in gripper_request_reasons.items():
            gripper_overridden, gripper_override_reason = _apply_gripper_channel_control(
                arm=arm,
                target_action=exec_action,
                expert_action=expert_action,
                raw_obs=raw_obs,
                last_exec_action=last_exec_action,
                last_teleop_raw_action=last_teleop_raw_action,
                fallback_action=policy_action_processed,
                state=gripper_soft_takeover,
                gripper_request_reason=gripper_reason,
                hold_without_manual=is_arm_override,
            )
            if gripper_overridden and gripper_override_reason is not None:
                override_reasons.append(gripper_override_reason)

        if override_reasons:
            action_source = "expert"
            is_expert = True
            if last_action_source != "expert":
                logging.info(
                    "[run_mix] source->expert (teleop override). reason=%s",
                    ", ".join(override_reasons),
                )
        elif last_action_source == "expert":
            logging.info(
                "[run_mix] source->policy (no expert override detected). last_reason=%s",
                "no_channel_override_signal",
            )
            _reset_gripper_soft_takeover(gripper_soft_takeover)
        else:
            pass

        _clip_gripper_channels(exec_action)
        last_action_source = action_source

        logging.debug(
            "[run_mix] action_source=%s reason=%s"
            " left_grip=%s right_grip=%s reset=%s",
            action_source,
            ",".join(override_reasons) if override_reasons else "no_channel_override_signal",
            teleop_raw_action.get("left_grip_pressed", False),
            teleop_raw_action.get("right_grip_pressed", False),
            teleop_raw_action.get("reset_requested", False),
        )

        last_teleop_raw_action = teleop_raw_action

        # (4) Execute action.
        robot_action_to_send = robot_action_processor((exec_action, raw_obs))
        _ = robot.send_action(robot_action_to_send)
        last_exec_action = dict(exec_action)

        if action_source == "expert":
            expert_exec_steps += 1
        else:
            policy_exec_steps += 1

        # Record rule: only expert-labeled steps are saved.
        if dataset is not None and is_expert:
            action_frame = build_dataset_frame(dataset.features, exec_action, prefix=ACTION)
            frame = {
                **observation_frame,
                **action_frame,
                "action_source": action_source,
                "is_expert": np.array([True], dtype=np.bool_),
                "task": single_task,
            }
            dataset.add_frame(frame)
            saved_expert_steps += 1

        if display_data:
            log_rerun_data(observation=obs_processed, action=exec_action)

        dt_s = time.perf_counter() - loop_start_t
        busy_wait(1 / fps - dt_s)
        timestamp_s = time.perf_counter() - start_episode_t

    return {
        "expert_exec_steps": expert_exec_steps,
        "policy_exec_steps": policy_exec_steps,
        "saved_expert_steps": saved_expert_steps,
    }

def run_record(record_cfg: RecordConfig):
    print("====== [START] Starting recording ======")
    dataset_name = None
    dataset_root = None
    try:
        dataset_name, data_version = generate_dataset_name(record_cfg)
        dataset_root = _resolve_record_dataset_root(dataset_name, record_cfg.run_mode)

        # Check joint offsets
        # if not record_cfg.debug:
        #     check_joint_offsets(record_cfg)        
        
        # Create RealSenseCamera configurations (3 cameras: left wrist, right wrist, head)
        left_wrist_image_cfg = RealSenseCameraConfig(
                                        serial_number_or_name=record_cfg.left_wrist_cam_serial,
                                        fps=record_cfg.fps,
                                        width=record_cfg.cam_width,
                                        height=record_cfg.cam_height,
                                        color_mode=ColorMode.RGB,
                                        use_depth=False,
                                        rotation=Cv2Rotation.NO_ROTATION)

        right_wrist_image_cfg = RealSenseCameraConfig(
                                        serial_number_or_name=record_cfg.right_wrist_cam_serial,
                                        fps=record_cfg.fps,
                                        width=record_cfg.cam_width,
                                        height=record_cfg.cam_height,
                                        color_mode=ColorMode.RGB,
                                        use_depth=False,
                                        rotation=Cv2Rotation.NO_ROTATION)

        head_image_cfg = RealSenseCameraConfig(
                                        serial_number_or_name=record_cfg.head_cam_serial,
                                        fps=record_cfg.fps,
                                        width=record_cfg.cam_width,
                                        height=record_cfg.cam_height,
                                        color_mode=ColorMode.RGB,
                                        use_depth=False,
                                        rotation=Cv2Rotation.NO_ROTATION)

        # Create the robot and teleoperator configurations
        camera_config = {
            "left_wrist_image": left_wrist_image_cfg,
            "right_wrist_image": right_wrist_image_cfg,
            "head_image": head_image_cfg,
        }
        
        # Create teleop config using the new method
        teleop_config = record_cfg.create_teleop_config()
        
        # Create robot configuration dynamically based on robot_type
        robot_config = create_robot_config(
            record_cfg.robot_type,
            robot_ip=record_cfg.robot_ip,
            robot_port=record_cfg.robot_port,
            cameras=camera_config,
            debug=record_cfg.debug,
            use_gripper=record_cfg.use_gripper,
            gripper_max_open=record_cfg.gripper_max_open,
            gripper_force=record_cfg.gripper_force,
            gripper_speed=record_cfg.gripper_speed,
            close_threshold=record_cfg.close_threshold,
            gripper_reverse=record_cfg.gripper_reverse,
            control_mode=record_cfg.control_mode,
        )
        
        # Initialize the robot dynamically based on robot_type
        robot = create_robot(record_cfg.robot_type, robot_config)

        # Configure the dataset features
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
        dataset_features = {**action_features, **obs_features}
        if record_cfg.run_mode == "run_mix":
            # Extend dataset schema for DAgger mixed collection metadata.
            dataset_features["action_source"] = {"dtype": "string", "shape": (1,), "names": None}
            dataset_features["is_expert"] = {"dtype": "bool", "shape": (1,), "names": None}

        if record_cfg.run_mode == "run_mix":
            logging.info("====== [RUN_MIX] Mix mode config ======")
            logging.info(
                "[run_mix] robot_type=%s control_mode=%s fps=%s episode_time=%s reset_time=%s",
                record_cfg.robot_type,
                record_cfg.control_mode,
                record_cfg.fps,
                record_cfg.episode_time_sec,
                record_cfg.reset_time_sec,
            )
            logging.info(
                "[run_mix] policy=%s policy_device=%s pretrained_path=%s",
                type(record_cfg.policy).__name__,
                record_cfg.policy.device,
                record_cfg.policy.pretrained_path,
            )
            logging.info(
                "[run_mix] action_delta_alignment=%s servo_delta=%s",
                getattr(record_cfg.policy, "action_delta_alignment", "step_wise"),
                servo_delta_flag_for_action_delta_alignment(
                    getattr(record_cfg.policy, "action_delta_alignment", "step_wise")
                ),
            )
            logging.info(
                "[run_mix] teleop dual_arm=%s oculus_ip=%s left_scaler=%s left_signs=%s right_scaler=%s right_signs=%s",
                record_cfg.dual_arm,
                getattr(record_cfg, "oculus_ip", "n/a"),
                getattr(record_cfg, "left_pose_scaler", None),
                getattr(record_cfg, "left_channel_signs", None),
                getattr(record_cfg, "right_pose_scaler", None),
                getattr(record_cfg, "right_channel_signs", None),
            )
            logging.info(
                "[run_mix] override detection: movement_eps=%.4f change_eps=%.4f gripper_soft_takeover_eps=%.4f",
                RUN_MIX_MOVEMENT_EPS,
                RUN_MIX_CHANGE_EPS,
                RUN_MIX_GRIPPER_SOFT_TAKEOVER_EPS,
            )

        if record_cfg.resume:
            dataset = LeRobotDataset(
                dataset_name,
                root=dataset_root,
            )

            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer()
            sanity_check_dataset_robot_compatibility(dataset, robot, record_cfg.fps, dataset_features)
        else:
            # # Create the dataset
            dataset = LeRobotDataset.create(
                repo_id=dataset_name,
                fps=record_cfg.fps,
                features=dataset_features,
                robot_type=robot.name,
                root=dataset_root,
                use_videos=True,
                image_writer_threads=4,
            )
        # Set the episode metadata buffer size to 1, so that each episode is saved immediately
        dataset.meta.metadata_buffer_size = record_cfg.save_meta_period

        # Initialize keyboard listener.
        # Rerun visualization can introduce periodic stalls when transport is unstable,
        # so only initialize it when display is explicitly enabled.
        # Initialize keyboard listener.
        # Rerun visualization can introduce periodic stalls when transport is unstable,
        # so only initialize it when display is explicitly enabled.
        _, events = init_keyboard_listener()
        if record_cfg.display:
            init_rerun(session_name="recording")

        # Create processor
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
        preprocessor = None
        postprocessor = None

        # configure the teleop and policy
        if record_cfg.run_mode == "run_record":
            logging.info("====== [INFO] Running in teleoperation mode ======")
            teleop = OculusTeleop(teleop_config)
            policy = None
        elif record_cfg.run_mode == "run_policy":
            logging.info("====== [INFO] Running in policy mode ======")
            policy = make_policy(record_cfg.policy, ds_meta=dataset.meta)
            teleop = None
        elif record_cfg.run_mode == "run_mix":
            logging.info("====== [INFO] Running in mixed mode ======")
            policy = make_policy(record_cfg.policy, ds_meta=dataset.meta)
            teleop = OculusTeleop(teleop_config)
        else:
            raise ValueError(
                f"Unsupported run_mode: {record_cfg.run_mode}. "
                "Supported: run_record | run_policy | run_mix"
            )
        
        if policy is not None:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=record_cfg.policy,
                pretrained_path=record_cfg.policy.pretrained_path,
                dataset_stats=rename_stats(dataset.meta.stats, {}),  # 使用空字典作为rename_map
                preprocessor_overrides={
                    "device_processor": {"device": record_cfg.policy.device},
                    "rename_observations_processor": {"rename_map": {}},  # 使用空字典作为rename_map
                },
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        episode_idx = 0

        while episode_idx < record_cfg.num_episodes and not events["stop_recording"]:
            logging.info(f"====== [RECORD] Recording episode {episode_idx + 1} of {record_cfg.num_episodes} ======")
            if record_cfg.run_mode == "run_mix":
                mix_stats = run_mix_record_loop(
                    robot=robot,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    events=events,
                    fps=record_cfg.fps,
                    control_time_s=record_cfg.episode_time_sec,
                    single_task=record_cfg.task_description,
                    display_data=record_cfg.display,
                )
                logging.info(
                    "[run_mix] policy_exec_steps=%d expert_exec_steps=%d saved_expert_steps=%d",
                    mix_stats["policy_exec_steps"],
                    mix_stats["expert_exec_steps"],
                    mix_stats["saved_expert_steps"],
                )
            else:
                record_loop(
                    robot=robot,
                    events=events,
                    fps=record_cfg.fps,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dataset=dataset,
                    control_time_s=record_cfg.episode_time_sec,
                    single_task=record_cfg.task_description,
                    display_data=record_cfg.display,
                )

            rerecord_requested = bool(events["rerecord_episode"])
            if rerecord_requested:
                logging.info("Re-recording episode requested: discard current episode and enter reset state.")
                events["rerecord_episode"] = False
                # Left arrow also sets exit_early=True. Clear it here so reset phase does not exit immediately.
                events["exit_early"] = False
                if dataset.episode_buffer is not None:
                    dataset.clear_episode_buffer()
            elif record_cfg.run_mode == "run_mix":
                has_expert_frames = (
                    dataset.episode_buffer is not None and dataset.episode_buffer.get("size", 0) > 0
                )
                if has_expert_frames:
                    dataset.save_episode()
                else:
                    logging.warning(
                        "[run_mix] episode %d has no expert override frames; skip saving this episode.",
                        episode_idx + 1,
                    )
            else:
                dataset.save_episode()

            # Reset the environment between episodes, and also before a re-record attempt.
            if not events["stop_recording"] and (episode_idx < record_cfg.num_episodes - 1 or rerecord_requested):
                while True:
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)
                    user_input = input("====== [WAIT] Press Enter to reset the environment ======")
                    if user_input == "":
                        break  
                    else:
                        logging.info("====== [WARNING] Please press only Enter to continue ======")

                logging.info("====== [RESET] Resetting the environment ======")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=record_cfg.fps,
                    teleop=teleop,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    control_time_s=record_cfg.reset_time_sec,
                    single_task=record_cfg.task_description,
                    display_data=record_cfg.display,
                )

            if rerecord_requested:
                continue

            episode_idx += 1

        # Clean up
        logging.info("Stop recording")

        # Reset robot to home position at the end (same intent as pressing A in teleop).
        if record_cfg.reset_on_finish:
            try:
                robot.reset()
            except Exception as reset_err:
                logging.warning(f"[WARNING] reset_on_finish failed: {reset_err}")

        # Optional disconnect. For Nero, disconnect triggers client.close() -> robot_stop on server.
        if record_cfg.disconnect_on_finish:
            robot.disconnect()
        else:
            logging.info("[INFO] Skip robot.disconnect() to avoid stop/e-stop at session end.")

        if teleop is not None:
            teleop.disconnect()
        dataset.finalize()

        update_dataset_info(record_cfg, dataset_name, data_version)
        if record_cfg.push_to_hub:
            dataset.push_to_hub()

    except Exception as e:
        logging.info(f"====== [ERROR] {e} ======")
        dataset_path = dataset_root if dataset_root is not None else Path(HF_LEROBOT_HOME) / str(dataset_name)
        handle_incomplete_dataset(dataset_path)
        sys.exit(1)

    except KeyboardInterrupt:
        logging.info("\n====== [INFO] Ctrl+C detected, cleaning up incomplete dataset... ======")
        dataset_path = dataset_root if dataset_root is not None else Path(HF_LEROBOT_HOME) / str(dataset_name)
        handle_incomplete_dataset(dataset_path)
        sys.exit(1)


def main():
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "record_cfg.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    record_cfg = RecordConfig(cfg["record"])
    run_record(record_cfg)

if __name__ == "__main__":
    main()
