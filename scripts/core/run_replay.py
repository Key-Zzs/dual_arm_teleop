import time
import yaml
import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
from pathlib import Path
from typing import Dict, Any
from robots import SUPPORTED_ROBOTS, create_robot_config, create_robot
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

class ReplayConfig:
    def __init__(self, cfg: Dict[str, Any]):
        robot = cfg["robot"]

        # global config
        self.dataset_name: str = cfg["dataset_name"]
        self.episode_idx: int = int(cfg.get("episode_idx", 0))

        # robot config
        # Support both `robot_ip` and legacy `ip` in YAML replay config.
        self.robot_ip: str = robot.get("robot_ip", robot.get("ip", "localhost"))
        self.robot_port: int = robot.get("robot_port", 4242)
        self.control_mode: str = cfg.get("control_mode", "oculus")
        
        # Robot type selection (default to dobot_dual_arm for backward compatibility)
        self.robot_type: str = cfg.get("robot_type", "dobot_dual_arm")
        if self.robot_type not in SUPPORTED_ROBOTS:
            raise ValueError(
                f"Unsupported robot type: {self.robot_type}. "
                f"Supported types: {SUPPORTED_ROBOTS}"
            )

def run_replay(replay_cfg: ReplayConfig):
    episode_idx = replay_cfg.episode_idx

    robot_config = create_robot_config(
        replay_cfg.robot_type,
        robot_ip=replay_cfg.robot_ip,
        robot_port=replay_cfg.robot_port,
        debug=False,
        control_mode=replay_cfg.control_mode
    )
    
    robot = create_robot(replay_cfg.robot_type, robot_config)
    robot.connect()
    dataset = LeRobotDataset(replay_cfg.dataset_name)
    episode_indices = dataset.hf_dataset["episode_index"]
    selected_frame_indices = [
        i for i, ep in enumerate(episode_indices) if int(ep) == episode_idx
    ]

    if not selected_frame_indices:
        available_eps = sorted({int(ep) for ep in episode_indices})
        raise ValueError(
            f"Episode index {episode_idx} not found in dataset {replay_cfg.dataset_name}. "
            f"Available episodes: {available_eps[:20]}"
            + ("..." if len(available_eps) > 20 else "")
        )

    action_names = dataset.features["action"]["names"]
    log_say(
        f"Replaying episode {episode_idx} with {len(selected_frame_indices)} frames"
    )
    for frame_idx in selected_frame_indices:
        t0 = time.perf_counter()
        action_vec = dataset.hf_dataset[frame_idx]["action"]
        action = {
            name: float(action_vec[i]) for i, name in enumerate(action_names)
        }
        # print(f"action: {action}")
        robot.send_action(action)

        busy_wait(1.0 / dataset.fps - (time.perf_counter() - t0))

    robot.disconnect()

def main():
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "record_cfg.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    replay_cfg = ReplayConfig(cfg["replay"])

    run_replay(replay_cfg)