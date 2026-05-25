import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
from robots import (
    SUPPORTED_ROBOTS,
    create_robot_config,
    create_robot,
)
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _default_scripts_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_record_cfg_path() -> Path:
    return _default_scripts_dir() / "config" / "record_cfg.yaml"


def _load_record_cfg_yaml(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "record" not in cfg:
        raise ValueError(f"Reset config must contain a top-level `record` mapping: {cfg_path}")
    return cfg


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Reset a configured robot to its home position.")
    parser.add_argument(
        "--config",
        "--config-path",
        dest="config_path",
        type=Path,
        default=_default_record_cfg_path(),
        help="Path to record_cfg.yaml.",
    )
    args = parser.parse_args(argv)
    cfg = _load_record_cfg_yaml(args.config_path)

    robot_type = cfg["record"].get("robot_type", "dobot_dual_arm")
    robot_cfg = dict(cfg["record"]["robot"])
    robot_cfg["debug"] = False
    
    # 创建机器人配置
    robot_config = create_robot_config(
        robot_type=robot_type,
        **robot_cfg,
    )
    
    # 创建机器人实例并连接
    robot = create_robot(robot_type, robot_config)
    print("----------",robot.name)
    robot.connect()
    
    # 重置机器人到初始位置
    logging.info("Resetting robot to home position...")
    robot.reset()
    
    # 断开连接
    # robot.disconnect()
    logging.info("Robot reset completed successfully.")

if __name__ == "__main__":
    main()
