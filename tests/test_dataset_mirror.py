from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dual_arm_data_collection.lerobot_dual_arm_teleop.scripts.tools.mirror_dataset import (
    SliceSpec,
    apply_config_defaults,
    copy_dataset,
    mirror_pose,
    parse_args,
    transform_layout,
)


def _cfg() -> dict:
    return {
        "mirror": {"reflection_matrix": [1, 0, 0, 0, -1, 0, 0, 0, 1]},
        "joint_mirror": {
            "left_from_right_sign": [1, -1],
            "right_from_left_sign": [1, -1],
            "left_from_right_offset": [10, 20],
            "right_from_left_offset": [-10, -20],
        },
    }


def test_rotvec_pose_double_mirror_returns_original() -> None:
    s_mat = np.diag([1.0, -1.0, 1.0])
    pose = np.array([[0.2, -0.3, 0.4, 0.1, -0.2, 0.3]], dtype=np.float64)

    mirrored = mirror_pose(pose, "rotvec", s_mat)
    restored = mirror_pose(mirrored, "rotvec", s_mat)

    np.testing.assert_allclose(restored, pose, atol=1e-10)


def test_reflected_rotation_matrix_is_valid() -> None:
    s_mat = np.diag([1.0, -1.0, 1.0])
    rot = Rotation.from_rotvec([0.2, 0.3, -0.4]).as_matrix()
    mirrored = s_mat @ rot @ s_mat

    np.testing.assert_allclose(mirrored.T @ mirrored, np.eye(3), atol=1e-10)
    assert np.linalg.det(mirrored) == pytest.approx(1.0)


def test_joint_swap_sign_offset_and_gripper_swap() -> None:
    values = np.array([[1, 2, 3, 4, 0.25, 0.75]], dtype=np.float32)
    specs = [
        SliceSpec("left_joint", 0, 2, "joint_left"),
        SliceSpec("right_joint", 2, 4, "joint_right"),
        SliceSpec("left_gripper", 4, 5, "gripper_left"),
        SliceSpec("right_gripper", 5, 6, "gripper_right"),
    ]

    out = transform_layout(values, specs, _cfg())

    np.testing.assert_allclose(out[:, 0:2], [[13, 16]])
    np.testing.assert_allclose(out[:, 2:4], [[-9, -22]])
    np.testing.assert_allclose(out[:, 4:6], [[0.75, 0.25]])


def test_delta_pose_layout_swaps_and_mirrors() -> None:
    cfg = _cfg()
    values = np.array([[1, 2, 3, 0.1, 0.2, 0.3, 4, 5, 6, -0.1, -0.2, -0.3]], dtype=np.float64)
    specs = [
        SliceSpec("left_delta", 0, 6, "delta_ee_pose_left", "rotvec"),
        SliceSpec("right_delta", 6, 12, "delta_ee_pose_right", "rotvec"),
    ]

    out = transform_layout(values, specs, cfg)

    np.testing.assert_allclose(out[:, 0:3], [[4, -5, 6]], atol=1e-10)
    np.testing.assert_allclose(out[:, 6:9], [[1, -2, 3]], atol=1e-10)
    restored = transform_layout(out, specs, cfg)
    np.testing.assert_allclose(restored, values, atol=1e-10)


def test_image_horizontal_flip_twice_restores_and_wrist_swap() -> None:
    left = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    right = left + 20

    mirrored_left = np.flip(right, axis=1)
    mirrored_right = np.flip(left, axis=1)

    np.testing.assert_array_equal(np.flip(mirrored_right, axis=1), left)
    np.testing.assert_array_equal(np.flip(mirrored_left, axis=1), right)


def test_dst_exists_without_overwrite_refuses(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    with pytest.raises(FileExistsError):
        copy_dataset(src, dst, overwrite=False)


def test_config_defaults_and_cli_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_src = tmp_path / "config_src"
    config_dst = tmp_path / "config_dst"
    cli_dst = tmp_path / "cli_dst"
    config = {
        "dataset": {"src": str(config_src), "dst": str(config_dst)},
        "run": {
            "dry_run": False,
            "overwrite": False,
            "skip_video": True,
            "skip_fk_check": True,
            "validate_only": False,
            "num_vis_episodes": 3,
            "num_vis_frames_per_episode": 4,
        },
    }
    monkeypatch.setattr(
        "sys.argv",
        [
            "mirror_dataset.py",
            "--config",
            "mirror.yaml",
            "--dst",
            str(cli_dst),
            "--dry-run",
        ],
    )

    args = apply_config_defaults(parse_args(), config)

    assert args.src == config_src
    assert args.dst == cli_dst
    assert args.dry_run is True
    assert args.skip_video is True
    assert args.skip_fk_check is True
    assert args.num_vis_episodes == 3
    assert args.num_vis_frames_per_episode == 4
