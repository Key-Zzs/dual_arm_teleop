from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TELEOP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TELEOP_ROOT.parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(TELEOP_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION
from scripts.core.run_dagger_export import export_dagger_dataset


ACTION_FEATURE = {"dtype": "float32", "shape": (2,), "names": ["move", "gripper"]}
TRAIN_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (1,), "names": ["state"]},
    ACTION: ACTION_FEATURE,
}
RAW_FEATURES = {
    **TRAIN_FEATURES,
    "policy_action": ACTION_FEATURE,
    "expert_action": ACTION_FEATURE,
    "sent_action": ACTION_FEATURE,
    "action_source": {"dtype": "string", "shape": (1,), "names": None},
    "is_expert": {"dtype": "bool", "shape": (1,), "names": None},
    "intervention_segment_id": {"dtype": "int64", "shape": (1,), "names": None},
    "frame_role": {"dtype": "string", "shape": (1,), "names": None},
    "expert_label_complete": {"dtype": "bool", "shape": (1,), "names": None},
    "expert_action_missing": {"dtype": "string", "shape": (1,), "names": None},
    "success": {"dtype": "bool", "shape": (1,), "names": None},
}


def _action(base: float, step: int) -> np.ndarray:
    return np.asarray([base + step, step % 2], dtype=np.float32)


def _raw_frame(step: int, role: str, segment_id: int, success: bool) -> dict:
    sent_action = _action(10.0, step)
    expert_action = _action(100.0, step)
    return {
        "observation.state": np.asarray([float(step)], dtype=np.float32),
        ACTION: sent_action,
        "policy_action": _action(0.0, step),
        "expert_action": expert_action,
        "sent_action": sent_action,
        "action_source": "expert" if role in {"takeover_start", "recovery", "reset"} else "policy",
        "is_expert": np.asarray([role in {"takeover_start", "recovery", "reset"}], dtype=np.bool_),
        "intervention_segment_id": np.asarray([segment_id], dtype=np.int64),
        "frame_role": role,
        "expert_label_complete": np.asarray([role in {"takeover_start", "recovery"}], dtype=np.bool_),
        "expert_action_missing": "",
        "success": np.asarray([success], dtype=np.bool_),
        "task": "synthetic task",
    }


def _make_seed_dataset(root: Path) -> LeRobotDataset:
    dataset = LeRobotDataset.create(
        repo_id="seed",
        fps=10,
        features=TRAIN_FEATURES,
        root=root,
        robot_type="test_bot",
        use_videos=False,
    )
    for step in range(2):
        dataset.add_frame(
            {
                "observation.state": np.asarray([float(step)], dtype=np.float32),
                ACTION: _action(1.0, step),
                "task": "synthetic task",
            }
        )
    dataset.save_episode()
    dataset.finalize()
    return dataset


def _make_raw_dataset(root: Path) -> LeRobotDataset:
    dataset = LeRobotDataset.create(
        repo_id="raw",
        fps=10,
        features=RAW_FEATURES,
        root=root,
        robot_type="test_bot",
        use_videos=False,
    )

    for step, role in enumerate(["policy", "policy", "policy", "policy"]):
        dataset.add_frame(_raw_frame(step, role, -1, success=True))
    dataset.save_episode()

    for step, (role, segment_id) in enumerate(
        [("takeover_start", 0), ("policy", -1), ("takeover_start", 1)],
        start=4,
    ):
        dataset.add_frame(_raw_frame(step, role, segment_id, success=False))
    dataset.save_episode()

    for step, role in enumerate(["policy", "reset", "policy"], start=7):
        dataset.add_frame(_raw_frame(step, role, 2 if role == "reset" else -1, success=True))
    dataset.save_episode()
    dataset.finalize()
    return dataset


def _export(tmp_path: Path, mode: str, **kwargs) -> tuple[dict, LeRobotDataset]:
    seed_root = tmp_path / "seed"
    raw_root = tmp_path / "raw"
    output_root = tmp_path / f"export_{mode}"
    _make_seed_dataset(seed_root)
    _make_raw_dataset(raw_root)

    summary = export_dagger_dataset(
        raw_repo_id="raw",
        raw_root=raw_root,
        seed_repo_id="seed",
        seed_root=seed_root,
        output_repo_id=f"export_{mode}",
        output_root=output_root,
        export_mode=mode,
        min_segment_frames=1,
        min_episode_len_for_act=1,
        gripper_action_indices=[1],
        overwrite=True,
        **kwargs,
    )
    return summary, LeRobotDataset(f"export_{mode}", root=output_root)


def test_intervention_segments_legacy_label_source(tmp_path: Path) -> None:
    summary, exported = _export(tmp_path, "intervention_segments")

    assert summary["exported_dataset"]["total_episodes"] == 2
    assert summary["exported_dataset"]["total_frames"] == 2
    assert summary["rules"]["label_source"] == "expert_action"
    np.testing.assert_allclose(exported[0][ACTION].numpy(), _action(100.0, 4))


def test_full_success_episode_exports_sent_action_and_skips_reset(tmp_path: Path) -> None:
    summary, exported = _export(tmp_path, "full_success_episode")

    assert summary["stats"]["full_episode_exported"] == 1
    assert summary["stats"]["full_episode_skipped_no_success"] == 1
    assert summary["stats"]["full_episode_skipped_reset"] == 1
    assert summary["exported_dataset"]["total_frames"] == 4
    np.testing.assert_allclose(exported[0][ACTION].numpy(), _action(10.0, 0))


def test_hybrid_exports_full_and_limits_recovery_ratio(tmp_path: Path) -> None:
    summary, exported = _export(
        tmp_path,
        "hybrid",
        hybrid={
            "include_full_success_episode": True,
            "include_intervention_segments": True,
            "max_recovery_segments_per_episode": 10,
            "max_recovery_frames_ratio": 0.25,
            "prefer_full_episode_when_duplicate": True,
        },
    )

    assert summary["stats"]["hybrid_full_episodes"] == 1
    assert summary["stats"]["hybrid_recovery_segments"] == 1
    assert summary["stats"]["intervention_segments_skipped_ratio_cap"] == 1
    assert summary["stats"]["hybrid_recovery_ratio"] == 0.25
    assert summary["exported_dataset"]["total_frames"] == 5
    np.testing.assert_allclose(exported[4][ACTION].numpy(), _action(100.0, 4))


if __name__ == "__main__":
    import tempfile

    tmp_root = Path(tempfile.mkdtemp(prefix="dagger_export_modes_"))
    test_intervention_segments_legacy_label_source(tmp_root / "intervention")
    test_full_success_episode_exports_sent_action_and_skips_reset(tmp_root / "full")
    test_hybrid_exports_full_and_limits_recovery_ratio(tmp_root / "hybrid")
    print("dagger export mode smoke tests passed")
