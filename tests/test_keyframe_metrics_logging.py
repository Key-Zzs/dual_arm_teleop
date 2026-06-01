from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "src"))
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf-datasets-cache"

try:
    import torch
except ModuleNotFoundError:
    print("keyframe metrics logging tests skipped: torch is not installed in this Python environment")
    raise SystemExit(0)

import pytest

from lerobot.utils.keyframe_metrics import (
    compute_batch_annotation_metrics,
    scalarize_log_dict,
    summarize_annotation_distribution,
    to_jsonable,
    write_debug_json,
)


class _FakeHFDataset:
    def __init__(self, data: dict[str, list[int | float]]) -> None:
        self.data = data
        self.features = {key: {} for key in data}
        self.column_names = list(data)

    def __len__(self) -> int:
        return len(next(iter(self.data.values()))) if self.data else 0

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]
        return {column: torch.as_tensor(values[key]) for column, values in self.data.items()}


class _FakeDataset:
    def __init__(
        self,
        *,
        length: int,
        keyframe_weight: list[float] | None = None,
        gripper_event: list[int] | None = None,
    ) -> None:
        data: dict[str, list[int | float]] = {"episode_index": [0] * length}
        features: dict[str, dict] = {"episode_index": {}}
        if keyframe_weight is not None:
            data["annotation.keyframe_weight"] = keyframe_weight
            features["annotation.keyframe_weight"] = {}
        if gripper_event is not None:
            data["annotation.gripper_event"] = gripper_event
            features["annotation.gripper_event"] = {}

        self.hf_dataset = _FakeHFDataset(data)
        self.meta = SimpleNamespace(features=features)
        self.num_frames = length

    def __len__(self) -> int:
        return self.num_frames


def test_annotation_summary_counts_distribution_and_serializes() -> None:
    dataset = _FakeDataset(
        length=6,
        keyframe_weight=[1.0, 2.0, 6.0, 1.0, 6.0, 3.0],
        gripper_event=[0, 1, 2, 0, 5, 6],
    )

    summary = summarize_annotation_distribution(dataset)

    assert summary["annotation/has_annotation"] is True
    assert summary["annotation/num_frames"] == 6
    assert summary["annotation/keyframe_frame_count"] == 4
    assert summary["annotation/keyframe_frame_ratio"] == pytest.approx(4 / 6)
    assert summary["annotation/pre_closing_count"] == 1
    assert summary["annotation/closing_count"] == 1
    assert summary["annotation/opening_count"] == 1
    assert summary["annotation/post_opening_count"] == 1
    assert summary["annotation/mean_keyframe_weight"] == pytest.approx(4.25)
    json.dumps(to_jsonable(summary))


def test_annotation_summary_missing_annotation_fallback() -> None:
    dataset = _FakeDataset(length=4)

    summary = summarize_annotation_distribution(dataset)

    assert summary["annotation/has_annotation"] is False
    assert summary["annotation/num_frames"] == 4
    assert summary["annotation/keyframe_frame_count"] == 0
    json.dumps(to_jsonable(summary))


def test_batch_annotation_metrics_respect_padding_and_are_scalar() -> None:
    batch = {
        "action": torch.zeros(2, 4, 3),
        "action_is_pad": torch.tensor([[False, False, True, False], [False, False, False, True]]),
        "annotation.keyframe_weight": torch.tensor([[1.0, 6.0, 6.0, 2.0], [1.0, 3.0, 1.0, 6.0]]),
        "annotation.gripper_event": torch.tensor([[0, 2, 5, 1], [0, 5, 0, 2]]),
    }

    metrics = compute_batch_annotation_metrics(batch)

    assert metrics["batch/keyframe_ratio"] == pytest.approx(3 / 6)
    assert metrics["batch/opening_count"] == 1
    assert metrics["batch/closing_count"] == 1
    assert metrics["batch/normal_count"] == 3
    assert metrics["batch/valid_action_count"] == 6
    assert metrics["batch/padded_action_count"] == 2
    assert all(isinstance(value, (bool, int, float)) for value in metrics.values())


def test_sampler_stats_are_json_and_wandb_scalar_safe(tmp_path: Path) -> None:
    stats = {
        "keyframe_sampler/enabled": True,
        "keyframe_sampler/positive_sample_count": torch.tensor(3),
        "keyframe_sampler/positive_sample_ratio": torch.tensor(0.25),
        "keyframe_sampler/mean_sample_weight": 1.5,
        "keyframe_sampler/max_sample_weight": 3.0,
        "keyframe_sampler/normal_sample_weight": 1.0,
        "keyframe_sampler/positive_sample_weight": 3.0,
        "keyframe_sampler/annotation_missing": False,
        "keyframe_sampler/fallback_to_default": False,
        "keyframe_sampler/disabled_reason": None,
    }

    scalar_stats = scalarize_log_dict(stats)
    assert scalar_stats["keyframe_sampler/positive_sample_count"] == 3
    assert scalar_stats["keyframe_sampler/positive_sample_ratio"] == pytest.approx(0.25)
    assert "keyframe_sampler/disabled_reason" not in scalar_stats

    path = write_debug_json(tmp_path, "keyframe_sampler_summary.json", stats)
    payload = json.loads(path.read_text())
    assert payload["keyframe_sampler/positive_sample_count"] == 3


def test_real_annotated_dataset_summary_smoke_if_available() -> None:
    dataset_root = Path(
        "/home/geist/.cache/huggingface/lerobot/nero_task3_step1/empty_merged_E113_gripper_annotated"
    )
    if not dataset_root.exists():
        pytest.skip(f"annotated dataset smoke skipped; path does not exist: {dataset_root}")

    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        "nero_task3_step1/empty_merged_E113_gripper_annotated",
        root=dataset_root,
    )
    summary = summarize_annotation_distribution(dataset)
    assert summary["annotation/has_annotation"] is True
    assert summary["annotation/keyframe_frame_count"] >= 0
