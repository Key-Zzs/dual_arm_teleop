from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "src"))

try:
    import torch
except ModuleNotFoundError:
    print("keyframe sampler tests skipped: torch is not installed in this Python environment")
    raise SystemExit(0)

from lerobot.datasets.sampler import build_keyframe_weighted_sampler, compute_keyframe_sample_weights


def _use_tmp_hf_datasets_cache() -> None:
    cache_dir = Path("/tmp/hf-datasets-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    try:
        import datasets.config as datasets_config
    except ModuleNotFoundError:
        return
    datasets_config.HF_DATASETS_CACHE = cache_dir


class _FakeHFDataset:
    def __init__(self, data: dict[str, list[int | float]]) -> None:
        self.data = data
        self.features = {key: {} for key in data}
        self.column_names = list(data)

    def __len__(self) -> int:
        return len(next(iter(self.data.values())))

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]
        return {column: torch.as_tensor(values[key]) for column, values in self.data.items()}


class _FakeDataset:
    def __init__(
        self,
        *,
        episodes: list[tuple[int, int]],
        keyframe_weight: list[float] | None = None,
        gripper_event: list[int] | None = None,
    ) -> None:
        length = max(end for _, end in episodes)
        episode_index = [0] * length
        for ep_idx, (start, end) in enumerate(episodes):
            for idx in range(start, end):
                episode_index[idx] = ep_idx

        data: dict[str, list[int | float]] = {"episode_index": episode_index}
        features: dict[str, dict] = {"episode_index": {}}
        if keyframe_weight is not None:
            data["annotation.keyframe_weight"] = keyframe_weight
            features["annotation.keyframe_weight"] = {}
        if gripper_event is not None:
            data["annotation.gripper_event"] = gripper_event
            features["annotation.gripper_event"] = {}

        self.hf_dataset = _FakeHFDataset(data)
        self.meta = SimpleNamespace(
            features=features,
            episodes=[
                {"dataset_from_index": start, "dataset_to_index": end} for start, end in episodes
            ],
        )

    def __len__(self) -> int:
        return len(self.hf_dataset)


def test_keyframe_sample_weights_from_events_and_weight_threshold() -> None:
    dataset = _FakeDataset(
        episodes=[(0, 12)],
        keyframe_weight=[1.0, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        gripper_event=[0, 0, 0, 0, 2, 0, 0, 5, 0, 0, 1, 0],
    )

    weights = compute_keyframe_sample_weights(
        dataset,
        range(4),
        positive_sample_weight=3.0,
        normal_sample_weight=1.0,
        max_sample_weight=4.0,
    )

    assert weights is not None
    assert float(weights[0]) == 3.0  # annotation.keyframe_weight at index 2.
    assert float(weights[1]) == 3.0  # closing event 2 at index 4.
    assert float(weights[4]) == 3.0  # opening event 5 at index 7.
    assert float(weights[8]) == 3.0  # pre/post event 1 at index 10.

    no_pre_post_weights = compute_keyframe_sample_weights(
        dataset,
        range(4),
        include_pre_post_events=False,
        positive_sample_weight=3.0,
        normal_sample_weight=1.0,
        max_sample_weight=4.0,
    )
    assert no_pre_post_weights is not None
    assert float(no_pre_post_weights[8]) == 1.0

    clamped_weights = compute_keyframe_sample_weights(
        dataset,
        range(4),
        positive_sample_weight=10.0,
        normal_sample_weight=1.0,
        max_sample_weight=4.0,
    )
    assert clamped_weights is not None
    assert float(clamped_weights[1]) == 4.0


def test_keyframe_sample_weights_do_not_cross_episode_boundary() -> None:
    dataset = _FakeDataset(
        episodes=[(0, 3), (3, 6)],
        gripper_event=[0, 0, 0, 2, 0, 0],
    )

    weights = compute_keyframe_sample_weights(
        dataset,
        range(4),
        positive_sample_weight=3.0,
        normal_sample_weight=1.0,
        max_sample_weight=4.0,
    )

    assert weights is not None
    assert float(weights[1]) == 1.0
    assert float(weights[2]) == 1.0
    assert float(weights[3]) == 3.0


def test_keyframe_sample_weights_missing_annotation_fallback_and_error() -> None:
    dataset = _FakeDataset(episodes=[(0, 4)])

    weights = compute_keyframe_sample_weights(dataset, range(4), require_annotation=False)
    assert weights is None

    try:
        compute_keyframe_sample_weights(dataset, range(4), require_annotation=True)
    except ValueError as exc:
        assert "annotation columns are missing" in str(exc)
    else:
        raise AssertionError("require_annotation=True should raise when annotation columns are missing")


def test_keyframe_weighted_sampler_seed_is_reproducible() -> None:
    dataset = _FakeDataset(
        episodes=[(0, 12)],
        gripper_event=[0, 0, 0, 0, 2, 0, 0, 5, 0, 0, 0, 0],
    )
    cfg = {
        "enabled": True,
        "positive_sample_weight": 3.0,
        "normal_sample_weight": 1.0,
        "max_sample_weight": 4.0,
        "seed": 123,
    }

    first = build_keyframe_weighted_sampler(dataset, range(4), cfg)
    second = build_keyframe_weighted_sampler(dataset, range(4), cfg)

    assert first.sampler is not None
    assert second.sampler is not None
    assert list(iter(first.sampler)) == list(iter(second.sampler))


def test_keyframe_sampler_respects_eligible_indices() -> None:
    dataset = _FakeDataset(
        episodes=[(0, 8)],
        gripper_event=[0, 0, 0, 0, 2, 0, 0, 0],
    )

    result = build_keyframe_weighted_sampler(
        dataset,
        range(4),
        {
            "enabled": True,
            "positive_sample_weight": 3.0,
            "normal_sample_weight": 1.0,
            "max_sample_weight": 4.0,
            "seed": 123,
        },
        eligible_indices=[0, 1, 2, 3],
    )

    assert result.sampler is not None
    assert result.weights is not None
    assert result.stats["keyframe_sampler/eligible_sample_count"] == 4
    assert result.stats["keyframe_sampler/num_samples"] == 4
    assert float(result.weights[4]) == 0.0
    assert all(index in {0, 1, 2, 3} for index in list(iter(result.sampler)))


def test_real_annotated_dataset_smoke_if_present() -> None:
    root = Path(
        "/home/geist/.cache/huggingface/lerobot/nero_task3_step1/"
        "empty_merged_E113_gripper_annotated"
    )
    if not root.is_dir():
        return

    _use_tmp_hf_datasets_cache()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        print("real annotated dataset smoke skipped: datasets package is not installed")
        return

    dataset = LeRobotDataset(
        "nero_task3_step1/empty_merged_E113_gripper_annotated",
        root=root,
        download_videos=False,
    )
    result = build_keyframe_weighted_sampler(
        dataset,
        range(20),
        {
            "enabled": True,
            "require_annotation": False,
            "log_sampler_stats": True,
        },
    )

    assert result.weights is not None
    assert len(result.weights) == len(dataset)
    print("len(weights)", len(result.weights))
    print(
        "positive_sample_count",
        result.stats["keyframe_sampler/positive_sample_count"],
    )
    print(
        "positive_sample_ratio",
        result.stats["keyframe_sampler/positive_sample_ratio"],
    )
    print("mean_weight", result.stats["keyframe_sampler/mean_sample_weight"])
    print("max_weight", result.stats["keyframe_sampler/max_sample_weight"])


if __name__ == "__main__":
    test_keyframe_sample_weights_from_events_and_weight_threshold()
    test_keyframe_sample_weights_do_not_cross_episode_boundary()
    test_keyframe_sample_weights_missing_annotation_fallback_and_error()
    test_keyframe_weighted_sampler_seed_is_reproducible()
    test_keyframe_sampler_respects_eligible_indices()
    test_real_annotated_dataset_smoke_if_present()
    print("keyframe sampler tests passed")
