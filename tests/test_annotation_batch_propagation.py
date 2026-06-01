from __future__ import annotations

from types import SimpleNamespace

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.factory import (
    ACTION_ALIGNED_ANNOTATION_KEYS,
    add_annotation_delta_timestamps,
    resolve_delta_timestamps,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from lerobot.processor.converters import batch_to_transition, transition_to_batch
from lerobot.utils.constants import ACTION, OBS_STATE


ANNOTATION_WEIGHT = "annotation.keyframe_weight"
ANNOTATION_EVENT = "annotation.gripper_event"


def _policy_features(action_dim: int = 3) -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    return (
        {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,))},
        {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
    )


def _stats(action_dim: int = 3) -> dict[str, dict[str, torch.Tensor]]:
    return {
        OBS_STATE: {"mean": torch.zeros(7), "std": torch.ones(7)},
        ACTION: {
            "mean": torch.zeros(action_dim),
            "std": torch.ones(action_dim),
            "min": torch.full((action_dim,), -1.0),
            "max": torch.ones(action_dim),
        },
    }


def _act_config(action_dim: int = 3) -> ACTConfig:
    config = ACTConfig(chunk_size=4, n_action_steps=4, device="cpu", push_to_hub=False)
    config.input_features, config.output_features = _policy_features(action_dim)
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }
    return config


def _diffusion_config(action_dim: int = 3) -> DiffusionConfig:
    config = DiffusionConfig(horizon=8, n_action_steps=4, device="cpu", push_to_hub=False)
    config.input_features, config.output_features = _policy_features(action_dim)
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }
    return config


def _batch(batch_size: int, time_steps: int, action_dim: int = 3) -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.zeros(batch_size, 7, dtype=torch.float32),
        ACTION: torch.zeros(batch_size, time_steps, action_dim, dtype=torch.float32),
        "action_is_pad": torch.zeros(batch_size, time_steps, dtype=torch.bool),
        ANNOTATION_WEIGHT: torch.ones(batch_size, time_steps, dtype=torch.float32),
        ANNOTATION_EVENT: torch.zeros(batch_size, time_steps, dtype=torch.int64),
    }


def _assert_annotation_batch(processed: dict[str, torch.Tensor], batch_size: int, time_steps: int) -> None:
    assert processed[ANNOTATION_WEIGHT].shape == (batch_size, time_steps)
    assert processed[ANNOTATION_WEIGHT].dtype == torch.float32
    assert processed[ANNOTATION_EVENT].shape == (batch_size, time_steps)
    assert processed[ANNOTATION_EVENT].dtype == torch.int64


def test_converter_roundtrip_keeps_annotation_fields() -> None:
    batch = _batch(batch_size=2, time_steps=4)

    transition = batch_to_transition(batch)
    roundtrip = transition_to_batch(transition)

    _assert_annotation_batch(roundtrip, batch_size=2, time_steps=4)


def test_converter_squeezes_temporal_annotation_scalars() -> None:
    batch = _batch(batch_size=2, time_steps=4)
    batch[ANNOTATION_WEIGHT] = batch[ANNOTATION_WEIGHT].unsqueeze(-1).to(torch.float64)
    batch[ANNOTATION_EVENT] = batch[ANNOTATION_EVENT].unsqueeze(-1).to(torch.int32)

    transition = batch_to_transition(batch)
    roundtrip = transition_to_batch(transition)

    _assert_annotation_batch(roundtrip, batch_size=2, time_steps=4)


def test_act_processor_keeps_annotation_fields() -> None:
    config = _act_config()
    preprocessor, _ = make_act_pre_post_processors(config, dataset_stats=_stats())
    batch = _batch(batch_size=2, time_steps=config.chunk_size)

    processed = preprocessor(batch)

    _assert_annotation_batch(processed, batch_size=2, time_steps=config.chunk_size)
    assert ANNOTATION_WEIGHT not in preprocessor.steps[-1].features
    assert ANNOTATION_EVENT not in preprocessor.steps[-1].features


def test_diffusion_processor_keeps_annotation_fields() -> None:
    config = _diffusion_config()
    preprocessor, _ = make_diffusion_pre_post_processors(config, dataset_stats=_stats())
    batch = _batch(batch_size=2, time_steps=config.horizon)

    processed = preprocessor(batch)

    _assert_annotation_batch(processed, batch_size=2, time_steps=config.horizon)
    assert ANNOTATION_WEIGHT not in preprocessor.steps[-1].features
    assert ANNOTATION_EVENT not in preprocessor.steps[-1].features


def test_processor_accepts_batches_without_annotation_fields() -> None:
    config = _act_config()
    preprocessor, _ = make_act_pre_post_processors(config, dataset_stats=_stats())
    batch = _batch(batch_size=2, time_steps=config.chunk_size)
    batch.pop(ANNOTATION_WEIGHT)
    batch.pop(ANNOTATION_EVENT)

    processed = preprocessor(batch)

    assert not any(key.startswith("annotation.") for key in processed)


def test_annotation_temporal_query_follows_action_delta_indices() -> None:
    cfg = SimpleNamespace(
        action_delta_indices=[-1, 0, 1],
        observation_delta_indices=[-2, 0],
        reward_delta_indices=None,
    )
    ds_meta = SimpleNamespace(
        fps=10,
        features={
            OBS_STATE: {"dtype": "float32", "shape": (7,), "names": None},
            ACTION: {"dtype": "float32", "shape": (3,), "names": None},
            ANNOTATION_WEIGHT: {"dtype": "float32", "shape": (1,), "names": None},
            ANNOTATION_EVENT: {"dtype": "int64", "shape": (1,), "names": None},
        },
    )

    delta_timestamps = resolve_delta_timestamps(cfg, ds_meta)

    assert delta_timestamps[ACTION] == [-0.1, 0.0, 0.1]
    assert delta_timestamps[OBS_STATE] == [-0.2, 0.0]
    assert delta_timestamps[ANNOTATION_WEIGHT] == delta_timestamps[ACTION]
    assert delta_timestamps[ANNOTATION_EVENT] == delta_timestamps[ACTION]


def test_annotation_temporal_query_skips_missing_annotation_features() -> None:
    cfg = SimpleNamespace(
        action_delta_indices=[0, 1, 2],
        observation_delta_indices=None,
        reward_delta_indices=None,
    )
    ds_meta = SimpleNamespace(
        fps=20,
        features={ACTION: {"dtype": "float32", "shape": (3,), "names": None}},
    )

    delta_timestamps = resolve_delta_timestamps(cfg, ds_meta)

    assert delta_timestamps == {ACTION: [0.0, 0.05, 0.1]}
    assert not any(key.startswith("annotation.") for key in delta_timestamps)


def test_add_annotation_delta_timestamps_preserves_existing_entries() -> None:
    cfg = SimpleNamespace(action_delta_indices=[0, 1, 2])
    ds_meta = SimpleNamespace(
        fps=20,
        features={
            ANNOTATION_WEIGHT: {"dtype": "float32", "shape": (1,), "names": None},
            ANNOTATION_EVENT: {"dtype": "int64", "shape": (1,), "names": None},
        },
    )

    delta_timestamps = add_annotation_delta_timestamps({ANNOTATION_WEIGHT: [42.0]}, cfg, ds_meta)

    assert delta_timestamps[ANNOTATION_WEIGHT] == [42.0]
    assert delta_timestamps[ANNOTATION_EVENT] == [0.0, 0.05, 0.1]


def test_action_aligned_annotation_key_list_covers_exported_fields() -> None:
    assert set(ACTION_ALIGNED_ANNOTATION_KEYS) == {
        "annotation.keyframe_weight",
        "annotation.gripper_event",
        "annotation.left_keyframe_weight",
        "annotation.right_keyframe_weight",
        "annotation.left_gripper_event",
        "annotation.right_gripper_event",
    }


def test_dataset_padding_indices_align_annotation_with_action() -> None:
    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.meta = SimpleNamespace(episodes=[{"dataset_from_index": 0, "dataset_to_index": 3}])
    dataset.delta_indices = {
        ACTION: [0, 1, 2],
        ANNOTATION_WEIGHT: [0, 1, 2],
        ANNOTATION_EVENT: [0, 1, 2],
    }

    query_indices, padding = dataset._get_query_indices(idx=2, ep_idx=0)

    assert query_indices[ACTION] == [2, 2, 2]
    assert query_indices[ANNOTATION_WEIGHT] == query_indices[ACTION]
    assert query_indices[ANNOTATION_EVENT] == query_indices[ACTION]
    assert padding["action_is_pad"].tolist() == [False, True, True]
    assert padding[f"{ANNOTATION_WEIGHT}_is_pad"].tolist() == padding["action_is_pad"].tolist()
    assert padding[f"{ANNOTATION_EVENT}_is_pad"].tolist() == padding["action_is_pad"].tolist()


def test_annotation_is_not_in_policy_feature_inference() -> None:
    features = {
        OBS_STATE: {"dtype": "float32", "shape": (7,), "names": None},
        ACTION: {"dtype": "float32", "shape": (3,), "names": None},
        ANNOTATION_WEIGHT: {"dtype": "float32", "shape": (1,), "names": None},
        ANNOTATION_EVENT: {"dtype": "int64", "shape": (1,), "names": None},
    }

    policy_features = dataset_to_policy_features(features)

    assert set(policy_features) == {OBS_STATE, ACTION}
