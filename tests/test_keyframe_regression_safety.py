from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import draccus
import pytest
import torch
import yaml
from torch.utils.data._utils.collate import default_collate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT))

from lerobot.configs.train import TrainPipelineConfig as CoreTrainPipelineConfig
from lerobot.datasets.sampler import EpisodeAwareSampler, build_keyframe_weighted_sampler
from lerobot.policies.act.configuration_act import ACTConfig, ACTLossWeightingConfig
from lerobot.policies.act.modeling_act import (
    compute_unweighted_action_l1_loss,
    compute_weighted_action_l1_loss,
)
from lerobot.policies.diffusion.configuration_diffusion import (
    DiffusionConfig,
    DiffusionLossWeightingConfig,
)
from lerobot.policies.diffusion.modeling_diffusion import (
    compute_unweighted_denoising_mse_loss,
    compute_weighted_denoising_mse_loss,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.keyframe_metrics import (
    compute_batch_annotation_metrics,
    log_training_debug_startup,
    normalize_debug_metrics_config,
)
from scripts.core.policy_config_utils import build_policy_config, load_policy_yaml
from scripts.core.run_train import TrainPipelineConfig as DualArmTrainPipelineConfig

ANNOTATION_WEIGHT = "annotation.keyframe_weight"
ANNOTATION_EVENT = "annotation.gripper_event"
ACT_POLICY_CFG = PROJECT_ROOT / "scripts/config/policy_config/act_train_config.yaml"
DP_POLICY_CFG = PROJECT_ROOT / "scripts/config/policy_config/diffusion_train_config.yaml"
TRAIN_CFG = PROJECT_ROOT / "scripts/config/train_cfg.yaml"
REAL_DATASET_ROOT = Path("/home/geist/.cache/huggingface/lerobot/nero_task3_step1/empty_merged_E113")
REAL_ANNOTATED_ROOT = Path(
    "/home/geist/.cache/huggingface/lerobot/nero_task3_step1/empty_merged_E113_gripper_annotated"
)


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
            data[ANNOTATION_WEIGHT] = keyframe_weight
            features[ANNOTATION_WEIGHT] = {}
        if gripper_event is not None:
            data[ANNOTATION_EVENT] = gripper_event
            features[ANNOTATION_EVENT] = {}

        self.hf_dataset = _FakeHFDataset(data)
        self.meta = SimpleNamespace(
            features=features,
            episodes=[
                {"dataset_from_index": start, "dataset_to_index": end} for start, end in episodes
            ],
        )

    def __len__(self) -> int:
        return len(self.hf_dataset)


def _write_yaml(tmp_path: Path, name: str, payload: dict[str, Any]) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _use_tmp_hf_datasets_cache() -> None:
    cache_dir = Path("/tmp/hf-datasets-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    try:
        import datasets.config as datasets_config
    except ModuleNotFoundError:
        return
    datasets_config.HF_DATASETS_CACHE = cache_dir


def _strip_video_features(dataset: Any) -> None:
    dataset.meta.info["features"] = {
        key: feature
        for key, feature in dataset.meta.info["features"].items()
        if feature.get("dtype") != "video"
    }


def _load_real_dataset(root: Path, *, include_annotation: bool, action_steps: int = 4):
    if not root.is_dir():
        pytest.skip(f"real dataset smoke skipped; path does not exist: {root}")

    _use_tmp_hf_datasets_cache()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    delta = [step / info["fps"] for step in range(action_steps)]
    delta_timestamps = {ACTION: delta}
    if include_annotation:
        delta_timestamps[ANNOTATION_WEIGHT] = delta
        delta_timestamps[ANNOTATION_EVENT] = delta

    dataset = LeRobotDataset(
        f"nero_task3_step1/{root.name}",
        root=root,
        delta_timestamps=delta_timestamps,
        download_videos=False,
    )
    _strip_video_features(dataset)
    return dataset


def _collated_batch(dataset: Any, indices: list[int]) -> dict[str, Any]:
    return default_collate([dataset[index] for index in indices])


def _assert_act_and_dp_loss_smoke(batch: dict[str, Any], *, has_annotation: bool) -> None:
    action = batch[ACTION].to(dtype=torch.float32)
    pred = action + 0.1
    action_is_pad = batch.get("action_is_pad")
    if action_is_pad is None:
        action_is_pad = torch.zeros(action.shape[:2], dtype=torch.bool)

    act_cfg = ACTConfig(device="cpu", push_to_hub=False)
    act_cfg.loss_weighting.gripper_dim_indices = [12, 13]
    act_disabled = compute_unweighted_action_l1_loss(pred, action, action_is_pad)
    assert torch.isfinite(act_disabled)

    act_cfg.loss_weighting.enabled = True
    act_enabled, act_metrics = compute_weighted_action_l1_loss(
        pred,
        action,
        action_is_pad,
        batch,
        act_cfg,
    )
    assert torch.isfinite(act_enabled)
    if has_annotation:
        assert "loss/act_action_l1_weighted" in act_metrics
        assert "loss/mean_annotation_weight" in act_metrics

    dp_cfg = DiffusionConfig(device="cpu", push_to_hub=False, do_mask_loss_for_padding=True)
    dp_cfg.loss_weighting.gripper_dim_indices = [12, 13]
    dp_disabled = compute_unweighted_denoising_mse_loss(pred, action, action_is_pad, dp_cfg)
    assert torch.isfinite(dp_disabled)

    dp_cfg.loss_weighting.enabled = True
    dp_enabled, dp_metrics = compute_weighted_denoising_mse_loss(
        pred,
        action,
        action_is_pad,
        batch,
        dp_cfg,
    )
    assert torch.isfinite(dp_enabled)
    if has_annotation:
        assert "loss/dp_denoising_mse_weighted" in dp_metrics
        assert "loss/mean_annotation_weight" in dp_metrics


def test_act_old_yaml_without_loss_weighting_loads_disabled_defaults(tmp_path: Path) -> None:
    policy_yaml = load_policy_yaml(ACT_POLICY_CFG)
    policy_yaml.pop("loss_weighting", None)
    policy_yaml["device"] = "cpu"
    old_yaml_path = _write_yaml(tmp_path, "old_act_config.yaml", policy_yaml)

    parsed_cfg = draccus.parse(ACTConfig, old_yaml_path, args=[])
    built_cfg = build_policy_config("act", load_policy_yaml(old_yaml_path), config_path=old_yaml_path)

    for cfg in (parsed_cfg, built_cfg):
        assert isinstance(cfg.loss_weighting, ACTLossWeightingConfig)
        assert cfg.loss_weighting.enabled is False
        assert cfg.loss_weighting.keyframe_weight_column == ANNOTATION_WEIGHT
        assert cfg.loss_weighting.gripper_event_column == ANNOTATION_EVENT


def test_diffusion_old_yaml_without_loss_weighting_loads_disabled_defaults(tmp_path: Path) -> None:
    policy_yaml = load_policy_yaml(DP_POLICY_CFG)
    policy_yaml.pop("loss_weighting", None)
    policy_yaml["device"] = "cpu"
    old_yaml_path = _write_yaml(tmp_path, "old_diffusion_config.yaml", policy_yaml)

    parsed_cfg = draccus.parse(DiffusionConfig, old_yaml_path, args=[])
    built_cfg = build_policy_config(
        "diffusion",
        load_policy_yaml(old_yaml_path),
        config_path=old_yaml_path,
        mode="train",
    )

    for cfg in (parsed_cfg, built_cfg):
        assert isinstance(cfg.loss_weighting, DiffusionLossWeightingConfig)
        assert cfg.loss_weighting.enabled is False
        assert cfg.loss_weighting.keyframe_weight_column == ANNOTATION_WEIGHT
        assert cfg.loss_weighting.gripper_event_column == ANNOTATION_EVENT


def test_core_train_old_yaml_without_sampler_or_debug_fields_loads_defaults(tmp_path: Path) -> None:
    payload = {
        "dataset": {"repo_id": "local/old_dataset"},
        "policy": {
            "type": "act",
            "device": "cpu",
            "push_to_hub": False,
            "pretrained_backbone_weights": None,
        },
        "output_dir": str(tmp_path / "out"),
        "job_name": "old_act",
        "resume": False,
        "seed": 1000,
        "num_workers": 0,
        "batch_size": 1,
        "steps": 0,
        "eval_freq": 0,
        "log_freq": 1,
        "save_checkpoint": False,
        "save_freq": 1,
        "use_policy_training_preset": True,
        "eval": {"n_episodes": 1, "batch_size": 1},
        "wandb": {"enable": False},
    }
    old_yaml_path = _write_yaml(tmp_path, "old_train.yaml", payload)

    cfg = draccus.parse(CoreTrainPipelineConfig, old_yaml_path, args=[])

    assert cfg.keyframe_sampler.enabled is False
    assert cfg.debug_metrics.enabled is True
    sampler_result = build_keyframe_weighted_sampler(
        _FakeDataset(episodes=[(0, 4)]),
        None,
        cfg.keyframe_sampler,
    )
    assert sampler_result.sampler is None
    assert sampler_result.stats["keyframe_sampler/disabled_reason"] == "disabled_by_config"


def test_dual_arm_train_old_yaml_without_sampler_or_debug_fields_loads_defaults() -> None:
    train_yaml = yaml.safe_load(TRAIN_CFG.read_text(encoding="utf-8"))["train"]
    old_train_cfg = copy.deepcopy(train_yaml)
    old_train_cfg.pop("keyframe_sampler", None)
    old_train_cfg.pop("debug_metrics", None)
    old_train_cfg["training"]["device"] = "cpu"
    old_train_cfg["training"]["gpu"]["enabled"] = False
    old_train_cfg["wandb"]["enable"] = False
    old_train_cfg["wandb"]["mode"] = "disabled"

    cfg = DualArmTrainPipelineConfig(old_train_cfg)
    debug_cfg = normalize_debug_metrics_config(cfg.debug_metrics)

    assert cfg.keyframe_sampler["enabled"] is False
    assert debug_cfg["enabled"] is True
    assert debug_cfg["write_annotation_summary_json"] is True
    sampler_result = build_keyframe_weighted_sampler(
        _FakeDataset(episodes=[(0, 4)]),
        None,
        cfg.keyframe_sampler,
    )
    assert sampler_result.sampler is None
    assert sampler_result.stats["keyframe_sampler/disabled_reason"] == "disabled_by_config"


def test_keyframe_sampler_disabled_and_missing_annotation_fallbacks_are_noops() -> None:
    dataset = _FakeDataset(episodes=[(0, 6)])

    disabled = build_keyframe_weighted_sampler(dataset, None, {"enabled": False})
    assert disabled.sampler is None
    assert disabled.weights is None
    assert disabled.stats["keyframe_sampler/enabled"] is False
    assert disabled.stats["keyframe_sampler/disabled_reason"] == "disabled_by_config"

    missing = build_keyframe_weighted_sampler(dataset, range(4), {"enabled": True})
    assert missing.sampler is None
    assert missing.weights is None
    assert missing.stats["keyframe_sampler/annotation_missing"] is True
    assert missing.stats["keyframe_sampler/fallback_to_default"] is True
    assert missing.stats["keyframe_sampler/disabled_reason"] == "annotation_missing"


def test_keyframe_sampler_preserves_dp_episode_aware_eligible_indices() -> None:
    dataset = _FakeDataset(
        episodes=[(0, 10)],
        gripper_event=[0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    )
    dp_sampler = EpisodeAwareSampler([0], [10], drop_n_last_frames=3, shuffle=True)

    result = build_keyframe_weighted_sampler(
        dataset,
        range(4),
        {"enabled": True, "seed": 123},
        eligible_indices=dp_sampler.indices,
    )

    assert result.sampler is not None
    assert result.stats["keyframe_sampler/eligible_sample_count"] == len(dp_sampler.indices)
    assert result.stats["keyframe_sampler/positive_sample_count"] > 0
    assert set(iter(result.sampler)).issubset(set(dp_sampler.indices))


def test_dual_arm_train_rejects_keyframe_sampler_and_dagger_sampler_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.core.run_train as run_train_module

    class _FakeAccelerator:
        is_main_process = True
        device = torch.device("cpu")
        num_processes = 1

        def wait_for_everyone(self) -> None:
            return None

    class _FakeCfg(SimpleNamespace):
        def validate(self) -> None:
            self.optimizer = SimpleNamespace(grad_clip_norm=1.0)
            self.scheduler = None

        def to_dict(self) -> dict[str, Any]:
            return {}

    fake_cfg = _FakeCfg(
        training=SimpleNamespace(),
        requested_policy_device="cpu",
        policy=SimpleNamespace(
            device="cpu",
            type="act",
            push_to_hub=False,
            pretrained_path=None,
        ),
        dataset=SimpleNamespace(streaming=False),
        env=None,
        eval_freq=0,
        wandb=SimpleNamespace(enable=False, project=None),
        seed=None,
        resume=False,
        output_dir=Path("/tmp/keyframe-regression-conflict"),
        steps=0,
        batch_size=1,
        num_workers=0,
        keyframe_sampler={"enabled": True},
        dagger_sampling={"enabled": True},
        debug_metrics={"enabled": False},
    )
    fake_dataset = SimpleNamespace(
        meta=SimpleNamespace(
            episodes={"dataset_from_index": [0], "dataset_to_index": [4]},
            stats={},
        ),
        num_frames=4,
        num_episodes=1,
    )
    fake_policy = torch.nn.Linear(1, 1)
    fake_policy.config = fake_cfg.policy

    monkeypatch.setattr(
        run_train_module,
        "setup_training_device",
        lambda *args, **kwargs: SimpleNamespace(final_device=torch.device("cpu")),
    )
    monkeypatch.setattr(run_train_module, "init_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_train_module, "log_training_device_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_train_module, "make_dataset", lambda cfg: fake_dataset)
    monkeypatch.setattr(run_train_module, "make_policy", lambda *args, **kwargs: fake_policy)
    monkeypatch.setattr(
        run_train_module,
        "make_pre_post_processors",
        lambda *args, **kwargs: (lambda x: x, lambda x: x),
    )
    monkeypatch.setattr(
        run_train_module,
        "make_optimizer_and_scheduler",
        lambda *args, **kwargs: (torch.optim.SGD(fake_policy.parameters(), lr=0.1), None),
    )

    with pytest.raises(ValueError, match="keyframe_sampler and DAgger source-aware sampler"):
        run_train_module.run_train(fake_cfg, accelerator=_FakeAccelerator())


def test_training_debug_startup_runs_only_on_main_process(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.core.run_train as run_train_module

    class _TinyDataset:
        meta = SimpleNamespace(
            episodes={"dataset_from_index": [0], "dataset_to_index": [1]},
            stats={},
        )
        num_frames = 1
        num_episodes = 1

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {"index": torch.tensor(index)}

    class _FakeAccelerator:
        is_main_process = False
        device = torch.device("cpu")
        num_processes = 1

        def wait_for_everyone(self) -> None:
            return None

        def prepare(self, *args):
            return args

        def end_training(self) -> None:
            return None

    class _FakeCfg(SimpleNamespace):
        def validate(self) -> None:
            self.optimizer = SimpleNamespace(grad_clip_norm=1.0)
            self.scheduler = None

    fake_cfg = _FakeCfg(
        training=SimpleNamespace(),
        requested_policy_device="cpu",
        policy=SimpleNamespace(
            device="cpu",
            type="act",
            push_to_hub=False,
            pretrained_path=None,
        ),
        dataset=SimpleNamespace(streaming=False),
        env=None,
        eval_freq=0,
        wandb=SimpleNamespace(enable=False, project=None),
        seed=None,
        resume=False,
        output_dir=Path("/tmp/keyframe-regression-non-main"),
        steps=0,
        batch_size=1,
        num_workers=0,
        keyframe_sampler={"enabled": False},
        dagger_sampling={"enabled": False},
        debug_metrics={"enabled": True},
        save_checkpoint=False,
        save_freq=1000,
    )
    fake_policy = torch.nn.Linear(1, 1)
    fake_policy.config = fake_cfg.policy

    monkeypatch.setattr(
        run_train_module,
        "setup_training_device",
        lambda *args, **kwargs: SimpleNamespace(final_device=torch.device("cpu")),
    )
    monkeypatch.setattr(run_train_module, "init_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_train_module, "make_dataset", lambda cfg: _TinyDataset())
    monkeypatch.setattr(run_train_module, "make_policy", lambda *args, **kwargs: fake_policy)
    monkeypatch.setattr(
        run_train_module,
        "make_pre_post_processors",
        lambda *args, **kwargs: (lambda x: x, lambda x: x),
    )
    monkeypatch.setattr(
        run_train_module,
        "make_optimizer_and_scheduler",
        lambda *args, **kwargs: (torch.optim.SGD(fake_policy.parameters(), lr=0.1), None),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("debug startup should only run on the main process")

    monkeypatch.setattr(run_train_module, "log_training_debug_startup", fail_if_called)

    run_train_module.run_train(fake_cfg, accelerator=_FakeAccelerator())


def test_real_unannotated_dataset_smoke_batch_loss_sampler_and_debug(tmp_path: Path) -> None:
    dataset = _load_real_dataset(REAL_DATASET_ROOT, include_annotation=False)

    batch = _collated_batch(dataset, [0, 1])
    assert ANNOTATION_WEIGHT not in batch
    assert ANNOTATION_EVENT not in batch

    batch_metrics = compute_batch_annotation_metrics(batch)
    assert batch_metrics == {}

    sampler_result = build_keyframe_weighted_sampler(dataset, range(4), {"enabled": True})
    assert sampler_result.sampler is None
    assert sampler_result.stats["keyframe_sampler/annotation_missing"] is True

    debug_report = log_training_debug_startup(
        dataset=dataset,
        output_dir=tmp_path,
        debug_metrics_config={"enabled": True},
        sampler_stats=sampler_result.stats,
    )
    assert debug_report["annotation_summary"]["annotation/has_annotation"] is False
    assert (tmp_path / "debug" / "annotation_summary.json").is_file()
    assert (tmp_path / "debug" / "keyframe_sampler_summary.json").is_file()

    _assert_act_and_dp_loss_smoke(batch, has_annotation=False)


def test_real_annotated_dataset_smoke_batch_loss_sampler_and_debug(tmp_path: Path) -> None:
    dataset = _load_real_dataset(REAL_ANNOTATED_ROOT, include_annotation=True)

    batch = _collated_batch(dataset, [0, 1])
    assert batch[ANNOTATION_WEIGHT].shape == batch[ACTION].shape[:2]
    assert batch[ANNOTATION_EVENT].shape == batch[ACTION].shape[:2]

    batch_metrics = compute_batch_annotation_metrics(batch)
    assert batch_metrics["batch/has_annotation"] is True

    disabled_sampler = build_keyframe_weighted_sampler(dataset, None, {"enabled": False})
    assert disabled_sampler.sampler is None

    sampler_result = build_keyframe_weighted_sampler(dataset, range(4), {"enabled": True})
    assert sampler_result.weights is not None
    assert sampler_result.stats["keyframe_sampler/annotation_missing"] is False
    assert sampler_result.stats["keyframe_sampler/positive_sample_count"] > 0

    debug_report = log_training_debug_startup(
        dataset=dataset,
        output_dir=tmp_path,
        debug_metrics_config={
            "enabled": True,
            "write_batch_metrics_preview": True,
        },
        sampler_stats=sampler_result.stats,
    )
    assert debug_report["annotation_summary"]["annotation/has_annotation"] is True
    assert (tmp_path / "debug" / "annotation_summary.json").is_file()
    assert (tmp_path / "debug" / "keyframe_sampler_summary.json").is_file()

    _assert_act_and_dp_loss_smoke(batch, has_annotation=True)
