#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME


DEFAULT_KEEP_FRAME_ROLES = ("takeover_start", "recovery")


@dataclass
class DAggerExportConfig:
    raw_root: str | Path
    output_root: str | Path
    seed_root: str | Path | None = None
    raw_repo_id: str | None = None
    output_repo_id: str | None = None
    seed_repo_id: str | None = None
    keep_frame_roles: tuple[str, ...] = DEFAULT_KEEP_FRAME_ROLES
    min_segment_frames: int = 1
    overwrite: bool = False
    image_writer_processes: int = 0
    image_writer_threads: int = 4


def _resolve_root(repo_id: str | None, root: str | Path | None) -> Path:
    if root is not None:
        return Path(root)
    if repo_id is None:
        raise ValueError("Either repo_id or root must be provided.")
    return HF_LEROBOT_HOME / repo_id


def _repo_id_from_root(root: Path, fallback: str) -> str:
    if root.name:
        return root.name
    return fallback


def _to_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return _to_scalar(value[0])
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_bool(value: Any) -> bool:
    return bool(_to_scalar(value))


def _to_int(value: Any, default: int = -1) -> int:
    value = _to_scalar(value)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_str(value: Any) -> str:
    value = _to_scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _coerce_numpy_feature(value: Any, feature: dict[str, Any]) -> Any:
    dtype = feature["dtype"]
    if dtype in {"image", "video", "string"}:
        return value

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.dtype(dtype))
    expected_shape = tuple(feature["shape"])
    if array.shape == ():
        array = array.reshape(expected_shape)
    return array


def _frame_role_at(raw_hf_dataset, idx: int) -> str:
    column_names = raw_hf_dataset.column_names
    if "frame_role" in column_names:
        return _to_str(raw_hf_dataset["frame_role"][idx])
    if "is_expert" in column_names and _to_bool(raw_hf_dataset["is_expert"][idx]):
        return "recovery"
    return "policy"


def _segment_id_at(raw_hf_dataset, idx: int, fallback: int) -> int:
    if "intervention_segment_id" not in raw_hf_dataset.column_names:
        return fallback
    return _to_int(raw_hf_dataset["intervention_segment_id"][idx], default=fallback)


def _iter_export_segments(
    raw_dataset: LeRobotDataset,
    keep_frame_roles: set[str],
) -> list[dict[str, Any]]:
    """Return truly continuous intervention slices from the raw run_mix timeline.

    Export keeps only takeover/recovery roles by default, drops reset/ignore/policy
    frames, and flushes whenever the raw index or intervention_segment_id is no
    longer continuous. This prevents sparse takeover frames from being interpreted
    as one continuous ACT chunk.
    """

    raw_dataset._ensure_hf_dataset_loaded()
    raw_hf_dataset = raw_dataset.hf_dataset
    segments: list[dict[str, Any]] = []

    for raw_episode_index in range(raw_dataset.num_episodes):
        episode = raw_dataset.meta.episodes[raw_episode_index]
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])

        current_indices: list[int] = []
        current_segment_id: int | None = None
        last_idx: int | None = None

        def flush() -> None:
            nonlocal current_indices, current_segment_id, last_idx
            if current_indices:
                segments.append(
                    {
                        "raw_episode_index": raw_episode_index,
                        "intervention_segment_id": current_segment_id,
                        "indices": current_indices,
                    }
                )
            current_indices = []
            current_segment_id = None
            last_idx = None

        for idx in range(start, end):
            role = _frame_role_at(raw_hf_dataset, idx)
            if role not in keep_frame_roles:
                flush()
                continue

            segment_id = _segment_id_at(raw_hf_dataset, idx, fallback=idx)
            if (
                current_indices
                and (segment_id != current_segment_id or last_idx is None or idx != last_idx + 1)
            ):
                flush()

            current_indices.append(idx)
            current_segment_id = segment_id
            last_idx = idx

        flush()

    return segments


def _make_standard_frame(
    raw_item: dict[str, Any],
    output_features: dict[str, dict],
) -> dict[str, Any]:
    if "expert_action" not in raw_item:
        raise KeyError(
            "Raw run_mix dataset is missing `expert_action`; cannot export DAgger labels safely."
        )

    frame: dict[str, Any] = {"task": raw_item["task"]}
    for key, feature in output_features.items():
        if key in DEFAULT_FEATURES:
            continue
        if key == ACTION:
            # DAgger supervision must use the expert label. The mixed/sent action
            # is intentionally not used as ACT's standard training target.
            frame[key] = _coerce_numpy_feature(raw_item["expert_action"], feature)
            continue
        if key not in raw_item:
            raise KeyError(f"Raw run_mix frame is missing required training feature `{key}`.")
        frame[key] = _coerce_numpy_feature(raw_item[key], feature)
    return frame


def export_dagger_dataset(
    raw_root: str | Path,
    output_root: str | Path,
    seed_root: str | Path | None = None,
    raw_repo_id: str | None = None,
    output_repo_id: str | None = None,
    seed_repo_id: str | None = None,
    keep_frame_roles: tuple[str, ...] = DEFAULT_KEEP_FRAME_ROLES,
    min_segment_frames: int = 1,
    overwrite: bool = False,
    image_writer_processes: int = 0,
    image_writer_threads: int = 4,
) -> dict[str, Any]:
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    seed_root = _resolve_root(seed_repo_id, seed_root)
    raw_repo_id = raw_repo_id or _repo_id_from_root(raw_root, "raw_run_mix")
    output_repo_id = output_repo_id or _repo_id_from_root(output_root, "dagger_export")
    seed_repo_id = seed_repo_id or _repo_id_from_root(seed_root, "seed")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_root}")
        shutil.rmtree(output_root)

    seed_meta = LeRobotDatasetMetadata(seed_repo_id, root=seed_root)
    raw_dataset = LeRobotDataset(raw_repo_id, root=raw_root)
    output_features = seed_meta.features
    use_videos = any(feature["dtype"] == "video" for feature in output_features.values())

    output_dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=seed_meta.fps,
        features=output_features,
        robot_type=seed_meta.robot_type,
        root=output_root,
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )
    output_dataset.meta.metadata_buffer_size = 1

    segments = _iter_export_segments(raw_dataset, set(keep_frame_roles))
    exported_segments = 0
    exported_frames = 0
    skipped_short_segments = 0
    exported_raw_episode_indices: set[int] = set()
    exported_intervention_ids: set[tuple[int, int | None]] = set()

    for segment in segments:
        indices = segment["indices"]
        if len(indices) < min_segment_frames:
            skipped_short_segments += 1
            continue

        for raw_idx in indices:
            raw_item = raw_dataset[raw_idx]
            output_dataset.add_frame(_make_standard_frame(raw_item, output_features))

        output_dataset.save_episode()
        exported_segments += 1
        exported_frames += len(indices)
        exported_raw_episode_indices.add(int(segment["raw_episode_index"]))
        exported_intervention_ids.add(
            (int(segment["raw_episode_index"]), segment["intervention_segment_id"])
        )

    output_dataset.finalize()

    raw_dataset._ensure_hf_dataset_loaded()
    raw_hf_dataset = raw_dataset.hf_dataset
    total_raw_frames = len(raw_dataset)
    expert_frames = (
        sum(1 for value in raw_hf_dataset["is_expert"] if _to_bool(value))
        if "is_expert" in raw_hf_dataset.column_names
        else exported_frames
    )
    intervention_ids = set()
    if "intervention_segment_id" in raw_hf_dataset.column_names:
        for value in raw_hf_dataset["intervention_segment_id"]:
            segment_id = _to_int(value)
            if segment_id >= 0:
                intervention_ids.add(segment_id)

    summary = {
        "raw_dataset": {
            "repo_id": raw_repo_id,
            "root": str(raw_root),
            "total_frames": total_raw_frames,
            "total_episodes": raw_dataset.num_episodes,
        },
        "seed_dataset": {
            "repo_id": seed_repo_id,
            "root": str(seed_root),
        },
        "exported_dataset": {
            "repo_id": output_repo_id,
            "root": str(output_root),
            "total_frames": exported_frames,
            "total_episodes": exported_segments,
        },
        "rules": {
            "label_source": "expert_action",
            "standard_action_field": ACTION,
            "keep_frame_roles": list(keep_frame_roles),
            "dropped_frame_roles": ["policy", "reset", "ignore"],
            "min_segment_frames": min_segment_frames,
        },
        "stats": {
            "expert_frame_ratio": expert_frames / max(1, total_raw_frames),
            "expert_episode_count": len(exported_raw_episode_indices),
            "intervention_count": len(exported_intervention_ids)
            if exported_intervention_ids
            else (len(intervention_ids) if intervention_ids else len(segments)),
            "skipped_short_segments": skipped_short_segments,
        },
    }

    summary_path = output_root / "dagger_export_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(
        "[dagger_export] raw=%s exported=%s frames=%d episodes=%d",
        raw_root,
        output_root,
        exported_frames,
        exported_segments,
    )
    return summary


def export_from_config(config: DAggerExportConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        config = DAggerExportConfig(**config)
    return export_dagger_dataset(**config.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export standardized ACT DAgger data from raw run_mix logs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "dagger_rounds_cfg.yaml",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    export_cfg = cfg.get("dagger_export", cfg)
    export_from_config(export_cfg)


if __name__ == "__main__":
    main()
