#!/usr/bin/env python

import argparse
import json
import logging
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.datasets.video_utils import VideoEncodingManager, decode_video_frames

try:
    from lerobot.datasets.video_utils import _default_decoder_cache
except ImportError:  # pragma: no cover - compatibility with older LeRobot versions.
    _default_decoder_cache = None

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)["preprocess_dataset"]


def _positive_int(value: Any, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0. Got {value}.")
    return value


def _as_path_or_none(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


def _resolve_for_safety(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _paths_overlap(left: Path, right: Path) -> bool:
    left = _resolve_for_safety(left)
    right = _resolve_for_safety(right)
    return left == right or left in right.parents or right in left.parents


def _assert_output_is_separate_from_source(source: LeRobotDataset, cfg: dict[str, Any]) -> None:
    output_cfg = cfg["output"]
    output_root = _as_path_or_none(output_cfg.get("root"))
    if output_root is None:
        if str(output_cfg.get("repo_id", "")).strip("/") == str(source.repo_id).strip("/"):
            raise ValueError(
                "Refusing to preprocess in place: output.repo_id is the same as source.repo_id "
                "and output.root is not set. Choose a separate output.root/repo_id."
            )
        return

    source_root = _resolve_for_safety(Path(source.root))
    if _paths_overlap(output_root, source_root):
        raise ValueError(
            "Refusing to preprocess in place: output.root overlaps source.root. "
            f"source.root={source_root} output.root={_resolve_for_safety(output_root)}"
        )


def _select_episodes(dataset: LeRobotDataset, cfg: dict[str, Any]) -> list[int]:
    episodes = cfg["source"].get("episodes")
    if episodes is None:
        episodes = list(range(dataset.meta.total_episodes))
    else:
        episodes = [int(ep) for ep in episodes]

    max_episodes = cfg["source"].get("max_episodes")
    if max_episodes is not None:
        episodes = episodes[: int(max_episodes)]
    if not episodes:
        raise ValueError("No source episodes selected.")

    valid = set(range(dataset.meta.total_episodes))
    invalid = sorted(set(episodes) - valid)
    if invalid:
        raise ValueError(
            f"Invalid source episode indices: {invalid}. "
            f"Dataset has episodes 0..{dataset.meta.total_episodes - 1}."
        )

    duplicates = sorted({ep for ep in episodes if episodes.count(ep) > 1})
    if duplicates:
        raise ValueError(f"Duplicate source episodes are not supported: {duplicates}.")
    return episodes


def _indices_matching(action_names: list[str], suffixes: set[str]) -> list[int]:
    return [
        idx
        for idx, name in enumerate(action_names)
        if "delta_ee_pose" in name and name.rsplit(".", 1)[-1] in suffixes
    ]


def _gripper_indices(action_names: list[str]) -> list[int]:
    return [idx for idx, name in enumerate(action_names) if "gripper" in name]


def _expanded_mask(event_mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not event_mask.any():
        return event_mask.copy()

    keep = event_mask.copy()
    event_indices = np.flatnonzero(event_mask)
    for idx in event_indices:
        start = max(0, idx - radius)
        end = min(len(keep), idx + radius + 1)
        keep[start:end] = True
    return keep


def _gripper_event_mask(actions: np.ndarray, action_names: list[str], cfg: dict[str, Any]) -> np.ndarray:
    gripper_cfg = cfg.get("gripper_events", {}) or {}
    if not gripper_cfg.get("enabled", False):
        return np.zeros(actions.shape[0], dtype=bool)

    indices = _gripper_indices(action_names)
    if not indices or actions.shape[0] == 0:
        return np.zeros(actions.shape[0], dtype=bool)

    threshold = float(gripper_cfg.get("change_threshold", 0.5))
    diffs = np.abs(np.diff(actions[:, indices], axis=0))
    event_mask = np.zeros(actions.shape[0], dtype=bool)
    event_mask[1:] = (diffs >= threshold).any(axis=1)
    event_mask[:-1] |= (diffs >= threshold).any(axis=1)
    return _expanded_mask(event_mask, int(gripper_cfg.get("keep_radius_frames", 15)))


def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        raise ValueError("action_smoothing.median_window must be odd.")

    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    out = np.empty_like(values)
    for idx in range(values.shape[0]):
        out[idx] = np.median(padded[idx : idx + window], axis=0)
    return out


def _ema_filter(values: np.ndarray, alpha: float) -> np.ndarray:
    if values.shape[0] == 0:
        return values.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = np.empty_like(values)
    out[0] = values[0]
    for idx in range(1, values.shape[0]):
        out[idx] = alpha * values[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def _smooth_actions(actions: np.ndarray, action_names: list[str], cfg: dict[str, Any]) -> np.ndarray:
    smoothing_cfg = cfg.get("action_smoothing", {}) or {}
    if not smoothing_cfg.get("enabled", False):
        return actions.copy()

    smoothed = actions.copy()
    indices: list[int] = []
    if smoothing_cfg.get("smooth_cartesian", True):
        indices.extend(_indices_matching(action_names, {"x", "y", "z", "rx", "ry", "rz"}))
    if smoothing_cfg.get("smooth_gripper", False):
        indices.extend(_gripper_indices(action_names))
    indices = sorted(set(indices))
    if not indices:
        return smoothed

    values = actions[:, indices]
    method = str(smoothing_cfg.get("method", "median")).lower()
    if method == "median":
        values = _median_filter(values, int(smoothing_cfg.get("median_window", 3)))
    elif method == "ema":
        values = _ema_filter(values, float(smoothing_cfg.get("ema_alpha", 0.35)))
    elif method == "median_ema":
        values = _median_filter(values, int(smoothing_cfg.get("median_window", 3)))
        values = _ema_filter(values, float(smoothing_cfg.get("ema_alpha", 0.35)))
    else:
        raise ValueError("action_smoothing.method must be one of: median, ema, median_ema")
    smoothed[:, indices] = values

    max_translation = smoothing_cfg.get("max_translation_delta")
    if max_translation is not None:
        max_translation = float(max_translation)
        for idx in _indices_matching(action_names, {"x", "y", "z"}):
            smoothed[:, idx] = np.clip(smoothed[:, idx], -max_translation, max_translation)

    max_rotation = smoothing_cfg.get("max_rotation_delta")
    if max_rotation is not None:
        max_rotation = float(max_rotation)
        for idx in _indices_matching(action_names, {"rx", "ry", "rz"}):
            smoothed[:, idx] = np.clip(smoothed[:, idx], -max_rotation, max_rotation)

    return smoothed


def _motion_mask(actions: np.ndarray, action_names: list[str], cfg: dict[str, Any]) -> np.ndarray:
    trim_cfg = cfg.get("static_trim", {}) or {}
    translation_indices = _indices_matching(action_names, {"x", "y", "z"})
    rotation_indices = _indices_matching(action_names, {"rx", "ry", "rz"})

    translation_norm = (
        np.linalg.norm(actions[:, translation_indices], axis=1)
        if translation_indices
        else np.zeros(actions.shape[0], dtype=np.float32)
    )
    rotation_norm = (
        np.linalg.norm(actions[:, rotation_indices], axis=1)
        if rotation_indices
        else np.zeros(actions.shape[0], dtype=np.float32)
    )
    return (translation_norm >= float(trim_cfg.get("translation_norm_threshold", 0.001))) | (
        rotation_norm >= float(trim_cfg.get("rotation_norm_threshold", 0.005))
    )


def _trim_static_runs(moving_or_protected: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    trim_cfg = cfg.get("static_trim", {}) or {}
    if not trim_cfg.get("enabled", False):
        return np.ones_like(moving_or_protected, dtype=bool)

    min_static = int(trim_cfg.get("min_static_frames", 30))
    keep_start = int(trim_cfg.get("keep_start_frames", 5))
    keep_end = int(trim_cfg.get("keep_end_frames", 5))
    keep = moving_or_protected.copy()
    static = ~moving_or_protected

    idx = 0
    while idx < len(static):
        if not static[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(static) and static[idx]:
            idx += 1
        end = idx
        run_len = end - start
        if run_len <= min_static:
            keep[start:end] = True
            continue
        keep[start : min(end, start + keep_start)] = True
        keep[max(start, end - keep_end) : end] = True
    return keep


def _episode_arrays(dataset: LeRobotDataset, start: int, end: int) -> np.ndarray:
    raw_dataset = dataset.hf_dataset.with_format(None)
    return np.asarray(raw_dataset[start:end]["action"], dtype=np.float32)


def _to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return value


def _restore_image_layout(value: Any, feature: dict[str, Any]) -> Any:
    array = _to_numpy(value)
    if not isinstance(array, np.ndarray):
        return array

    expected_shape = tuple(feature.get("shape", ()))
    if (
        array.ndim == 3
        and len(expected_shape) == 3
        and expected_shape[-1] in (1, 3, 4)
        and array.shape[0] == expected_shape[-1]
        and array.shape[1:] == expected_shape[:2]
    ):
        return np.transpose(array, (1, 2, 0))
    return array


def _frame_from_source_item(
    source: LeRobotDataset,
    item: dict[str, Any],
    action: np.ndarray,
) -> dict[str, Any]:
    frame = {}
    for key in source.features:
        if key in DEFAULT_FEATURES:
            continue
        if key == "action":
            frame[key] = action.astype(np.float32)
        elif source.features[key]["dtype"] in ["image", "video"]:
            frame[key] = _restore_image_layout(item[key], source.features[key])
        else:
            frame[key] = _to_numpy(item[key])
    frame["task"] = item["task"]
    return frame


def _apply_output_chunk_settings(output: LeRobotDataset, output_cfg: dict[str, Any]) -> None:
    chunk_settings = {
        key: output_cfg.get(key)
        for key in ("chunks_size", "data_files_size_in_mb", "video_files_size_in_mb")
        if output_cfg.get(key) is not None
    }
    if not chunk_settings:
        return

    output.meta.update_chunk_settings(
        chunks_size=(
            _positive_int(chunk_settings["chunks_size"], "output.chunks_size")
            if "chunks_size" in chunk_settings
            else None
        ),
        data_files_size_in_mb=(
            _positive_int(
                chunk_settings["data_files_size_in_mb"],
                "output.data_files_size_in_mb",
            )
            if "data_files_size_in_mb" in chunk_settings
            else None
        ),
        video_files_size_in_mb=(
            _positive_int(
                chunk_settings["video_files_size_in_mb"],
                "output.video_files_size_in_mb",
            )
            if "video_files_size_in_mb" in chunk_settings
            else None
        ),
    )


def _validate_pending_image_files(output: LeRobotDataset, expected_frames: int) -> None:
    if len(output.meta.camera_keys) == 0:
        return

    output._wait_image_writer()
    errors: list[str] = []
    for key in output.meta.camera_keys:
        paths = [Path(path) for path in output.episode_buffer.get(key, [])]
        if len(paths) != expected_frames:
            errors.append(f"{key}: expected {expected_frames} image paths, found {len(paths)}")
            continue

        missing = [path for path in paths if not path.is_file()]
        empty = [path for path in paths if path.is_file() and path.stat().st_size == 0]
        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            errors.append(f"{key}: {len(missing)} missing image file(s), e.g. {preview}")
        if empty:
            preview = ", ".join(str(path) for path in empty[:3])
            errors.append(f"{key}: {len(empty)} empty image file(s), e.g. {preview}")

    if errors:
        raise RuntimeError(
            "Image writing failed before episode encoding:\n" + "\n".join(f"- {err}" for err in errors)
        )


def _sample_verification_indices(num_frames: int, verify_cfg: dict[str, Any]) -> list[int]:
    if num_frames <= 0:
        return []
    if bool(verify_cfg.get("full_scan", True)):
        return list(range(num_frames))

    sample_count = int(verify_cfg.get("sample_count", 512))
    if sample_count <= 0:
        return []
    if sample_count >= num_frames:
        return list(range(num_frames))
    return sorted({int(idx) for idx in np.linspace(0, num_frames - 1, sample_count)})


def _scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.item() if value.shape == () or value.size == 1 else value[0]
    return value


def _probe_video_keys(
    dataset: LeRobotDataset,
    idx: int,
    backend: str,
) -> list[dict[str, Any]]:
    item = dataset.hf_dataset[idx]
    ep_idx = int(_scalar(item["episode_index"]))
    timestamp = float(_scalar(item["timestamp"]))
    ep = dataset.meta.episodes[ep_idx]
    results: list[dict[str, Any]] = []

    for key in dataset.meta.video_keys:
        from_ts = float(ep[f"videos/{key}/from_timestamp"])
        shifted_ts = from_ts + timestamp
        video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, key)
        try:
            if backend == "torchcodec" and _default_decoder_cache is not None:
                _default_decoder_cache.clear()
            decode_video_frames(video_path, [shifted_ts], dataset.tolerance_s, backend)
            status = "ok"
            error = None
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"

        results.append(
            {
                "video_key": key,
                "status": status,
                "error": error,
                "video_path": str(video_path),
                "query_timestamp": shifted_ts,
            }
        )

    if _default_decoder_cache is not None:
        _default_decoder_cache.clear()
    return results


def _verify_output_dataset(
    cfg: dict[str, Any],
    expected_frames: int,
    expected_episodes: int,
) -> None:
    verify_cfg = cfg.get("verification", {}) or {}
    if not verify_cfg.get("enabled", True):
        logger.info("[VERIFY] skipped by config")
        return

    output_cfg = cfg["output"]
    output_root = _as_path_or_none(output_cfg.get("root"))
    backend = str(
        verify_cfg.get("video_backend") or output_cfg.get("video_backend") or "torchcodec"
    ).strip().lower()
    output = LeRobotDataset(
        output_cfg["repo_id"],
        root=output_root,
        video_backend=backend,
    )

    if output.meta.total_frames != expected_frames:
        raise RuntimeError(
            "Output frame count mismatch after preprocessing: "
            f"metadata has {output.meta.total_frames}, expected {expected_frames}."
        )
    if output.meta.total_episodes != expected_episodes:
        raise RuntimeError(
            "Output episode count mismatch after preprocessing: "
            f"metadata has {output.meta.total_episodes}, expected {expected_episodes}."
        )

    indices = _sample_verification_indices(len(output), verify_cfg)
    logger.info(
        "[VERIFY] decoding %d/%d frame(s) with backend=%s",
        len(indices),
        len(output),
        backend,
    )

    failures: list[dict[str, Any]] = []
    max_failures = int(verify_cfg.get("max_failures", 20))
    for position, idx in enumerate(indices, start=1):
        try:
            output[idx]
        except Exception as exc:
            item = output.hf_dataset[idx]
            record = {
                "idx": int(idx),
                "episode_index": int(_scalar(item["episode_index"])),
                "frame_index": int(_scalar(item["frame_index"])),
                "timestamp": float(_scalar(item["timestamp"])),
                "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
                "video_keys": _probe_video_keys(output, int(idx), backend),
            }
            failures.append(record)
            logger.error("[VERIFY FAIL] %s", record)
            if len(failures) >= max_failures:
                break

        if position % 5000 == 0:
            logger.info("[VERIFY] decoded %d/%d frame(s)", position, len(indices))

    if failures:
        report_path = output.root / "meta" / "preprocess_validation_failures.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(failures, f, indent=2)
        message = (
            f"Output dataset verification found {len(failures)} decode failure(s). "
            f"Report written to {report_path}."
        )
        if verify_cfg.get("fail_on_error", True):
            raise RuntimeError(message)
        logger.warning(message)
        return

    logger.info("[VERIFY] OK")


def _create_output_dataset(source: LeRobotDataset, cfg: dict[str, Any]) -> LeRobotDataset:
    _assert_output_is_separate_from_source(source, cfg)
    output_cfg = cfg["output"]
    output_root = _as_path_or_none(output_cfg.get("root"))
    if output_root is not None and output_root.exists():
        if output_cfg.get("overwrite", False):
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"Output dataset already exists: {output_root}. "
                "Set output.overwrite=true to replace it."
            )

    output = LeRobotDataset.create(
        repo_id=output_cfg["repo_id"],
        root=output_root,
        fps=source.fps,
        features={
            key: value
            for key, value in source.features.items()
            if key not in DEFAULT_FEATURES
        },
        robot_type=source.meta.info.get("robot_type"),
        use_videos=len(source.meta.video_keys) > 0,
        image_writer_threads=int(output_cfg.get("image_writer_threads", 4)),
        batch_encoding_size=int(output_cfg.get("batch_encoding_size", 1)),
        video_backend=output_cfg.get("video_backend"),
    )
    _apply_output_chunk_settings(output, output_cfg)
    return output


def preprocess_dataset(cfg: dict[str, Any]) -> None:
    source_cfg = cfg["source"]
    source = LeRobotDataset(
        source_cfg["repo_id"],
        root=_as_path_or_none(source_cfg.get("root")),
        video_backend=source_cfg.get("video_backend"),
    )
    episodes = _select_episodes(source, cfg)
    action_names = source.features["action"]["names"]
    dry_run = bool(cfg.get("dry_run", False))
    quality_cfg = cfg.get("quality", {}) or {}
    min_output_frames = int(quality_cfg.get("min_output_frames", 1))
    drop_short_episodes = bool(quality_cfg.get("drop_short_episodes", False))

    output = None if dry_run else _create_output_dataset(source, cfg)
    total_in = 0
    total_out = 0
    written_episodes = 0

    context = nullcontext() if dry_run else VideoEncodingManager(output)
    with context:
        for ep_idx in episodes:
            ep = source.meta.episodes[int(ep_idx)]
            start = int(ep["dataset_from_index"])
            end = int(ep["dataset_to_index"])
            actions = _episode_arrays(source, start, end)

            smoothed_actions = _smooth_actions(actions, action_names, cfg)
            gripper_keep = _gripper_event_mask(actions, action_names, cfg)
            motion = _motion_mask(smoothed_actions, action_names, cfg)
            keep_mask = _trim_static_runs(motion | gripper_keep, cfg)
            keep_indices = np.flatnonzero(keep_mask) + start

            total_in += end - start
            logger.info(
                "[EP %s -> %s] frames %d -> %d (%.1f%% kept), motion=%.1f%% gripper_keep=%.1f%%",
                ep_idx,
                written_episodes,
                end - start,
                len(keep_indices),
                100.0 * len(keep_indices) / max(end - start, 1),
                100.0 * float(motion.mean()) if len(motion) else 0.0,
                100.0 * float(gripper_keep.mean()) if len(gripper_keep) else 0.0,
            )
            if len(keep_indices) < min_output_frames:
                message = (
                    f"Episode {ep_idx} would produce only {len(keep_indices)} frame(s); "
                    f"min_output_frames={min_output_frames}."
                )
                if drop_short_episodes:
                    logger.warning("[SKIP] %s", message)
                    continue
                raise RuntimeError(message)

            if dry_run:
                total_out += len(keep_indices)
                written_episodes += 1
                continue

            for source_idx in keep_indices:
                local_idx = int(source_idx - start)
                item = source[int(source_idx)]
                frame = _frame_from_source_item(source, item, smoothed_actions[local_idx])
                output.add_frame(frame)
            _validate_pending_image_files(output, len(keep_indices))
            output.save_episode()
            total_out += len(keep_indices)
            written_episodes += 1

    if not dry_run:
        _verify_output_dataset(cfg, total_out, written_episodes)

    logger.info(
        "[DONE] episodes %d -> %d, frames %d -> %d (%.1f%% kept)%s",
        len(episodes),
        written_episodes,
        total_in,
        total_out,
        100.0 * total_out / max(total_in, 1),
        " [dry-run]" if dry_run else f" output={cfg['output']['repo_id']}",
    )


def main() -> None:
    default_cfg = Path(__file__).resolve().parents[1] / "config" / "dataset_config" / "preprocess_dataset_cfg.yaml"
    parser = argparse.ArgumentParser(description="Preprocess a LeRobot dataset for ACT training.")
    parser.add_argument("--config", type=Path, default=default_cfg)
    parser.add_argument("--dry-run", action="store_true", help="Only report frame counts; do not write output.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Override source.max_episodes.")
    parser.add_argument("--overwrite", action="store_true", help="Override output.overwrite=true.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.dry_run:
        cfg["dry_run"] = True
    if args.max_episodes is not None:
        cfg["source"]["max_episodes"] = args.max_episodes
    if args.overwrite:
        cfg["output"]["overwrite"] = True

    preprocess_dataset(cfg)


if __name__ == "__main__":
    main()
