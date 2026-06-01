#!/usr/bin/env python3
"""Gripper transition detector and annotation exporter for LeRobot datasets.

By default this script reads an existing LeRobot dataset, detects gripper
opening/closing transitions from action columns, and writes diagnostic reports
only. With ``--export-annotated-copy`` it first copies the source dataset to a
new root, then adds gripper transition annotation columns only in that copy.
It never modifies the source dataset.

python scripts/debug/annotate_gripper_transition.py \
    --dataset-root /home/geist/.cache/huggingface/lerobot/nero_task3_step1/empty_merged_E113 \
    --output-dir /home/geist/.cache/huggingface/lerobot/nero_task3_step1/empty_merged_E113/report   \
    --max-episodes 10   \
    --plot   \
    --dry-run   \
    --detector hysteresis   \
    --open-high true   \
    --open-threshold 0.9   \
    --close-threshold 0.2   \
    --event-frame reached_state   \
    --pre-window 10   \
    --post-window 6   \
    --expected-left-opening 2   \
    --expected-left-closing 2   \
    --expected-right-opening 2   \
    --expected-right-closing 2 \
    --overwrite
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LeRobotDataset = None
_LEROBOT_IMPORT_ERROR: BaseException | None = None

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    _LEROBOT_IMPORT_ERROR = exc
    repo_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo_root / "src"))
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError as fallback_exc:
        _LEROBOT_IMPORT_ERROR = fallback_exc


EVENT_NORMAL = 0
EVENT_PRE_CLOSING = 1
EVENT_CLOSING = 2
EVENT_POST_CLOSING = 3
EVENT_PRE_OPENING = 4
EVENT_OPENING = 5
EVENT_POST_OPENING = 6
EVENT_TRANSITION_UNKNOWN = 7

EVENT_NAMES = {
    EVENT_NORMAL: "normal",
    EVENT_PRE_CLOSING: "pre_closing",
    EVENT_CLOSING: "closing",
    EVENT_POST_CLOSING: "post_closing",
    EVENT_PRE_OPENING: "pre_opening",
    EVENT_OPENING: "opening",
    EVENT_POST_OPENING: "post_opening",
    EVENT_TRANSITION_UNKNOWN: "transition_unknown",
}

EVENT_WEIGHTS = {
    EVENT_NORMAL: 1.0,
    EVENT_PRE_CLOSING: 2.0,
    EVENT_CLOSING: 6.0,
    EVENT_POST_CLOSING: 3.0,
    EVENT_PRE_OPENING: 2.0,
    EVENT_OPENING: 6.0,
    EVENT_POST_OPENING: 3.0,
    EVENT_TRANSITION_UNKNOWN: 4.0,
}

EVENT_PRIORITIES = {
    EVENT_NORMAL: 0,
    EVENT_PRE_CLOSING: 1,
    EVENT_POST_CLOSING: 1,
    EVENT_PRE_OPENING: 1,
    EVENT_POST_OPENING: 1,
    EVENT_TRANSITION_UNKNOWN: 2,
    EVENT_CLOSING: 3,
    EVENT_OPENING: 3,
}

SIDE_LEFT = "left"
SIDE_RIGHT = "right"

STABLE_TRANSITION_ZONE = 0
STABLE_OPEN = 1
STABLE_CLOSED = -1

EXPECTED_COUNT_FIELDS = (
    (SIDE_LEFT, "opening", "expected_left_opening"),
    (SIDE_LEFT, "closing", "expected_left_closing"),
    (SIDE_RIGHT, "opening", "expected_right_opening"),
    (SIDE_RIGHT, "closing", "expected_right_closing"),
)

ANNOTATION_COLUMN_DTYPES = {
    "left_gripper_event": "int64",
    "right_gripper_event": "int64",
    "gripper_event": "int64",
    "left_keyframe_weight": "float32",
    "right_keyframe_weight": "float32",
    "keyframe_weight": "float32",
}


@dataclass(frozen=True)
class GripperDims:
    left: int | None
    right: int | None
    warnings: list[str]


@dataclass(frozen=True)
class Transition:
    frame: int
    event: int
    strength: float


@dataclass
class SideDetection:
    side: str
    dim: int | None
    name: str | None
    values: np.ndarray
    smoothed_values: np.ndarray
    events: np.ndarray
    weights: np.ndarray
    mode: str | None
    open_high: bool | None
    transitions: list[Transition]
    warnings: list[str]

    @property
    def opening_count(self) -> int:
        return sum(1 for transition in self.transitions if transition.event == EVENT_OPENING)

    @property
    def closing_count(self) -> int:
        return sum(1 for transition in self.transitions if transition.event == EVENT_CLOSING)

    @property
    def unknown_count(self) -> int:
        return sum(1 for transition in self.transitions if transition.event == EVENT_TRANSITION_UNKNOWN)


@dataclass
class EpisodeDetection:
    episode_index: int
    frame_indices: np.ndarray
    global_indices: np.ndarray
    left: SideDetection
    right: SideDetection
    combined_events: np.ndarray
    combined_weights: np.ndarray
    warnings: list[str]
    expected_count_ok: bool = True
    unexpected_count_warning: str = ""

    @property
    def num_frames(self) -> int:
        return int(len(self.combined_events))

    @property
    def keyframe_frame_count(self) -> int:
        return int(np.count_nonzero(self.combined_events != EVENT_NORMAL))

    @property
    def keyframe_ratio(self) -> float:
        if self.num_frames == 0:
            return 0.0
        return float(self.keyframe_frame_count / self.num_frames)

    @property
    def max_weight(self) -> float:
        if self.combined_weights.size == 0:
            return 1.0
        return float(np.max(self.combined_weights))


@dataclass(frozen=True)
class AnnotationExportPaths:
    report: Path
    frames: Path
    episodes: Path


def _require_lerobot() -> None:
    if LeRobotDataset is None:
        raise ModuleNotFoundError(
            "LeRobot and its dataset dependencies are not importable in this Python environment. "
            "Activate the environment used for LeRobot recording/training before running this script."
        ) from _LEROBOT_IMPORT_ERROR


def _infer_repo_id(root: Path) -> str:
    return root.expanduser().resolve(strict=False).name or "local_gripper_transition_dataset"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve_for_safety(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _path_inside(path: Path, root: Path) -> bool:
    path = _resolve_for_safety(path)
    root = _resolve_for_safety(root)
    return path == root or root in path.parents


def _validate_output_dataset_path(dataset_root: Path, output_dataset_root: Path) -> None:
    dataset_root = _resolve_for_safety(dataset_root)
    output_dataset_root = _resolve_for_safety(output_dataset_root)
    if dataset_root == output_dataset_root:
        raise ValueError("--output-dataset-root must be different from --dataset-root.")
    if _path_inside(output_dataset_root, dataset_root):
        raise ValueError("--output-dataset-root must not be inside --dataset-root.")
    if _path_inside(dataset_root, output_dataset_root):
        raise ValueError("--dataset-root must not be inside --output-dataset-root.")


def _validate_export_report_dir(dataset_root: Path, output_dir: Path) -> None:
    dataset_root = _resolve_for_safety(dataset_root)
    output_dir = _resolve_for_safety(output_dir)
    if _path_inside(output_dir, dataset_root):
        raise ValueError("--output-dir must not be inside --dataset-root when exporting a dataset copy.")


def _annotation_columns(prefix: str) -> dict[str, str]:
    prefix = str(prefix).strip()
    if not prefix:
        raise ValueError("--annotation-prefix must not be empty.")
    if prefix.startswith(".") or prefix.endswith("."):
        raise ValueError("--annotation-prefix must not start or end with '.'.")
    if "/" in prefix or "\\" in prefix:
        raise ValueError("--annotation-prefix must not contain path separators.")
    if any(ch.isspace() for ch in prefix):
        raise ValueError("--annotation-prefix must not contain whitespace.")
    return {key: f"{prefix}.{key}" for key in ANNOTATION_COLUMN_DTYPES}


def _annotation_feature_schema(dtype: str) -> dict[str, Any]:
    return {"dtype": dtype, "shape": [1], "names": None}


def _add_annotation_features_to_info_dict(info: dict[str, Any], columns: dict[str, str]) -> dict[str, Any]:
    info = copy.deepcopy(info)
    features = info.setdefault("features", {})
    for suffix, column in columns.items():
        features[column] = _annotation_feature_schema(ANNOTATION_COLUMN_DTYPES[suffix])
    return info


def _data_parquet_paths(root: Path) -> list[Path]:
    return sorted((root / "data").glob("*/*.parquet"))


def _parquet_row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.read_metadata(path).num_rows)


def _parquet_schema_names(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.read_schema(path).names)


def _file_stat_summary(path: Path, *, include_sha256: bool = False) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    summary: dict[str, Any] = {
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256 and path.is_file():
        import hashlib

        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        summary["sha256"] = digest.hexdigest()
    return summary


def _tree_file_stat_summary(root: Path, *, relative_dir: str | None = None) -> dict[str, Any]:
    base = root / relative_dir if relative_dir else root
    if not base.exists():
        return {"exists": False, "file_count": 0, "total_size_bytes": 0, "max_mtime_ns": None}

    file_count = 0
    total_size = 0
    max_mtime_ns: int | None = None
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        file_count += 1
        total_size += int(stat.st_size)
        if max_mtime_ns is None:
            max_mtime_ns = int(stat.st_mtime_ns)
        else:
            max_mtime_ns = max(max_mtime_ns, int(stat.st_mtime_ns))
    return {
        "exists": True,
        "file_count": file_count,
        "total_size_bytes": int(total_size),
        "max_mtime_ns": max_mtime_ns,
    }


def _dataset_mtime_summary(root: Path, annotation_columns: list[str]) -> dict[str, Any]:
    data_files = _data_parquet_paths(root)
    data_file_summaries = []
    data_rows = 0
    annotation_columns_present: dict[str, list[str]] = {}
    for path in data_files:
        rel = str(path.relative_to(root))
        rows = _parquet_row_count(path)
        data_rows += rows
        schema_names = set(_parquet_schema_names(path))
        present = [column for column in annotation_columns if column in schema_names]
        if present:
            annotation_columns_present[rel] = present
        data_file_summaries.append(
            {
                "path": rel,
                "rows": rows,
                **_file_stat_summary(path),
            }
        )

    info_path = root / "meta" / "info.json"
    info_features: dict[str, Any] = {}
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info_features = info.get("features", {})

    return {
        "root": str(root),
        "info_json": _file_stat_summary(info_path, include_sha256=True),
        "data_parquet_count": len(data_files),
        "data_row_count": int(data_rows),
        "data_files": data_file_summaries,
        "videos": _tree_file_stat_summary(root, relative_dir="videos"),
        "annotation_features_present": [column for column in annotation_columns if column in info_features],
        "annotation_columns_present": annotation_columns_present,
    }


def _flatten_feature_names(names: Any) -> list[str] | None:
    if names is None:
        return None
    if isinstance(names, dict):
        out: list[str] = []
        for value in names.values():
            flattened = _flatten_feature_names(value)
            if flattened is None:
                out.append(str(value))
            else:
                out.extend(flattened)
        return out
    if isinstance(names, (list, tuple)):
        return [str(value) for value in names]
    return [str(names)]


def _action_schema(dataset: Any) -> tuple[tuple[int, ...], int, list[str] | None]:
    action_feature = dataset.features.get("action")
    if action_feature is None:
        raise ValueError("Input dataset has no 'action' feature in metadata.")

    raw_shape = action_feature.get("shape") or ()
    action_shape = tuple(int(dim) for dim in raw_shape)
    if not action_shape:
        raise ValueError(f"Cannot determine action shape from metadata: {action_feature!r}")
    action_dim = int(np.prod(action_shape))

    names = _flatten_feature_names(action_feature.get("names"))
    if names is not None and len(names) != action_dim:
        raise ValueError(
            "Action feature names length does not match action shape. "
            f"len(names)={len(names)} action_shape={action_shape} names={names}"
        )
    return action_shape, action_dim, names


def _validate_dim(dim: int | None, action_dim: int, label: str) -> int | None:
    if dim is None:
        return None
    if dim < 0 or dim >= action_dim:
        raise ValueError(f"{label}={dim} is outside action dim {action_dim}.")
    return int(dim)


def _side_from_name(name: str) -> str | None:
    lowered = name.lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", lowered) if token]
    token_set = set(tokens)

    if {"left", "l"}.intersection(token_set) or lowered.startswith("left"):
        return SIDE_LEFT
    if {"right", "r"}.intersection(token_set) or lowered.startswith("right"):
        return SIDE_RIGHT
    if "left_gripper" in lowered or "gripper_left" in lowered:
        return SIDE_LEFT
    if "right_gripper" in lowered or "gripper_right" in lowered:
        return SIDE_RIGHT
    return None


def _candidate_score(name: str, side: str | None) -> int:
    lowered = name.lower()
    score = 0
    if side is not None and _side_from_name(name) == side:
        score += 100
    if "gripper_cmd_bin" in lowered:
        score += 40
    elif "cmd_bin" in lowered or lowered.endswith("_bin") or ".bin" in lowered:
        score += 35
    if "gripper_cmd" in lowered:
        score += 30
    if "gripper" in lowered:
        score += 20
    if "width" in lowered or "aperture" in lowered:
        score += 8
    if "state" in lowered:
        score += 2
    return score


def _best_candidate(candidates: list[tuple[int, str]], side: str) -> tuple[int, str] | None:
    side_candidates = [(idx, name) for idx, name in candidates if _side_from_name(name) == side]
    if not side_candidates:
        return None

    scored = sorted(
        ((idx, name, _candidate_score(name, side)) for idx, name in side_candidates),
        key=lambda item: (-item[2], item[0]),
    )
    return scored[0][0], scored[0][1]


def _resolve_gripper_dims(
    *,
    action_names: list[str] | None,
    action_shape: tuple[int, ...],
    action_dim: int,
    left_dim: int | None,
    right_dim: int | None,
    gripper_name_regex: str,
) -> GripperDims:
    warnings: list[str] = []
    left = _validate_dim(left_dim, action_dim, "--left-gripper-dim")
    right = _validate_dim(right_dim, action_dim, "--right-gripper-dim")

    if left is not None and right is not None:
        if left == right:
            raise ValueError("--left-gripper-dim and --right-gripper-dim must not point to the same dim.")
        return GripperDims(left=left, right=right, warnings=warnings)

    if action_names is None:
        if left is None and right is None:
            raise ValueError(
                "Could not infer gripper dims because action metadata has no names. "
                f"Pass --left-gripper-dim/--right-gripper-dim manually. action_shape={action_shape}"
            )
        warnings.append("Action names are absent; using only explicit gripper dim arguments.")
        return GripperDims(left=left, right=right, warnings=warnings)

    pattern = re.compile(gripper_name_regex, flags=re.IGNORECASE)
    candidates = [(idx, name) for idx, name in enumerate(action_names) if pattern.search(name)]
    if not candidates:
        raise ValueError(
            "Could not find gripper action dims from metadata names. "
            f"regex={gripper_name_regex!r} action_shape={action_shape} action_names={action_names}"
        )

    if left is None:
        best_left = _best_candidate(candidates, SIDE_LEFT)
        if best_left is not None:
            left = best_left[0]

    if right is None:
        best_right = _best_candidate(candidates, SIDE_RIGHT)
        if best_right is not None:
            right = best_right[0]

    if left is None and right is None:
        generic = [(idx, name) for idx, name in candidates if _side_from_name(name) is None]
        if len(generic) == 1:
            left = generic[0][0]
            warnings.append(
                "Found one generic gripper action name without left/right side; "
                f"using it as left_gripper_dim={left}. Pass explicit dims to override."
            )
        else:
            raise ValueError(
                "Could not determine left/right gripper dims from action names. "
                f"Pass explicit dims. action_shape={action_shape} action_names={action_names}"
            )

    if left is not None and right is not None and left == right:
        raise ValueError(
            "Resolved left and right gripper dims to the same index. "
            f"left={left} right={right} action_names={action_names}"
        )

    if left is None:
        warnings.append("Left gripper dim was not resolved; left event columns will stay normal/weight=1.")
    if right is None:
        warnings.append("Right gripper dim was not resolved; right event columns will stay normal/weight=1.")

    return GripperDims(left=left, right=right, warnings=warnings)


def _parse_episode_indexes(values: list[str] | None) -> list[int] | None:
    if not values:
        return None

    episode_indexes: list[int] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                parts = token.split(":")
                if len(parts) not in (2, 3):
                    raise argparse.ArgumentTypeError(f"Invalid episode range: {token!r}")
                start = int(parts[0]) if parts[0] else 0
                stop = int(parts[1])
                step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
                episode_indexes.extend(range(start, stop, step))
            else:
                episode_indexes.append(int(token))
    return episode_indexes


def _select_episode_indexes(dataset: Any, requested: list[int] | None, max_episodes: int | None) -> list[int]:
    episode_by_index = {
        int(dataset.meta.episodes[idx]["episode_index"]): dataset.meta.episodes[idx]
        for idx in range(len(dataset.meta.episodes))
    }
    if requested is None:
        selected = sorted(episode_by_index)
    else:
        missing = [idx for idx in requested if idx not in episode_by_index]
        if missing:
            raise ValueError(f"Requested episode indexes are not present in metadata: {missing}")
        selected = list(dict.fromkeys(int(idx) for idx in requested))

    if max_episodes is not None:
        selected = selected[: int(max_episodes)]
    return selected


def _episode_by_index(dataset: Any) -> dict[int, dict[str, Any]]:
    return {
        int(dataset.meta.episodes[idx]["episode_index"]): dataset.meta.episodes[idx]
        for idx in range(len(dataset.meta.episodes))
    }


def _episode_batch(dataset: Any, episode: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = int(episode["dataset_from_index"])
    end = int(episode["dataset_to_index"])
    raw_dataset = dataset.hf_dataset.with_format(None)
    batch = raw_dataset[start:end]

    if "action" not in batch:
        raise KeyError("Input dataset data has no 'action' column.")

    actions = np.asarray(batch["action"], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    if actions.ndim != 2:
        raise ValueError(f"Expected action array [frames, dim], got shape {actions.shape}")

    num_frames = actions.shape[0]
    if "frame_index" in batch:
        frame_indices = np.asarray(batch["frame_index"], dtype=np.int64).reshape(-1)
    else:
        frame_indices = np.arange(num_frames, dtype=np.int64)
    if "index" in batch:
        global_indices = np.asarray(batch["index"], dtype=np.int64).reshape(-1)
    else:
        global_indices = np.arange(start, end, dtype=np.int64)

    if len(frame_indices) != num_frames or len(global_indices) != num_frames:
        raise ValueError(
            "Episode metadata/action length mismatch: "
            f"actions={num_frames} frame_index={len(frame_indices)} index={len(global_indices)}"
        )
    return actions, frame_indices, global_indices


def _fill_invalid(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    warnings: list[str] = []
    if values.size == 0:
        return values.copy(), warnings

    finite = np.isfinite(values)
    invalid_count = int(np.count_nonzero(~finite))
    if invalid_count == 0:
        return values.copy(), warnings
    if not finite.any():
        warnings.append("All gripper values are NaN/inf; leaving side unannotated.")
        return np.zeros_like(values, dtype=np.float32), warnings

    indices = np.arange(values.size)
    filled = values.copy()
    filled[~finite] = np.interp(indices[~finite], indices[finite], values[finite])
    warnings.append(f"Filled {invalid_count} NaN/inf gripper values by linear interpolation.")
    return filled.astype(np.float32), warnings


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.copy()
    if window < 1:
        raise ValueError("--smooth-window must be >= 1")

    left = (window - 1) // 2
    right = window // 2
    padded = np.pad(values, (left, right), mode="edge")
    return np.asarray([np.median(padded[idx : idx + window]) for idx in range(values.size)], dtype=np.float32)


def _infer_mode(values: np.ndarray, requested_mode: str) -> str:
    if requested_mode in {"binary", "continuous"}:
        return requested_mode

    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return "continuous"

    rounded_unique = np.unique(np.round(valid, decimals=3))
    if rounded_unique.size <= 2:
        return "binary"

    if rounded_unique.size <= 4 and valid.size >= 10:
        low = float(np.min(valid))
        high = float(np.max(valid))
        value_range = high - low
        if value_range > 1e-6:
            nearest_endpoint = np.minimum(np.abs(valid - low), np.abs(valid - high))
            endpoint_fraction = float(np.mean(nearest_endpoint <= max(1e-3, 0.05 * value_range)))
            if endpoint_fraction >= 0.95:
                return "binary"

    return "continuous"


def _resolve_open_high(requested: str, feature_name: str | None) -> tuple[bool | None, list[str]]:
    if requested == "true":
        return True, []
    if requested == "false":
        return False, []

    warnings: list[str] = []
    lowered = (feature_name or "").lower()
    has_open = "open" in lowered
    has_close = "close" in lowered or "closed" in lowered

    if has_open and not has_close:
        return True, []
    if has_close and not has_open:
        return False, []
    if "width" in lowered or "aperture" in lowered:
        return True, []

    warnings.append(
        f"Could not infer open-high convention from action name {feature_name!r}; "
        "marking detected transitions as transition_unknown. Pass --open-high true/false."
    )
    return None, warnings


def _event_from_direction(sign: int, open_high: bool | None) -> int:
    if open_high is None or sign == 0:
        return EVENT_TRANSITION_UNKNOWN
    if sign > 0:
        return EVENT_OPENING if open_high else EVENT_CLOSING
    return EVENT_CLOSING if open_high else EVENT_OPENING


def _filter_transition_gap(transitions: list[Transition], min_gap: int) -> list[Transition]:
    if min_gap <= 0 or len(transitions) <= 1:
        return transitions

    clusters: list[list[Transition]] = []
    current: list[Transition] = [transitions[0]]
    for transition in transitions[1:]:
        if transition.frame - current[-1].frame <= min_gap:
            current.append(transition)
        else:
            clusters.append(current)
            current = [transition]
    clusters.append(current)

    filtered: list[Transition] = []
    for cluster in clusters:
        strongest = max(cluster, key=lambda item: (item.strength, -item.frame))
        filtered.append(strongest)
    return sorted(filtered, key=lambda item: item.frame)


def _detect_transitions_derivative(
    values: np.ndarray,
    *,
    mode: str,
    open_high: bool | None,
    delta_threshold: float,
    binary_threshold: float,
    min_transition_gap: int,
) -> list[Transition]:
    if values.size <= 1:
        return []

    transitions: list[Transition] = []
    if mode == "binary":
        state = values >= float(binary_threshold)
        changes = np.flatnonzero(state[1:] != state[:-1]) + 1
        for frame in changes:
            sign = 1 if state[frame] and not state[frame - 1] else -1
            event = _event_from_direction(sign, open_high)
            strength = float(abs(values[frame] - values[frame - 1]))
            transitions.append(Transition(frame=int(frame), event=event, strength=strength))
    else:
        deltas = np.diff(values)
        frames = np.flatnonzero(np.abs(deltas) >= float(delta_threshold)) + 1
        for frame in frames:
            delta = float(deltas[frame - 1])
            sign = 1 if delta > 0 else -1 if delta < 0 else 0
            event = _event_from_direction(sign, open_high)
            transitions.append(Transition(frame=int(frame), event=event, strength=abs(delta)))

    return _filter_transition_gap(transitions, int(min_transition_gap))


def _binary_stable_state(value: float, *, open_high: bool | None, binary_threshold: float) -> int:
    if open_high is False:
        return STABLE_OPEN if value <= float(binary_threshold) else STABLE_CLOSED
    return STABLE_OPEN if value >= float(binary_threshold) else STABLE_CLOSED


def _continuous_stable_state(
    value: float,
    *,
    open_high: bool | None,
    open_threshold: float,
    close_threshold: float,
) -> int:
    if open_high is False:
        if value <= float(close_threshold):
            return STABLE_OPEN
        if value >= float(open_threshold):
            return STABLE_CLOSED
    else:
        if value >= float(open_threshold):
            return STABLE_OPEN
        if value <= float(close_threshold):
            return STABLE_CLOSED
    return STABLE_TRANSITION_ZONE


def _stable_state(
    value: float,
    *,
    mode: str,
    open_high: bool | None,
    binary_threshold: float,
    open_threshold: float,
    close_threshold: float,
) -> int:
    if mode == "binary":
        return _binary_stable_state(
            value,
            open_high=open_high,
            binary_threshold=binary_threshold,
        )
    return _continuous_stable_state(
        value,
        open_high=open_high,
        open_threshold=open_threshold,
        close_threshold=close_threshold,
    )


def _event_from_stable_state_change(previous_state: int, next_state: int, open_high: bool | None) -> int:
    if open_high is None:
        return EVENT_TRANSITION_UNKNOWN
    if previous_state == STABLE_OPEN and next_state == STABLE_CLOSED:
        return EVENT_CLOSING
    if previous_state == STABLE_CLOSED and next_state == STABLE_OPEN:
        return EVENT_OPENING
    return EVENT_TRANSITION_UNKNOWN


def _transition_event_frame(
    *,
    reached_frame: int,
    transition_start_frame: int | None,
    event_frame: str,
) -> int:
    if event_frame == "reached_state":
        return int(reached_frame)

    if transition_start_frame is None:
        return int(reached_frame)

    if event_frame == "start":
        return int(transition_start_frame)

    if event_frame == "midpoint":
        transition_end_frame = max(int(transition_start_frame), int(reached_frame) - 1)
        return int((int(transition_start_frame) + transition_end_frame) // 2)

    raise ValueError(f"Unsupported --event-frame value: {event_frame!r}")


def _detect_transitions_hysteresis(
    values: np.ndarray,
    *,
    mode: str,
    open_high: bool | None,
    binary_threshold: float,
    open_threshold: float,
    close_threshold: float,
    event_frame: str,
    min_transition_gap: int,
) -> list[Transition]:
    if values.size <= 1:
        return []
    if mode != "binary" and float(open_threshold) <= float(close_threshold):
        raise ValueError("--open-threshold must be greater than --close-threshold for hysteresis mode.")

    transitions: list[Transition] = []
    previous_stable_state: int | None = None
    previous_stable_frame: int | None = None
    transition_start_frame: int | None = None

    for frame, value in enumerate(values):
        state = _stable_state(
            float(value),
            mode=mode,
            open_high=open_high,
            binary_threshold=binary_threshold,
            open_threshold=open_threshold,
            close_threshold=close_threshold,
        )

        if state == STABLE_TRANSITION_ZONE:
            if previous_stable_state is not None and transition_start_frame is None:
                transition_start_frame = int(frame)
            continue

        if previous_stable_state is None:
            previous_stable_state = state
            previous_stable_frame = int(frame)
            transition_start_frame = None
            continue

        if state == previous_stable_state:
            previous_stable_frame = int(frame)
            transition_start_frame = None
            continue

        event = _event_from_stable_state_change(previous_stable_state, state, open_high)
        frame_for_event = _transition_event_frame(
            reached_frame=int(frame),
            transition_start_frame=transition_start_frame,
            event_frame=event_frame,
        )
        if previous_stable_frame is None:
            strength = 0.0
        else:
            strength = float(abs(float(values[frame]) - float(values[previous_stable_frame])))
        transitions.append(Transition(frame=frame_for_event, event=event, strength=strength))

        previous_stable_state = state
        previous_stable_frame = int(frame)
        transition_start_frame = None

    return _filter_transition_gap(transitions, int(min_transition_gap))


def _detect_transitions(
    values: np.ndarray,
    *,
    detector: str,
    mode: str,
    open_high: bool | None,
    delta_threshold: float,
    binary_threshold: float,
    open_threshold: float,
    close_threshold: float,
    event_frame: str,
    min_transition_gap: int,
) -> list[Transition]:
    if detector == "derivative":
        return _detect_transitions_derivative(
            values,
            mode=mode,
            open_high=open_high,
            delta_threshold=delta_threshold,
            binary_threshold=binary_threshold,
            min_transition_gap=min_transition_gap,
        )
    if detector == "hysteresis":
        return _detect_transitions_hysteresis(
            values,
            mode=mode,
            open_high=open_high,
            binary_threshold=binary_threshold,
            open_threshold=open_threshold,
            close_threshold=close_threshold,
            event_frame=event_frame,
            min_transition_gap=min_transition_gap,
        )
    raise ValueError(f"Unsupported --detector value: {detector!r}")


def _apply_event_window(
    events: np.ndarray,
    weights: np.ndarray,
    priorities: np.ndarray,
    *,
    start: int,
    end: int,
    event: int,
) -> None:
    if start >= end:
        return
    new_priority = EVENT_PRIORITIES[event]
    new_weight = EVENT_WEIGHTS[event]
    current_priorities = priorities[start:end]
    current_weights = weights[start:end]
    update = (new_priority > current_priorities) | (
        (new_priority == current_priorities) & (new_weight > current_weights)
    )
    if not np.any(update):
        return
    event_slice = events[start:end]
    weight_slice = weights[start:end]
    priority_slice = priorities[start:end]
    event_slice[update] = event
    weight_slice[update] = new_weight
    priority_slice[update] = new_priority


def _events_from_transitions(
    num_frames: int,
    transitions: list[Transition],
    *,
    pre_window: int,
    post_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    events = np.full(num_frames, EVENT_NORMAL, dtype=np.int64)
    weights = np.full(num_frames, EVENT_WEIGHTS[EVENT_NORMAL], dtype=np.float32)
    priorities = np.zeros(num_frames, dtype=np.int64)

    for transition in transitions:
        frame = int(transition.frame)
        if frame < 0 or frame >= num_frames:
            continue
        if transition.event == EVENT_CLOSING:
            pre_event = EVENT_PRE_CLOSING
            post_event = EVENT_POST_CLOSING
        elif transition.event == EVENT_OPENING:
            pre_event = EVENT_PRE_OPENING
            post_event = EVENT_POST_OPENING
        else:
            pre_event = EVENT_TRANSITION_UNKNOWN
            post_event = EVENT_TRANSITION_UNKNOWN

        _apply_event_window(
            events,
            weights,
            priorities,
            start=max(0, frame - int(pre_window)),
            end=frame,
            event=pre_event,
        )
        _apply_event_window(events, weights, priorities, start=frame, end=frame + 1, event=transition.event)
        _apply_event_window(
            events,
            weights,
            priorities,
            start=frame + 1,
            end=min(num_frames, frame + int(post_window) + 1),
            event=post_event,
        )

    return events, weights


def _empty_side_detection(side: str, num_frames: int, warning: str | None = None) -> SideDetection:
    warnings = [warning] if warning else []
    values = np.full(num_frames, np.nan, dtype=np.float32)
    events = np.full(num_frames, EVENT_NORMAL, dtype=np.int64)
    weights = np.full(num_frames, EVENT_WEIGHTS[EVENT_NORMAL], dtype=np.float32)
    return SideDetection(
        side=side,
        dim=None,
        name=None,
        values=values,
        smoothed_values=values.copy(),
        events=events,
        weights=weights,
        mode=None,
        open_high=None,
        transitions=[],
        warnings=warnings,
    )


def _detect_side(
    *,
    side: str,
    actions: np.ndarray,
    dim: int | None,
    action_names: list[str] | None,
    detector: str,
    mode_arg: str,
    open_high_arg: str,
    delta_threshold: float,
    binary_threshold: float,
    open_threshold: float,
    close_threshold: float,
    event_frame: str,
    pre_window: int,
    post_window: int,
    min_transition_gap: int,
    smooth_window: int,
) -> SideDetection:
    num_frames = int(actions.shape[0])
    if dim is None:
        return _empty_side_detection(side, num_frames)
    if dim < 0 or dim >= actions.shape[1]:
        return _empty_side_detection(
            side,
            num_frames,
            warning=f"{side} gripper dim {dim} is outside episode action dim {actions.shape[1]}.",
        )

    name = action_names[dim] if action_names is not None and dim < len(action_names) else None
    raw_values = np.asarray(actions[:, dim], dtype=np.float32)
    values, warnings = _fill_invalid(raw_values)
    if "All gripper values are NaN/inf; leaving side unannotated." in warnings:
        return SideDetection(
            side=side,
            dim=dim,
            name=name,
            values=raw_values,
            smoothed_values=values,
            events=np.full(num_frames, EVENT_NORMAL, dtype=np.int64),
            weights=np.full(num_frames, EVENT_WEIGHTS[EVENT_NORMAL], dtype=np.float32),
            mode=None,
            open_high=None,
            transitions=[],
            warnings=warnings,
        )

    smoothed = _rolling_median(values, int(smooth_window))
    mode = _infer_mode(smoothed, mode_arg)
    open_high, open_high_warnings = _resolve_open_high(open_high_arg, name)
    warnings.extend(open_high_warnings)

    transitions = _detect_transitions(
        smoothed,
        detector=detector,
        mode=mode,
        open_high=open_high,
        delta_threshold=delta_threshold,
        binary_threshold=binary_threshold,
        open_threshold=open_threshold,
        close_threshold=close_threshold,
        event_frame=event_frame,
        min_transition_gap=min_transition_gap,
    )
    events, weights = _events_from_transitions(
        num_frames,
        transitions,
        pre_window=pre_window,
        post_window=post_window,
    )
    return SideDetection(
        side=side,
        dim=dim,
        name=name,
        values=raw_values,
        smoothed_values=smoothed,
        events=events,
        weights=weights,
        mode=mode,
        open_high=open_high,
        transitions=transitions,
        warnings=warnings,
    )


def _combine_events(left: SideDetection, right: SideDetection) -> tuple[np.ndarray, np.ndarray]:
    if left.events.shape != right.events.shape:
        raise ValueError(f"Left/right event length mismatch: {left.events.shape} vs {right.events.shape}")

    combined_events = np.empty_like(left.events)
    combined_weights = np.maximum(left.weights, right.weights).astype(np.float32)
    for idx, (left_event, right_event) in enumerate(zip(left.events, right.events, strict=True)):
        left_event = int(left_event)
        right_event = int(right_event)
        left_key = (EVENT_PRIORITIES[left_event], EVENT_WEIGHTS[left_event])
        right_key = (EVENT_PRIORITIES[right_event], EVENT_WEIGHTS[right_event])
        combined_events[idx] = left_event if left_key >= right_key else right_event
    return combined_events, combined_weights


def _detect_episode(
    *,
    episode_index: int,
    actions: np.ndarray,
    frame_indices: np.ndarray,
    global_indices: np.ndarray,
    dims: GripperDims,
    action_names: list[str] | None,
    args: argparse.Namespace,
) -> EpisodeDetection:
    warnings: list[str] = []
    if actions.shape[0] == 0:
        warnings.append("Episode has zero frames.")
    resolved_dims = [idx for idx in [dims.left, dims.right] if idx is not None]
    if resolved_dims and actions.shape[1] <= max(resolved_dims):
        warnings.append(f"Episode action dim {actions.shape[1]} is smaller than resolved gripper dims.")

    left = _detect_side(
        side=SIDE_LEFT,
        actions=actions,
        dim=dims.left,
        action_names=action_names,
        detector=args.detector,
        mode_arg=args.mode,
        open_high_arg=args.open_high,
        delta_threshold=args.delta_threshold,
        binary_threshold=args.binary_threshold,
        open_threshold=args.open_threshold,
        close_threshold=args.close_threshold,
        event_frame=args.event_frame,
        pre_window=args.pre_window,
        post_window=args.post_window,
        min_transition_gap=args.min_transition_gap,
        smooth_window=args.smooth_window,
    )
    right = _detect_side(
        side=SIDE_RIGHT,
        actions=actions,
        dim=dims.right,
        action_names=action_names,
        detector=args.detector,
        mode_arg=args.mode,
        open_high_arg=args.open_high,
        delta_threshold=args.delta_threshold,
        binary_threshold=args.binary_threshold,
        open_threshold=args.open_threshold,
        close_threshold=args.close_threshold,
        event_frame=args.event_frame,
        pre_window=args.pre_window,
        post_window=args.post_window,
        min_transition_gap=args.min_transition_gap,
        smooth_window=args.smooth_window,
    )
    combined_events, combined_weights = _combine_events(left, right)

    warnings.extend(left.warnings)
    warnings.extend(right.warnings)
    total_transitions = len(left.transitions) + len(right.transitions)
    too_many_threshold = max(10, int(np.ceil(actions.shape[0] * 0.08)))
    if total_transitions > too_many_threshold:
        warnings.append(
            f"Episode has {total_transitions} transitions after gap filtering; "
            f"threshold for 'too many' is {too_many_threshold}."
        )

    return EpisodeDetection(
        episode_index=int(episode_index),
        frame_indices=frame_indices,
        global_indices=global_indices,
        left=left,
        right=right,
        combined_events=combined_events,
        combined_weights=combined_weights,
        warnings=warnings,
    )


def _event_distribution(detections: list[EpisodeDetection]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for detection in detections:
        counter.update(EVENT_NAMES[int(event)] for event in detection.combined_events)
    return dict(counter)


def _weight_distribution(detections: list[EpisodeDetection]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for detection in detections:
        counter.update(f"{float(weight):.6g}" for weight in detection.combined_weights)
    return dict(counter)


def _expected_counts_from_args(args: argparse.Namespace) -> dict[str, dict[str, int | None]]:
    return {
        SIDE_LEFT: {
            "opening": args.expected_left_opening,
            "closing": args.expected_left_closing,
        },
        SIDE_RIGHT: {
            "opening": args.expected_right_opening,
            "closing": args.expected_right_closing,
        },
    }


def _actual_count(detection: EpisodeDetection, side: str, transition_type: str) -> int:
    side_detection = detection.left if side == SIDE_LEFT else detection.right
    if transition_type == "opening":
        return side_detection.opening_count
    if transition_type == "closing":
        return side_detection.closing_count
    raise ValueError(f"Unsupported transition type: {transition_type!r}")


def _expected_count_warning(detection: EpisodeDetection, args: argparse.Namespace) -> str:
    mismatches: list[str] = []
    for side, transition_type, arg_name in EXPECTED_COUNT_FIELDS:
        expected = getattr(args, arg_name)
        if expected is None:
            continue
        actual = _actual_count(detection, side, transition_type)
        if actual != int(expected):
            mismatches.append(f"{side}_{transition_type}={actual} expected={int(expected)}")
    if not mismatches:
        return ""
    return "Unexpected transition counts: " + ", ".join(mismatches)


def _apply_expected_count_check(detection: EpisodeDetection, args: argparse.Namespace) -> None:
    warning = _expected_count_warning(detection, args)
    detection.expected_count_ok = warning == ""
    detection.unexpected_count_warning = warning
    if warning:
        detection.warnings.append(warning)


def _episode_summary(detection: EpisodeDetection) -> dict[str, Any]:
    return {
        "episode_index": int(detection.episode_index),
        "num_frames": detection.num_frames,
        "left_opening_count": detection.left.opening_count,
        "left_closing_count": detection.left.closing_count,
        "left_unknown_count": detection.left.unknown_count,
        "right_opening_count": detection.right.opening_count,
        "right_closing_count": detection.right.closing_count,
        "right_unknown_count": detection.right.unknown_count,
        "keyframe_frame_count": detection.keyframe_frame_count,
        "keyframe_ratio": detection.keyframe_ratio,
        "max_weight": detection.max_weight,
        "expected_count_ok": bool(detection.expected_count_ok),
        "unexpected_count_warning": detection.unexpected_count_warning,
        "warning": " | ".join(detection.warnings),
    }


def _build_report(
    *,
    dataset_root: Path,
    dataset: Any,
    selected_episodes: list[int],
    action_shape: tuple[int, ...],
    action_names: list[str] | None,
    dims: GripperDims,
    detections: list[EpisodeDetection],
    args: argparse.Namespace,
    warnings: list[str],
) -> dict[str, Any]:
    num_frames = sum(detection.num_frames for detection in detections)
    keyframe_ratios = [detection.keyframe_ratio for detection in detections]
    episodes_without_transition = [
        detection.episode_index
        for detection in detections
        if len(detection.left.transitions) + len(detection.right.transitions) == 0
    ]
    episodes_with_too_many_transitions = [
        detection.episode_index
        for detection in detections
        if any("threshold for 'too many'" in warning for warning in detection.warnings)
    ]
    episodes_with_unexpected_transition_count = [
        detection.episode_index for detection in detections if not detection.expected_count_ok
    ]

    return {
        "dataset_root": str(dataset_root),
        "total_dataset_episodes": int(dataset.meta.total_episodes),
        "total_dataset_frames": int(dataset.meta.total_frames),
        "num_episodes": len(detections),
        "num_frames": int(num_frames),
        "selected_episodes": selected_episodes,
        "action_shape": list(action_shape),
        "action_names": action_names,
        "left_gripper_dim": dims.left,
        "right_gripper_dim": dims.right,
        "left_gripper_name": (
            action_names[dims.left] if action_names is not None and dims.left is not None else None
        ),
        "right_gripper_name": (
            action_names[dims.right] if action_names is not None and dims.right is not None else None
        ),
        "mode": args.mode,
        "resolved_modes": {
            "left": sorted(
                {detection.left.mode for detection in detections if detection.left.mode is not None}
            ),
            "right": sorted(
                {detection.right.mode for detection in detections if detection.right.mode is not None}
            ),
        },
        "open_high": args.open_high,
        "resolved_open_high": {
            "left": sorted({str(detection.left.open_high) for detection in detections}),
            "right": sorted({str(detection.right.open_high) for detection in detections}),
        },
        "detector": args.detector,
        "open_threshold": float(args.open_threshold),
        "close_threshold": float(args.close_threshold),
        "event_frame": args.event_frame,
        "pre_window": int(args.pre_window),
        "post_window": int(args.post_window),
        "delta_threshold": float(args.delta_threshold),
        "binary_threshold": float(args.binary_threshold),
        "min_transition_gap": int(args.min_transition_gap),
        "smooth_window": int(args.smooth_window),
        "total_left_opening": int(sum(detection.left.opening_count for detection in detections)),
        "total_left_closing": int(sum(detection.left.closing_count for detection in detections)),
        "total_left_unknown": int(sum(detection.left.unknown_count for detection in detections)),
        "total_right_opening": int(sum(detection.right.opening_count for detection in detections)),
        "total_right_closing": int(sum(detection.right.closing_count for detection in detections)),
        "total_right_unknown": int(sum(detection.right.unknown_count for detection in detections)),
        "episodes_without_transition": episodes_without_transition,
        "episodes_with_too_many_transitions": episodes_with_too_many_transitions,
        "episodes_with_unexpected_transition_count": episodes_with_unexpected_transition_count,
        "unexpected_transition_count_episode_count": len(episodes_with_unexpected_transition_count),
        "expected_counts": _expected_counts_from_args(args),
        "mean_keyframe_ratio": float(np.mean(keyframe_ratios)) if keyframe_ratios else 0.0,
        "max_keyframe_ratio": float(np.max(keyframe_ratios)) if keyframe_ratios else 0.0,
        "event_distribution": _event_distribution(detections),
        "weight_distribution": _weight_distribution(detections),
        "warnings": warnings,
        "episode_summary": [_episode_summary(detection) for detection in detections],
    }


def _prepare_output_dir(output_dir: Path, *, overwrite: bool, plot: bool) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": output_dir / "gripper_transition_report.json",
        "frames": output_dir / "gripper_transition_frames.csv",
        "episodes": output_dir / "gripper_transition_episode_summary.csv",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output files already exist: {existing_text}. Pass --overwrite to replace them."
        )
    if plot:
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        paths["plots"] = plot_dir
    return paths


def _write_frames_csv(path: Path, detections: list[EpisodeDetection]) -> None:
    fieldnames = [
        "episode_index",
        "frame_index",
        "global_index",
        "left_gripper_value",
        "right_gripper_value",
        "left_event",
        "right_event",
        "combined_event",
        "left_event_name",
        "right_event_name",
        "combined_event_name",
        "left_weight",
        "right_weight",
        "combined_weight",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for detection in detections:
            for idx in range(detection.num_frames):
                left_event = int(detection.left.events[idx])
                right_event = int(detection.right.events[idx])
                combined_event = int(detection.combined_events[idx])
                writer.writerow(
                    {
                        "episode_index": int(detection.episode_index),
                        "frame_index": int(detection.frame_indices[idx]),
                        "global_index": int(detection.global_indices[idx]),
                        "left_gripper_value": _csv_float(detection.left.values[idx]),
                        "right_gripper_value": _csv_float(detection.right.values[idx]),
                        "left_event": left_event,
                        "right_event": right_event,
                        "combined_event": combined_event,
                        "left_event_name": EVENT_NAMES[left_event],
                        "right_event_name": EVENT_NAMES[right_event],
                        "combined_event_name": EVENT_NAMES[combined_event],
                        "left_weight": f"{float(detection.left.weights[idx]):.6g}",
                        "right_weight": f"{float(detection.right.weights[idx]):.6g}",
                        "combined_weight": f"{float(detection.combined_weights[idx]):.6g}",
                    }
                )


def _csv_float(value: float | np.floating) -> str:
    value = float(value)
    if not np.isfinite(value):
        return ""
    return f"{value:.9g}"


def _write_episode_summary_csv(path: Path, detections: list[EpisodeDetection]) -> None:
    fieldnames = [
        "episode_index",
        "num_frames",
        "left_opening_count",
        "left_closing_count",
        "right_opening_count",
        "right_closing_count",
        "keyframe_frame_count",
        "keyframe_ratio",
        "max_weight",
        "expected_count_ok",
        "unexpected_count_warning",
        "warning",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for detection in detections:
            summary = _episode_summary(detection)
            writer.writerow({key: summary[key] for key in fieldnames})


def _write_report_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _plot_episode(path: Path, detection: EpisodeDetection) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "matplotlib is not available; rerun without --plot or install matplotlib."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(detection.num_frames)
    fig, ax = plt.subplots(figsize=(14, 5))

    for idx, event in enumerate(detection.combined_events):
        event = int(event)
        if event == EVENT_NORMAL:
            continue
        color = "#f4d35e" if event in (EVENT_PRE_CLOSING, EVENT_PRE_OPENING) else "#ee964b"
        if event in (EVENT_CLOSING, EVENT_OPENING):
            color = "#d62828" if event == EVENT_CLOSING else "#2a9d8f"
        elif event == EVENT_TRANSITION_UNKNOWN:
            color = "#6c757d"
        ax.axvspan(idx - 0.5, idx + 0.5, color=color, alpha=0.18, linewidth=0)

    if detection.left.dim is not None:
        ax.plot(x, detection.left.smoothed_values, label="left gripper", color="#1f77b4", linewidth=1.2)
    if detection.right.dim is not None:
        ax.plot(x, detection.right.smoothed_values, label="right gripper", color="#ff7f0e", linewidth=1.2)
    for transition in detection.left.transitions:
        ax.axvline(transition.frame, color="#1f77b4", linestyle="--", linewidth=0.8, alpha=0.6)
    for transition in detection.right.transitions:
        ax.axvline(transition.frame, color="#ff7f0e", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_title(f"Episode {detection.episode_index} gripper transition dry-run")
    ax.set_xlabel("frame")
    ax.set_ylabel("gripper action")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_outputs(
    paths: dict[str, Path],
    report: dict[str, Any],
    detections: list[EpisodeDetection],
) -> None:
    _write_report_json(paths["report"], report)
    _write_frames_csv(paths["frames"], detections)
    _write_episode_summary_csv(paths["episodes"], detections)
    if "plots" in paths:
        for detection in detections:
            _plot_episode(
                paths["plots"] / f"episode_{detection.episode_index:04d}_gripper_transition.png",
                detection,
            )


def _default_episode_detection(episode: dict[str, Any]) -> EpisodeDetection:
    episode_index = int(episode["episode_index"])
    start = int(episode["dataset_from_index"])
    end = int(episode["dataset_to_index"])
    num_frames = max(0, end - start)
    return EpisodeDetection(
        episode_index=episode_index,
        frame_indices=np.arange(num_frames, dtype=np.int64),
        global_indices=np.arange(start, end, dtype=np.int64),
        left=_empty_side_detection(SIDE_LEFT, num_frames),
        right=_empty_side_detection(SIDE_RIGHT, num_frames),
        combined_events=np.full(num_frames, EVENT_NORMAL, dtype=np.int64),
        combined_weights=np.full(num_frames, EVENT_WEIGHTS[EVENT_NORMAL], dtype=np.float32),
        warnings=[],
    )


def _complete_export_detections(
    episode_lookup: dict[int, dict[str, Any]],
    selected_detections: list[EpisodeDetection],
) -> tuple[list[EpisodeDetection], list[int]]:
    detection_by_episode = {int(detection.episode_index): detection for detection in selected_detections}
    export_detections: list[EpisodeDetection] = []
    default_normal_episode_indexes: list[int] = []
    for episode_index in sorted(episode_lookup):
        detection = detection_by_episode.get(int(episode_index))
        if detection is None:
            detection = _default_episode_detection(episode_lookup[int(episode_index)])
            default_normal_episode_indexes.append(int(episode_index))
        export_detections.append(detection)
    return export_detections, default_normal_episode_indexes


def _build_annotation_arrays(
    *,
    total_frames: int,
    detections: list[EpisodeDetection],
    columns: dict[str, str],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        columns["left_gripper_event"]: np.full(total_frames, EVENT_NORMAL, dtype=np.int64),
        columns["right_gripper_event"]: np.full(total_frames, EVENT_NORMAL, dtype=np.int64),
        columns["gripper_event"]: np.full(total_frames, EVENT_NORMAL, dtype=np.int64),
        columns["left_keyframe_weight"]: np.full(
            total_frames, EVENT_WEIGHTS[EVENT_NORMAL], dtype=np.float32
        ),
        columns["right_keyframe_weight"]: np.full(
            total_frames, EVENT_WEIGHTS[EVENT_NORMAL], dtype=np.float32
        ),
        columns["keyframe_weight"]: np.full(total_frames, EVENT_WEIGHTS[EVENT_NORMAL], dtype=np.float32),
    }

    for detection in detections:
        global_indices = np.asarray(detection.global_indices, dtype=np.int64).reshape(-1)
        if global_indices.size != detection.num_frames:
            raise ValueError(
                f"Episode {detection.episode_index} global index length mismatch: "
                f"{global_indices.size} != {detection.num_frames}"
            )
        if global_indices.size == 0:
            continue
        if int(global_indices.min()) < 0 or int(global_indices.max()) >= total_frames:
            raise ValueError(
                f"Episode {detection.episode_index} global indices are outside total_frames={total_frames}."
            )

        arrays[columns["left_gripper_event"]][global_indices] = detection.left.events.astype(np.int64)
        arrays[columns["right_gripper_event"]][global_indices] = detection.right.events.astype(np.int64)
        arrays[columns["gripper_event"]][global_indices] = detection.combined_events.astype(np.int64)
        arrays[columns["left_keyframe_weight"]][global_indices] = detection.left.weights.astype(np.float32)
        arrays[columns["right_keyframe_weight"]][global_indices] = detection.right.weights.astype(np.float32)
        arrays[columns["keyframe_weight"]][global_indices] = detection.combined_weights.astype(np.float32)

    return arrays


def _check_source_has_no_annotation_fields(source_root: Path, columns: dict[str, str]) -> None:
    annotation_columns = list(columns.values())
    info_path = source_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    source_features = info.get("features", {})
    feature_hits = [column for column in annotation_columns if column in source_features]
    if feature_hits:
        raise ValueError(
            "Source dataset meta/info.json already contains target annotation features: "
            f"{feature_hits}. Use a different --annotation-prefix or an unannotated source dataset."
        )

    parquet_hits: dict[str, list[str]] = {}
    for path in _data_parquet_paths(source_root):
        schema_names = set(_parquet_schema_names(path))
        present = [column for column in annotation_columns if column in schema_names]
        if present:
            parquet_hits[str(path.relative_to(source_root))] = present
    if parquet_hits:
        raise ValueError(
            "Source dataset parquet files already contain target annotation columns: "
            f"{parquet_hits}. Use a different --annotation-prefix or an unannotated source dataset."
        )


def _remove_existing_output_root(output_root: Path) -> None:
    if output_root.is_symlink() or output_root.is_file():
        output_root.unlink()
    else:
        shutil.rmtree(output_root)


def _copy_source_dataset_root(
    *,
    source_root: Path,
    output_root: Path,
    overwrite_output: bool,
    copy_videos: bool,
    copy_mode: str,
) -> list[str]:
    warnings: list[str] = []
    if output_root.exists():
        if not overwrite_output:
            raise FileExistsError(
                f"Output dataset root already exists: {output_root}. Pass --overwrite-output to rebuild it."
            )
        _remove_existing_output_root(output_root)

    def ignore_root_videos(directory: str, names: list[str]) -> set[str]:
        if _resolve_for_safety(Path(directory)) == source_root and (
            not copy_videos or copy_mode == "symlink"
        ):
            return {"videos"} if "videos" in names else set()
        return set()

    shutil.copytree(source_root, output_root, symlinks=False, ignore=ignore_root_videos)

    source_videos = source_root / "videos"
    output_videos = output_root / "videos"
    if copy_videos and copy_mode == "symlink":
        if source_videos.exists():
            output_videos.symlink_to(source_videos, target_is_directory=True)
        else:
            warnings.append("Source dataset has no videos directory to symlink.")
    elif not copy_videos and source_videos.exists():
        warnings.append("Videos were intentionally not copied because --copy-videos false was used.")

    return warnings


def _write_annotation_features_to_info(output_root: Path, columns: dict[str, str]) -> Path:
    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    updated_info = _add_annotation_features_to_info_dict(info, columns)
    info_path.write_text(
        json.dumps(_json_ready(updated_info), ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    return info_path


def _append_annotation_columns_to_parquets(
    *,
    source_root: Path,
    output_root: Path,
    annotation_arrays: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pandas as pd

    annotation_columns = list(annotation_arrays)
    output_paths = _data_parquet_paths(output_root)
    if not output_paths:
        raise FileNotFoundError(f"No data parquet files found under {output_root / 'data'}")

    records: list[dict[str, Any]] = []
    for output_path in output_paths:
        rel_path = output_path.relative_to(output_root)
        source_path = source_root / rel_path
        if not source_path.exists():
            raise FileNotFoundError(f"Copied parquet has no source counterpart: {source_path}")

        source_rows = _parquet_row_count(source_path)
        df = pd.read_parquet(output_path)
        output_rows_before = int(len(df))
        if output_rows_before != source_rows:
            raise ValueError(
                f"Row count mismatch before annotation for {rel_path}: "
                f"source={source_rows} output={output_rows_before}"
            )
        if "index" not in df.columns:
            raise KeyError(f"{rel_path} is missing required global 'index' column.")

        row_indices = df["index"].to_numpy(dtype=np.int64)
        if row_indices.size != output_rows_before:
            raise ValueError(f"{rel_path} index length mismatch: {row_indices.size} != {output_rows_before}")
        if row_indices.size:
            max_index = int(row_indices.max())
            min_index = int(row_indices.min())
            total_frames = len(next(iter(annotation_arrays.values())))
            if min_index < 0 or max_index >= total_frames:
                raise ValueError(
                    f"{rel_path} has index range [{min_index}, {max_index}] "
                    f"outside total_frames={total_frames}."
                )

        for column in annotation_columns:
            values = annotation_arrays[column][row_indices]
            if values.dtype.kind in {"i", "u"}:
                df[column] = values.astype(np.int64)
            else:
                df[column] = values.astype(np.float32)

        df.to_parquet(output_path, index=False)
        output_rows_after = _parquet_row_count(output_path)
        if output_rows_after != output_rows_before:
            raise ValueError(
                f"Row count changed after annotation for {rel_path}: "
                f"before={output_rows_before} after={output_rows_after}"
            )
        records.append(
            {
                "path": str(rel_path),
                "source_rows": source_rows,
                "output_rows_before": output_rows_before,
                "output_rows_after": output_rows_after,
                "added_columns": annotation_columns,
            }
        )

    row_count_check = {
        "ok": all(record["source_rows"] == record["output_rows_after"] for record in records),
        "source_total_rows": int(sum(record["source_rows"] for record in records)),
        "output_total_rows_before": int(sum(record["output_rows_before"] for record in records)),
        "output_total_rows_after": int(sum(record["output_rows_after"] for record in records)),
        "per_file": records,
    }
    return records, row_count_check


def _validate_annotation_export(
    *,
    source_root: Path,
    output_root: Path,
    annotation_columns: list[str],
    source_total_frames: int,
    row_count_check: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, details: Any = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "details": details})

    add_check("output_dataset_root_exists", output_root.exists(), str(output_root))
    add_check(
        "output_dataset_root_differs_from_source",
        _resolve_for_safety(output_root) != _resolve_for_safety(source_root),
    )

    output_parquet_missing: dict[str, list[str]] = {}
    for path in _data_parquet_paths(output_root):
        names = set(_parquet_schema_names(path))
        missing = [column for column in annotation_columns if column not in names]
        if missing:
            output_parquet_missing[str(path.relative_to(output_root))] = missing
    add_check("output_parquets_have_annotation_columns", not output_parquet_missing, output_parquet_missing)

    output_info = json.loads((output_root / "meta" / "info.json").read_text(encoding="utf-8"))
    output_features = output_info.get("features", {})
    missing_features = [column for column in annotation_columns if column not in output_features]
    add_check("output_meta_has_annotation_features", not missing_features, missing_features)

    add_check(
        "output_total_frames_matches_source",
        int(row_count_check["output_total_rows_after"]) == int(source_total_frames),
        row_count_check,
    )
    add_check("output_row_counts_match_source", bool(row_count_check["ok"]), row_count_check)

    source_parquet_hits: dict[str, list[str]] = {}
    for path in _data_parquet_paths(source_root):
        names = set(_parquet_schema_names(path))
        present = [column for column in annotation_columns if column in names]
        if present:
            source_parquet_hits[str(path.relative_to(source_root))] = present
    add_check("source_parquets_have_no_annotation_columns", not source_parquet_hits, source_parquet_hits)

    source_info = json.loads((source_root / "meta" / "info.json").read_text(encoding="utf-8"))
    source_features = source_info.get("features", {})
    source_feature_hits = [column for column in annotation_columns if column in source_features]
    add_check("source_meta_has_no_annotation_features", not source_feature_hits, source_feature_hits)

    try:
        output_dataset = LeRobotDataset(  # type: ignore[operator]
            _infer_repo_id(output_root),
            root=output_root,
            download_videos=False,
        )
        output_columns = list(output_dataset.hf_dataset.column_names)
        missing_loaded_columns = [column for column in annotation_columns if column not in output_columns]
        add_check(
            "output_lerobot_dataset_loads_with_annotation_columns",
            len(missing_loaded_columns) == 0 and len(output_dataset.hf_dataset) == int(source_total_frames),
            {
                "loaded_frames": int(len(output_dataset.hf_dataset)),
                "missing_loaded_columns": missing_loaded_columns,
            },
        )
    except Exception as exc:  # pragma: no cover - depends on local dataset env
        add_check("output_lerobot_dataset_loads_with_annotation_columns", False, repr(exc))

    errors = [check for check in checks if not check["ok"]]
    validation = {"ok": not errors, "checks": checks}
    if errors:
        details = "\n".join(f"- {error['name']}: {error['details']}" for error in errors)
        raise ValueError(f"Annotation export validation failed:\n{details}")
    return validation


def _prepare_annotation_export_paths(
    output_dir: Path,
    *,
    overwrite: bool,
) -> AnnotationExportPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = AnnotationExportPaths(
        report=output_dir / "annotation_export_report.json",
        frames=output_dir / "annotation_export_frames.csv",
        episodes=output_dir / "annotation_export_episode_summary.csv",
    )
    existing = [path for path in (paths.report, paths.frames, paths.episodes) if path.exists()]
    if existing and not overwrite:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Annotation export report files already exist: "
            f"{existing_text}. Pass --overwrite to replace them."
        )
    return paths


def _write_annotation_export_outputs(
    paths: AnnotationExportPaths,
    report: dict[str, Any],
    detections: list[EpisodeDetection],
) -> None:
    _write_report_json(paths.report, report)
    _write_frames_csv(paths.frames, detections)
    _write_episode_summary_csv(paths.episodes, detections)


def _build_annotation_export_report(
    *,
    source_root: Path,
    output_root: Path,
    source_summary_before: dict[str, Any],
    source_summary_after: dict[str, Any],
    output_summary: dict[str, Any],
    selected_episodes: list[int],
    default_normal_episode_indexes: list[int],
    annotation_columns: list[str],
    updated_meta_info_path: Path,
    parquet_update_records: list[dict[str, Any]],
    row_count_check: dict[str, Any],
    validation: dict[str, Any],
    dry_run_report: dict[str, Any],
    export_detections: list[EpisodeDetection],
    args: argparse.Namespace,
    warnings: list[str],
) -> dict[str, Any]:
    source_unchanged = source_summary_before == source_summary_after
    keyframe_ratios = [detection.keyframe_ratio for detection in export_detections]
    return {
        "source_dataset_root": str(source_root),
        "output_dataset_root": str(output_root),
        "source_dataset_fingerprint_or_mtime_summary": {
            "before": source_summary_before,
            "after": source_summary_after,
            "unchanged": source_unchanged,
        },
        "output_dataset_fingerprint_or_mtime_summary": output_summary,
        "num_episodes": int(len(export_detections)),
        "num_frames": int(sum(detection.num_frames for detection in export_detections)),
        "selected_episodes": selected_episodes,
        "default_normal_episode_indexes": default_normal_episode_indexes,
        "annotation_columns": annotation_columns,
        "updated_meta_info_path": str(updated_meta_info_path),
        "parquet_files_updated": parquet_update_records,
        "row_count_check": row_count_check,
        "expected_count_check": {
            "expected_counts": dry_run_report["expected_counts"],
            "episodes_with_unexpected_transition_count": dry_run_report[
                "episodes_with_unexpected_transition_count"
            ],
            "unexpected_transition_count_episode_count": dry_run_report[
                "unexpected_transition_count_episode_count"
            ],
            "strict_expected_counts": bool(args.strict_expected_counts),
        },
        "event_distribution": _event_distribution(export_detections),
        "weight_distribution": _weight_distribution(export_detections),
        "mean_keyframe_ratio": float(np.mean(keyframe_ratios)) if keyframe_ratios else 0.0,
        "max_keyframe_ratio": float(np.max(keyframe_ratios)) if keyframe_ratios else 0.0,
        "copy_videos": args.copy_videos,
        "copy_mode": args.copy_mode,
        "validation": validation,
        "source_dataset_unchanged": source_unchanged,
        "warnings": warnings,
        "episode_summary": [_episode_summary(detection) for detection in export_detections],
    }


def _format_action_names(action_names: list[str] | None) -> str:
    if action_names is None:
        return "<none>"
    preview = [f"{idx}:{name}" for idx, name in enumerate(action_names[:24])]
    suffix = "" if len(action_names) <= 24 else f", ... ({len(action_names)} total)"
    return ", ".join(preview) + suffix


def _print_summary(report: dict[str, Any], paths: dict[str, Path]) -> None:
    warnings = report["warnings"]
    print("Dataset:")
    print(f"  root: {report['dataset_root']}")
    print(f"Action names: {_format_action_names(report['action_names'])}")
    print("Detected gripper dims:")
    print(f"  left: {report['left_gripper_dim']} ({report['left_gripper_name']})")
    print(f"  right: {report['right_gripper_dim']} ({report['right_gripper_name']})")
    print(f"Detector: {report['detector']}")
    print(f"Open threshold: {report['open_threshold']}")
    print(f"Close threshold: {report['close_threshold']}")
    print(f"Event frame: {report['event_frame']}")
    print(f"Mode: requested={report['mode']} resolved={report['resolved_modes']}")
    print(f"Open-high convention: requested={report['open_high']} resolved={report['resolved_open_high']}")
    print(f"Episodes: {report['num_episodes']} of {report['total_dataset_episodes']}")
    print(f"Frames: {report['num_frames']}")
    print(f"Left opening / closing: {report['total_left_opening']} / {report['total_left_closing']}")
    print(f"Right opening / closing: {report['total_right_opening']} / {report['total_right_closing']}")
    if report["total_left_unknown"] or report["total_right_unknown"]:
        print(
            "Unknown transitions left/right: "
            f"{report['total_left_unknown']} / {report['total_right_unknown']}"
        )
    unexpected_episodes = report["episodes_with_unexpected_transition_count"]
    print(f"Unexpected transition count episodes: {len(unexpected_episodes)}")
    if unexpected_episodes:
        preview = ", ".join(str(idx) for idx in unexpected_episodes[:20])
        suffix = "" if len(unexpected_episodes) <= 20 else f", ... ({len(unexpected_episodes)} total)"
        print(f"  {preview}{suffix}")
    print(f"Mean keyframe ratio: {report['mean_keyframe_ratio']:.4f}")
    print("Output files:")
    print(f"  {paths['report']}")
    print(f"  {paths['frames']}")
    print(f"  {paths['episodes']}")
    if "plots" in paths:
        print(f"  {paths['plots']}/")
    print(f"Warnings: {len(warnings)}")
    for warning in warnings[:12]:
        print(f"  - {warning}")
    if len(warnings) > 12:
        print(f"  - ... {len(warnings) - 12} more; see gripper_transition_report.json")


def _print_export_summary(report: dict[str, Any], paths: AnnotationExportPaths) -> None:
    print("Annotation export:")
    print(f"  source: {report['source_dataset_root']}")
    print(f"  output: {report['output_dataset_root']}")
    print(f"  annotation columns: {', '.join(report['annotation_columns'])}")
    print(f"  updated meta/info.json: {report['updated_meta_info_path']}")
    print(f"  parquet files updated: {len(report['parquet_files_updated'])}")
    print(
        "  row counts: "
        f"source={report['row_count_check']['source_total_rows']} "
        f"output={report['row_count_check']['output_total_rows_after']} "
        f"ok={report['row_count_check']['ok']}"
    )
    print(f"  source unchanged: {report['source_dataset_unchanged']}")
    print(f"  output validation ok: {report['validation']['ok']}")
    print(f"  default-normal episodes: {len(report['default_normal_episode_indexes'])}")
    print("Output files:")
    print(f"  {paths.report}")
    print(f"  {paths.frames}")
    print(f"  {paths.episodes}")
    warnings = report["warnings"]
    print(f"Warnings: {len(warnings)}")
    for warning in warnings[:12]:
        print(f"  - {warning}")
    if len(warnings) > 12:
        print(f"  - ... {len(warnings) - 12} more; see annotation_export_report.json")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gripper transition detector and annotation exporter for LeRobot datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to an existing LeRobot dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for report JSON/CSV/plots. "
            "Required for dry-run; export defaults to output dataset root."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write diagnostic reports only. Cannot be combined with --export-annotated-copy.",
    )
    parser.add_argument(
        "--export-annotated-copy",
        action="store_true",
        help="Copy the input dataset and add gripper transition annotation columns in the copy.",
    )
    parser.add_argument(
        "--output-dataset-root",
        type=Path,
        default=None,
        help="Destination dataset root for --export-annotated-copy.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete and rebuild --output-dataset-root if it already exists.",
    )
    parser.add_argument(
        "--annotation-prefix",
        default="annotation",
        help="Prefix for annotation columns added to the exported dataset.",
    )
    parser.add_argument(
        "--copy-videos",
        choices=("true", "false"),
        default="true",
        help="Whether the exported dataset should keep videos.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="Copy all files, or symlink the root videos directory while copying metadata/parquet files.",
    )
    parser.add_argument(
        "--validate-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate annotation columns, row counts, source immutability, and LeRobotDataset loading.",
    )
    parser.add_argument(
        "--episode-indexes",
        nargs="+",
        default=None,
        help="Episode indexes, comma list, or Python-style ranges such as '0,2,5:10'.",
    )
    parser.add_argument("--max-episodes", type=_positive_int, default=None, help="Limit processed episodes.")
    parser.add_argument(
        "--left-gripper-dim",
        type=int,
        default=None,
        help="Explicit left gripper action dim.",
    )
    parser.add_argument(
        "--right-gripper-dim",
        type=int,
        default=None,
        help="Explicit right gripper action dim.",
    )
    parser.add_argument(
        "--gripper-name-regex",
        default="gripper",
        help="Regex used to find candidate gripper action names in metadata.",
    )
    parser.add_argument(
        "--open-high",
        choices=("true", "false", "auto"),
        default="auto",
        help="Whether larger gripper values mean opening.",
    )
    parser.add_argument("--mode", choices=("auto", "binary", "continuous"), default="auto")
    parser.add_argument(
        "--detector",
        choices=("hysteresis", "derivative"),
        default="hysteresis",
        help="Transition detector to use; derivative preserves the previous delta-threshold behavior.",
    )
    parser.add_argument("--delta-threshold", type=float, default=0.05)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument(
        "--open-threshold",
        type=float,
        default=0.8,
        help="Stable high threshold for the hysteresis detector.",
    )
    parser.add_argument(
        "--close-threshold",
        type=float,
        default=0.2,
        help="Stable low threshold for the hysteresis detector.",
    )
    parser.add_argument(
        "--event-frame",
        choices=("reached_state", "start", "midpoint"),
        default="reached_state",
        help="Frame used to label a hysteresis transition event.",
    )
    parser.add_argument("--pre-window", type=_positive_int, default=5)
    parser.add_argument("--post-window", type=_positive_int, default=8)
    parser.add_argument("--min-transition-gap", type=_positive_int, default=3)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Rolling median window; 1 disables smoothing.",
    )
    parser.add_argument("--expected-left-opening", type=_positive_int, default=None)
    parser.add_argument("--expected-left-closing", type=_positive_int, default=None)
    parser.add_argument("--expected-right-opening", type=_positive_int, default=None)
    parser.add_argument("--expected-right-closing", type=_positive_int, default=None)
    parser.add_argument(
        "--strict-expected-counts",
        action="store_true",
        help="Return a non-zero exit code when any episode does not match expected transition counts.",
    )
    parser.add_argument("--plot", action="store_true", help="Write per-episode transition plots.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing report files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.export_annotated_copy and args.dry_run:
        parser.error("--export-annotated-copy cannot be combined with --dry-run.")
    if args.export_annotated_copy and args.output_dataset_root is None:
        parser.error("--export-annotated-copy requires --output-dataset-root.")
    if not args.export_annotated_copy and args.output_dataset_root is not None:
        parser.error("--output-dataset-root is only valid with --export-annotated-copy.")
    if not args.export_annotated_copy and args.output_dir is None:
        parser.error("--output-dir is required for dry-run diagnostics.")
    if args.smooth_window < 1:
        parser.error("--smooth-window must be >= 1")
    if args.delta_threshold < 0:
        parser.error("--delta-threshold must be >= 0")
    if args.binary_threshold < 0:
        parser.error("--binary-threshold must be >= 0")
    if args.detector == "hysteresis" and args.open_threshold <= args.close_threshold:
        parser.error("--open-threshold must be greater than --close-threshold")

    _require_lerobot()
    dataset_root = args.dataset_root.expanduser().resolve(strict=False)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not (dataset_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Dataset root is missing meta/info.json: {dataset_root}")

    annotation_columns = _annotation_columns(args.annotation_prefix)
    output_dataset_root: Path | None = None
    output_dir: Path | None = args.output_dir.expanduser().resolve(strict=False) if args.output_dir else None
    if args.export_annotated_copy:
        output_dataset_root = args.output_dataset_root.expanduser().resolve(strict=False)
        _validate_output_dataset_path(dataset_root, output_dataset_root)
        if output_dir is None:
            output_dir = output_dataset_root
        _validate_export_report_dir(dataset_root, output_dir)
        _check_source_has_no_annotation_fields(dataset_root, annotation_columns)

    dataset = LeRobotDataset(  # type: ignore[operator]
        _infer_repo_id(dataset_root),
        root=dataset_root,
        download_videos=False,
    )
    action_shape, action_dim, action_names = _action_schema(dataset)
    dims = _resolve_gripper_dims(
        action_names=action_names,
        action_shape=action_shape,
        action_dim=action_dim,
        left_dim=args.left_gripper_dim,
        right_dim=args.right_gripper_dim,
        gripper_name_regex=args.gripper_name_regex,
    )
    selected_episodes = _select_episode_indexes(
        dataset,
        _parse_episode_indexes(args.episode_indexes),
        args.max_episodes,
    )

    episode_lookup = _episode_by_index(dataset)
    detections: list[EpisodeDetection] = []
    warnings: list[str] = list(dims.warnings)
    for episode_index in selected_episodes:
        try:
            actions, frame_indices, global_indices = _episode_batch(
                dataset,
                episode_lookup[int(episode_index)],
            )
            if actions.shape[1] != action_dim:
                raise ValueError(
                    f"Episode action dim {actions.shape[1]} does not match metadata action_dim={action_dim}"
                )
            detection = _detect_episode(
                episode_index=int(episode_index),
                actions=actions,
                frame_indices=frame_indices,
                global_indices=global_indices,
                dims=dims,
                action_names=action_names,
                args=args,
            )
        except Exception as exc:  # Keep processing other episodes after data-level failures.
            ep = episode_lookup[int(episode_index)]
            start = int(ep.get("dataset_from_index", 0))
            end = int(ep.get("dataset_to_index", start))
            num_frames = max(0, end - start)
            warning = f"Episode {episode_index}: skipped after error: {exc}"
            warnings.append(warning)
            detection = EpisodeDetection(
                episode_index=int(episode_index),
                frame_indices=np.arange(num_frames, dtype=np.int64),
                global_indices=np.arange(start, end, dtype=np.int64),
                left=_empty_side_detection(SIDE_LEFT, num_frames, warning=warning),
                right=_empty_side_detection(SIDE_RIGHT, num_frames),
                combined_events=np.full(num_frames, EVENT_NORMAL, dtype=np.int64),
                combined_weights=np.full(num_frames, EVENT_WEIGHTS[EVENT_NORMAL], dtype=np.float32),
                warnings=[warning],
            )

        _apply_expected_count_check(detection, args)
        if detection.warnings:
            warnings.extend(f"Episode {episode_index}: {warning}" for warning in detection.warnings)
        detections.append(detection)

    detection_report = _build_report(
        dataset_root=dataset_root,
        dataset=dataset,
        selected_episodes=selected_episodes,
        action_shape=action_shape,
        action_names=action_names,
        dims=dims,
        detections=detections,
        args=args,
        warnings=warnings,
    )

    if not args.export_annotated_copy:
        assert output_dir is not None
        output_paths = _prepare_output_dir(output_dir, overwrite=args.overwrite, plot=args.plot)
        _write_outputs(output_paths, detection_report, detections)
        _print_summary(detection_report, output_paths)
        if args.strict_expected_counts and detection_report["episodes_with_unexpected_transition_count"]:
            return 2
        return 0

    assert output_dataset_root is not None
    assert output_dir is not None
    export_detections, default_normal_episode_indexes = _complete_export_detections(
        episode_lookup,
        detections,
    )
    export_warnings = list(warnings)
    if default_normal_episode_indexes:
        preview = ", ".join(str(idx) for idx in default_normal_episode_indexes[:20])
        suffix = (
            ""
            if len(default_normal_episode_indexes) <= 20
            else f", ... ({len(default_normal_episode_indexes)} total)"
        )
        export_warnings.append(
            "Some episodes were not selected for detection and were exported with normal annotations: "
            f"{preview}{suffix}"
        )

    source_summary_before = _dataset_mtime_summary(dataset_root, list(annotation_columns.values()))
    copy_warnings = _copy_source_dataset_root(
        source_root=dataset_root,
        output_root=output_dataset_root,
        overwrite_output=bool(args.overwrite_output),
        copy_videos=args.copy_videos == "true",
        copy_mode=args.copy_mode,
    )
    export_warnings.extend(copy_warnings)
    annotation_arrays = _build_annotation_arrays(
        total_frames=int(dataset.meta.total_frames),
        detections=export_detections,
        columns=annotation_columns,
    )
    parquet_update_records, row_count_check = _append_annotation_columns_to_parquets(
        source_root=dataset_root,
        output_root=output_dataset_root,
        annotation_arrays=annotation_arrays,
    )
    updated_meta_info_path = _write_annotation_features_to_info(output_dataset_root, annotation_columns)

    validation: dict[str, Any]
    if args.validate_output:
        validation = _validate_annotation_export(
            source_root=dataset_root,
            output_root=output_dataset_root,
            annotation_columns=list(annotation_columns.values()),
            source_total_frames=int(dataset.meta.total_frames),
            row_count_check=row_count_check,
        )
    else:
        validation = {"ok": None, "checks": [], "skipped": True}
        export_warnings.append("--validate-output was disabled; output dataset validation was skipped.")

    source_summary_after = _dataset_mtime_summary(dataset_root, list(annotation_columns.values()))
    output_summary = _dataset_mtime_summary(output_dataset_root, list(annotation_columns.values()))

    export_report = _build_annotation_export_report(
        source_root=dataset_root,
        output_root=output_dataset_root,
        source_summary_before=source_summary_before,
        source_summary_after=source_summary_after,
        output_summary=output_summary,
        selected_episodes=selected_episodes,
        default_normal_episode_indexes=default_normal_episode_indexes,
        annotation_columns=list(annotation_columns.values()),
        updated_meta_info_path=updated_meta_info_path,
        parquet_update_records=parquet_update_records,
        row_count_check=row_count_check,
        validation=validation,
        dry_run_report=detection_report,
        export_detections=export_detections,
        args=args,
        warnings=export_warnings,
    )
    export_report_overwrite = (
        args.overwrite or _resolve_for_safety(output_dir) == _resolve_for_safety(output_dataset_root)
    )
    export_paths = _prepare_annotation_export_paths(output_dir, overwrite=export_report_overwrite)
    _write_annotation_export_outputs(export_paths, export_report, export_detections)
    _print_export_summary(export_report, export_paths)

    if args.strict_expected_counts and detection_report["episodes_with_unexpected_transition_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
