#!/usr/bin/env python3
"""Offline phase annotation for dual-arm LeRobot datasets.

This script uses demonstration actions to infer which arm is active in each
frame, then writes a new dataset whose ``observation.state`` has two appended
phase dimensions:

    phase_right_arm, phase_left_arm

Using expert actions is valid for offline dataset annotation. Do not use this
logic directly at online inference time: deployment phase should come from an
execution state machine or another runtime signal that is available without
expert actions.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    torch = None

DEFAULT_FEATURES = {
    "timestamp": {},
    "frame_index": {},
    "episode_index": {},
    "index": {},
    "task_index": {},
}
LeRobotDataset = None
get_feature_stats = None
load_stats = None
write_info = None
write_stats = None
_LEROBOT_IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    from lerobot.datasets.compute_stats import get_feature_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import DEFAULT_FEATURES, load_stats, write_info, write_stats
except ModuleNotFoundError as exc:  # pragma: no cover - support running from repo checkout
    _LEROBOT_IMPORT_ERROR = exc
    repo_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo_root / "src"))
    try:
        from lerobot.datasets.compute_stats import get_feature_stats
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.utils import DEFAULT_FEATURES, load_stats, write_info, write_stats
    except ModuleNotFoundError as fallback_exc:
        _LEROBOT_IMPORT_ERROR = fallback_exc


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


PHASE_RIGHT = 0
PHASE_LEFT = 1
LABEL_IDLE = 2
LABEL_AMBIGUOUS = 3

PHASE_NAMES = ("phase_right_arm", "phase_left_arm")
LABEL_NAMES = {
    PHASE_RIGHT: "right",
    PHASE_LEFT: "left",
    LABEL_IDLE: "idle",
    LABEL_AMBIGUOUS: "ambiguous",
}


def _require_lerobot() -> None:
    if (
        LeRobotDataset is None
        or get_feature_stats is None
        or load_stats is None
        or write_info is None
        or write_stats is None
    ):
        raise ModuleNotFoundError(
            "LeRobot and its dataset dependencies are not importable in this Python environment. "
            "Activate the same environment used for LeRobot training/recording before running annotation."
        ) from _LEROBOT_IMPORT_ERROR


@dataclass
class Segment:
    label: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class EpisodeAnnotation:
    episode_index: int
    phases: np.ndarray
    left_score: np.ndarray
    right_score: np.ndarray
    smoothed_dominance: np.ndarray
    initial_labels: np.ndarray
    preliminary_labels: np.ndarray
    report: dict[str, Any]
    warnings: list[str]


def _resolve_for_safety(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _paths_overlap(left: Path, right: Path) -> bool:
    left = _resolve_for_safety(left)
    right = _resolve_for_safety(right)
    return left == right or left in right.parents or right in left.parents


def _path_inside(path: Path, root: Path) -> bool:
    path = _resolve_for_safety(path)
    root = _resolve_for_safety(root)
    return path == root or root in path.parents


def _infer_repo_id(root: Path, explicit_repo_id: str | None) -> str:
    if explicit_repo_id:
        return explicit_repo_id
    root = root.expanduser()
    if root.parent.name:
        # Prefer a stable local id over an empty placeholder. The repo id is
        # only needed by LeRobot metadata loading when root is explicitly set.
        return root.name
    return "local_phase_dataset"


def _output_repo_id(source_repo_id: str, explicit_repo_id: str | None) -> str:
    if explicit_repo_id:
        return explicit_repo_id
    source_repo_id = source_repo_id.rstrip("/")
    return f"{source_repo_id}_phase"


def _parse_slice(text: str, action_dim: int, *, label: str) -> list[int]:
    value = str(text).strip()
    if "," in value:
        indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        parts = value.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"{label} must be a slice like '0:6' or comma indices, got {text!r}")
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if parts[1] else None
        step = int(parts[2]) if len(parts) == 3 and parts[2] else None
        normalized = slice(start, stop, step).indices(action_dim)
        start_i, stop_i, step_i = normalized
        if step_i != 1:
            raise ValueError(f"{label} must use step 1, got {text!r}")
        indices = list(range(start_i, stop_i, step_i))

    if len(indices) < 6:
        raise ValueError(f"{label} must select at least 6 delta_ee_pose dims, got {indices}")
    invalid = [idx for idx in indices if idx < 0 or idx >= action_dim]
    if invalid:
        raise ValueError(f"{label} has indices outside action dim {action_dim}: {invalid}")
    return indices


def _odd_window(window: int) -> int:
    window = int(window)
    if window < 1:
        raise ValueError("--smooth-window must be >= 1")
    if window % 2 == 0:
        raise ValueError("--smooth-window must be odd")
    return window


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = _odd_window(window)
    values = np.asarray(values, dtype=np.float32)
    if window <= 1 or values.size == 0:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _contiguous_segments(labels: np.ndarray) -> list[Segment]:
    labels = np.asarray(labels)
    if labels.size == 0:
        return []

    segments: list[Segment] = []
    start = 0
    current = int(labels[0])
    for idx in range(1, len(labels)):
        label = int(labels[idx])
        if label == current:
            continue
        segments.append(Segment(current, start, idx))
        start = idx
        current = label
    segments.append(Segment(current, start, len(labels)))
    return segments


def _nearest_arm_label(labels: np.ndarray, start: int, end: int) -> tuple[int | None, int | None]:
    previous = None
    for idx in range(start - 1, -1, -1):
        label = int(labels[idx])
        if label in (PHASE_RIGHT, PHASE_LEFT):
            previous = label
            break

    next_label = None
    for idx in range(end, len(labels)):
        label = int(labels[idx])
        if label in (PHASE_RIGHT, PHASE_LEFT):
            next_label = label
            break

    return previous, next_label


def _fill_unknown_with_neighbors(labels: np.ndarray, unknown_labels: set[int]) -> np.ndarray:
    out = labels.copy()
    for segment in _contiguous_segments(out):
        if segment.label not in unknown_labels:
            continue
        previous, next_label = _nearest_arm_label(out, segment.start, segment.end)
        if previous is None and next_label is None:
            out[segment.start : segment.end] = PHASE_RIGHT
        elif previous is None:
            out[segment.start : segment.end] = next_label
        elif next_label is None:
            out[segment.start : segment.end] = previous
        elif previous == next_label:
            out[segment.start : segment.end] = previous
        else:
            mid = segment.start + segment.length // 2
            out[segment.start : mid] = previous
            out[mid : segment.end] = next_label
    return out


def _delete_short_active_segments(labels: np.ndarray, min_active_frames: int) -> np.ndarray:
    if min_active_frames <= 1:
        return labels.copy()
    out = labels.copy()
    for segment in _contiguous_segments(out):
        if segment.label in (PHASE_RIGHT, PHASE_LEFT) and segment.length < min_active_frames:
            out[segment.start : segment.end] = LABEL_IDLE
    return out


def _fill_short_idle_gaps(labels: np.ndarray, max_idle_gap_frames: int) -> np.ndarray:
    if max_idle_gap_frames <= 0:
        return labels.copy()
    out = labels.copy()
    for segment in _contiguous_segments(out):
        if segment.label != LABEL_IDLE or segment.length > max_idle_gap_frames:
            continue
        previous, next_label = _nearest_arm_label(out, segment.start, segment.end)
        if previous is None and next_label is None:
            continue
        if previous is None:
            out[segment.start : segment.end] = next_label
        elif next_label is None:
            out[segment.start : segment.end] = previous
        elif previous == next_label:
            out[segment.start : segment.end] = previous
        else:
            mid = segment.start + segment.length // 2
            out[segment.start : mid] = previous
            out[mid : segment.end] = next_label
    return out


def _resolve_ambiguous_labels(
    initial_labels: np.ndarray,
    left_score: np.ndarray,
    right_score: np.ndarray,
) -> np.ndarray:
    labels = initial_labels.copy()
    ambiguous = labels == LABEL_AMBIGUOUS
    if not ambiguous.any():
        return labels

    diff = right_score - left_score
    scale = np.maximum(1.0, np.maximum(left_score, right_score))
    close = np.abs(diff) <= (1e-6 * scale)

    right_mask = ambiguous & (diff > 0) & ~close
    left_mask = ambiguous & (diff < 0) & ~close
    close_mask = ambiguous & close
    labels[right_mask] = PHASE_RIGHT
    labels[left_mask] = PHASE_LEFT
    labels[close_mask] = LABEL_AMBIGUOUS
    return _fill_unknown_with_neighbors(labels, {LABEL_AMBIGUOUS})


def _smooth_initial_labels(
    initial_labels: np.ndarray,
    left_score: np.ndarray,
    right_score: np.ndarray,
    *,
    min_active_frames: int,
    max_idle_gap_frames: int,
) -> np.ndarray:
    labels = _resolve_ambiguous_labels(initial_labels, left_score, right_score)
    labels = _delete_short_active_segments(labels, min_active_frames)
    labels = _fill_short_idle_gaps(labels, max_idle_gap_frames)
    labels = _fill_unknown_with_neighbors(labels, {LABEL_IDLE, LABEL_AMBIGUOUS})
    return labels


def _compute_scores(
    actions: np.ndarray,
    left_indices: list[int],
    right_indices: list[int],
    *,
    trans_threshold: float,
    rot_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if trans_threshold <= 0:
        raise ValueError("--trans-threshold must be > 0")
    if rot_threshold <= 0:
        raise ValueError("--rot-threshold must be > 0")

    left_delta = actions[:, left_indices[:6]]
    right_delta = actions[:, right_indices[:6]]

    left_trans = np.linalg.norm(left_delta[:, 0:3], axis=1)
    left_rot = np.linalg.norm(left_delta[:, 3:6], axis=1)
    right_trans = np.linalg.norm(right_delta[:, 0:3], axis=1)
    right_rot = np.linalg.norm(right_delta[:, 3:6], axis=1)

    left_score = np.maximum(left_trans / trans_threshold, left_rot / rot_threshold)
    right_score = np.maximum(right_trans / trans_threshold, right_rot / rot_threshold)
    return left_score.astype(np.float32), right_score.astype(np.float32)


def _initial_labels(left_score: np.ndarray, right_score: np.ndarray, dominance_ratio: float) -> np.ndarray:
    if dominance_ratio <= 1.0:
        raise ValueError("--dominance-ratio must be > 1.0")

    labels = np.full(left_score.shape, LABEL_AMBIGUOUS, dtype=np.int64)
    idle = (left_score < 1.0) & (right_score < 1.0)
    right = (~idle) & (right_score > left_score * dominance_ratio)
    left = (~idle) & (left_score > right_score * dominance_ratio)

    labels[idle] = LABEL_IDLE
    labels[right] = PHASE_RIGHT
    labels[left] = PHASE_LEFT
    return labels


def _find_crossing_near(smoothed_dominance: np.ndarray, center: int, radius: int) -> int | None:
    if smoothed_dominance.size == 0:
        return None
    start = max(1, center - radius)
    end = min(len(smoothed_dominance), center + radius + 1)
    candidates = [
        idx
        for idx in range(start, end)
        if smoothed_dominance[idx - 1] > 0.0 and smoothed_dominance[idx] <= 0.0
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda idx: abs(idx - center))


def _find_main_boundary(
    preliminary_labels: np.ndarray,
    left_score: np.ndarray,
    right_score: np.ndarray,
    smoothed_dominance: np.ndarray,
    *,
    min_active_frames: int,
    smooth_window: int,
) -> tuple[int, list[str]]:
    warnings: list[str] = []
    length = len(preliminary_labels)
    if length == 0:
        warnings.append("empty episode; no boundary was computed")
        return 0, warnings

    left_segments = [
        segment
        for segment in _contiguous_segments(preliminary_labels)
        if segment.label == PHASE_LEFT
        and segment.length >= max(1, min_active_frames)
        and float(np.mean(left_score[segment.start : segment.end])) >= 1.0
    ]

    if not left_segments:
        warnings.append("no reliable left active segment found; falling back to all right_arm")
        return length, warnings

    selected = None
    for segment in left_segments:
        has_right_before = bool(
            (preliminary_labels[: segment.start] == PHASE_RIGHT).any()
            or (right_score[: segment.start] >= 1.0).any()
        )
        if has_right_before:
            selected = segment
            break

    if selected is None:
        selected = left_segments[0]
        warnings.append("left active segment found without clear right prefix; boundary may be unreliable")

    search_radius = max(min_active_frames, smooth_window)
    crossing = _find_crossing_near(smoothed_dominance, selected.start, search_radius)
    boundary = int(crossing if crossing is not None else selected.start)

    if boundary <= 0:
        warnings.append("boundary was at frame 0; clamped to keep at least one right_arm frame")
        boundary = 1
    if boundary >= length:
        warnings.append("boundary was beyond episode end; falling back to all right_arm")
        boundary = length
    return boundary, warnings


def _count_arm_segments(labels: np.ndarray, label: int) -> int:
    return sum(1 for segment in _contiguous_segments(labels) if segment.label == label)


def _first_gripper_close_frame(actions: np.ndarray, dim: int | None) -> int | None:
    if dim is None:
        return None
    if dim < 0 or dim >= actions.shape[1]:
        raise ValueError(f"gripper dim {dim} is outside action dim {actions.shape[1]}")
    values = np.asarray(actions[:, dim], dtype=np.float32)
    if values.size == 0:
        return None

    # Current NERO/Franka command convention is open fraction: 1=open, 0=closed.
    closed = values <= 0.5
    if closed[0]:
        return 0
    crossing = np.flatnonzero((~closed[:-1]) & closed[1:])
    if crossing.size:
        return int(crossing[0] + 1)
    return None


def _phase_array_from_boundary(length: int, boundary: int) -> np.ndarray:
    phases = np.zeros(length, dtype=np.int64)
    phases[max(0, min(length, boundary)) :] = PHASE_LEFT
    if boundary >= length:
        phases[:] = PHASE_RIGHT
    return phases


def annotate_episode(
    *,
    episode_index: int,
    actions: np.ndarray,
    left_indices: list[int],
    right_indices: list[int],
    args: argparse.Namespace,
    fps: float,
) -> EpisodeAnnotation:
    left_score, right_score = _compute_scores(
        actions,
        left_indices,
        right_indices,
        trans_threshold=float(args.trans_threshold),
        rot_threshold=float(args.rot_threshold),
    )
    initial = _initial_labels(left_score, right_score, float(args.dominance_ratio))
    min_active_frames = max(1, int(round(float(args.min_active_sec) * fps)))
    max_idle_gap_frames = max(0, int(round(float(args.max_idle_gap_sec) * fps)))
    preliminary = _smooth_initial_labels(
        initial,
        left_score,
        right_score,
        min_active_frames=min_active_frames,
        max_idle_gap_frames=max_idle_gap_frames,
    )

    dominance = right_score - left_score
    smoothed_dominance = _moving_average(dominance, int(args.smooth_window))
    warnings: list[str] = []

    if args.use_main_boundary:
        boundary_frame, boundary_warnings = _find_main_boundary(
            preliminary,
            left_score,
            right_score,
            smoothed_dominance,
            min_active_frames=min_active_frames,
            smooth_window=int(args.smooth_window),
        )
        warnings.extend(boundary_warnings)
        phases = _phase_array_from_boundary(len(actions), boundary_frame)
    else:
        phases = preliminary.copy()
        phases = _fill_unknown_with_neighbors(phases, {LABEL_IDLE, LABEL_AMBIGUOUS})
        if not ((phases == PHASE_RIGHT).any() or (phases == PHASE_LEFT).any()):
            warnings.append("no arm phase found after smoothing; falling back to all right_arm")
            phases[:] = PHASE_RIGHT
        boundary_candidates = np.flatnonzero((phases[:-1] == PHASE_RIGHT) & (phases[1:] == PHASE_LEFT))
        boundary_frame = int(boundary_candidates[0] + 1) if boundary_candidates.size else len(actions)

    length = int(len(actions))
    right_count = int((phases == PHASE_RIGHT).sum())
    left_count = int((phases == PHASE_LEFT).sum())
    if right_count == 0 and length > 0:
        warnings.append("final labels had no right_arm frames; forcing first frame to right_arm")
        phases[0] = PHASE_RIGHT
        right_count = int((phases == PHASE_RIGHT).sum())
        left_count = int((phases == PHASE_LEFT).sum())
    if left_count == 0:
        warnings.append("final labels contain no left_arm frames")

    left_active = left_score >= 1.0
    right_active = right_score >= 1.0
    neither = ~left_active & ~right_active
    both = left_active & right_active
    only_left = left_active & ~right_active
    only_right = right_active & ~left_active

    report = {
        "episode_index": int(episode_index),
        "length": length,
        "boundary_frame": int(boundary_frame),
        "boundary_ratio": float(boundary_frame / length) if length else None,
        "right_frame_count": right_count,
        "left_frame_count": left_count,
        "idle_frame_count_before_smoothing": int((initial == LABEL_IDLE).sum()),
        "ambiguous_frame_count_before_smoothing": int((initial == LABEL_AMBIGUOUS).sum()),
        "right_active_segments": int(_count_arm_segments(preliminary, PHASE_RIGHT)),
        "left_active_segments": int(_count_arm_segments(preliminary, PHASE_LEFT)),
        "both_active_ratio": float(both.mean()) if length else 0.0,
        "only_right_ratio": float(only_right.mean()) if length else 0.0,
        "only_left_ratio": float(only_left.mean()) if length else 0.0,
        "neither_ratio": float(neither.mean()) if length else 0.0,
        "mean_left_score": float(left_score.mean()) if length else 0.0,
        "mean_right_score": float(right_score.mean()) if length else 0.0,
        "max_left_score": float(left_score.max()) if length else 0.0,
        "max_right_score": float(right_score.max()) if length else 0.0,
        "first_right_close_frame": _first_gripper_close_frame(actions, args.right_gripper_dim),
        "first_left_close_frame": _first_gripper_close_frame(actions, args.left_gripper_dim),
        "warnings": warnings,
    }

    return EpisodeAnnotation(
        episode_index=int(episode_index),
        phases=phases.astype(np.int64),
        left_score=left_score,
        right_score=right_score,
        smoothed_dominance=smoothed_dominance,
        initial_labels=initial,
        preliminary_labels=preliminary,
        report=report,
        warnings=warnings,
    )


def _to_numpy(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
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


def _normal_feature_value(value: Any, feature: dict[str, Any]) -> Any:
    if feature["dtype"] == "string":
        return str(value)
    if feature["dtype"] in ("image", "video"):
        return _restore_image_layout(value, feature)

    dtype = np.dtype(feature["dtype"])
    expected_shape = tuple(feature["shape"])
    array = np.asarray(_to_numpy(value), dtype=dtype)
    if expected_shape == (1,) and array.shape == ():
        array = array.reshape(1)
    return array


def _phase_one_hot(phase: int) -> np.ndarray:
    if int(phase) == PHASE_RIGHT:
        return np.array([1.0, 0.0], dtype=np.float32)
    return np.array([0.0, 1.0], dtype=np.float32)


def _clone_features_with_phase(source: LeRobotDataset) -> tuple[dict[str, dict], int, int]:
    features = {
        key: copy.deepcopy(value)
        for key, value in source.features.items()
        if key not in DEFAULT_FEATURES
    }

    state_key = "observation.state"
    if state_key in features:
        state_feature = features[state_key]
        if state_feature["dtype"] != "float32" or len(tuple(state_feature["shape"])) != 1:
            raise ValueError(
                "This script appends phase to a 1-D float32 observation.state. "
                f"Found {state_feature!r}"
            )
        old_dim = int(tuple(state_feature["shape"])[0])
        names = state_feature.get("names")
        if names is None:
            names = [f"state_{idx}" for idx in range(old_dim)]
        else:
            names = [str(name) for name in names]
        if len(names) != old_dim:
            raise ValueError(
                "observation.state names length does not match shape: "
                f"len(names)={len(names)} shape={state_feature['shape']}"
            )
        state_feature["shape"] = (old_dim + 2,)
        state_feature["names"] = names + list(PHASE_NAMES)
        new_dim = old_dim + 2
    else:
        old_dim = 0
        features[state_key] = {
            "dtype": "float32",
            "shape": (2,),
            "names": list(PHASE_NAMES),
        }
        new_dim = 2

    return features, old_dim, new_dim


def _frame_from_source_item(
    source: LeRobotDataset,
    item: dict[str, Any],
    phase: int,
    output_features: dict[str, dict],
) -> dict[str, Any]:
    frame: dict[str, Any] = {}
    phase_values = _phase_one_hot(phase)

    for key, feature in source.features.items():
        if key in DEFAULT_FEATURES:
            continue
        if key == "observation.state":
            state = np.asarray(_to_numpy(item[key]), dtype=np.float32)
            if state.ndim == 0:
                state = state.reshape(1)
            frame[key] = np.concatenate([state, phase_values], axis=0).astype(np.float32)
        elif feature["dtype"] in ("image", "video"):
            frame[key] = _restore_image_layout(item[key], feature)
        else:
            frame[key] = _normal_feature_value(item[key], feature)

    if "observation.state" not in source.features:
        frame["observation.state"] = phase_values

    # Validate against the output feature dtype/shape before LeRobot's add_frame
    # does the same; this keeps phase appending errors local to this script.
    expected = tuple(output_features["observation.state"]["shape"])
    if tuple(frame["observation.state"].shape) != expected:
        raise ValueError(
            "Unexpected observation.state shape after phase append: "
            f"{frame['observation.state'].shape} != {expected}"
        )

    frame["task"] = item["task"]
    return frame


def _episode_actions(dataset: LeRobotDataset, start: int, end: int) -> np.ndarray:
    raw_dataset = dataset.hf_dataset.with_format(None)
    batch = raw_dataset[start:end]
    if "action" not in batch:
        raise KeyError("Input dataset has no 'action' column.")
    actions = np.asarray(batch["action"], dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected action array [frames, dim], got shape {actions.shape}")
    return actions


def _create_output_dataset(
    source: LeRobotDataset,
    *,
    output_root: Path,
    output_repo_id: str,
    output_features: dict[str, dict],
    overwrite: bool,
    fps: int,
) -> LeRobotDataset:
    input_root = _resolve_for_safety(Path(source.root))
    output_root_resolved = _resolve_for_safety(output_root)
    if _paths_overlap(input_root, output_root_resolved):
        raise ValueError(
            "Refusing to write output inside or over the input dataset. "
            f"input_root={input_root} output_root={output_root_resolved}"
        )

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    return LeRobotDataset.create(
        repo_id=output_repo_id,
        root=output_root,
        fps=int(fps),
        features=output_features,
        robot_type=source.meta.info.get("robot_type"),
        use_videos=len(source.meta.video_keys) > 0,
        image_writer_threads=4,
        batch_encoding_size=1,
    )


def _phase_lookup_by_dataset_index(source: LeRobotDataset, annotations: dict[int, EpisodeAnnotation]) -> np.ndarray:
    phase_by_index = np.empty(int(source.meta.total_frames), dtype=np.int64)
    for ep in source.meta.episodes:
        ep_idx = int(ep["episode_index"])
        start = int(ep["dataset_from_index"])
        end = int(ep["dataset_to_index"])
        phases = annotations[ep_idx].phases
        if len(phases) != end - start:
            raise ValueError(
                f"Episode {ep_idx} phase length mismatch: {len(phases)} != {end - start}"
            )
        phase_by_index[start:end] = phases
    return phase_by_index


def _append_phase_to_copied_data_files(
    source: LeRobotDataset,
    output_root: Path,
    annotations: dict[int, EpisodeAnnotation],
    output_features: dict[str, dict],
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    import pandas as pd

    state_key = "observation.state"
    expected_state_dim = int(tuple(output_features[state_key]["shape"])[0])
    phase_by_index = _phase_lookup_by_dataset_index(source, annotations)
    episode_state_chunks: dict[int, list[np.ndarray]] = {}
    all_state_chunks: list[np.ndarray] = []

    data_paths = sorted((output_root / "data").glob("chunk-*/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No data parquet files found under {output_root / 'data'}")

    for data_path in data_paths:
        df = pd.read_parquet(data_path)
        row_indices = df["index"].to_numpy(dtype=np.int64)
        phase_values = np.stack([_phase_one_hot(int(phase_by_index[idx])) for idx in row_indices])

        if state_key in df:
            states = np.stack(df[state_key].to_numpy()).astype(np.float32)
            new_states = np.concatenate([states, phase_values], axis=1).astype(np.float32)
        else:
            new_states = phase_values.astype(np.float32)

        if new_states.shape[1] != expected_state_dim:
            raise ValueError(
                f"{data_path} produced {state_key} dim {new_states.shape[1]}, "
                f"expected {expected_state_dim}"
            )

        df[state_key] = [row.tolist() for row in new_states]
        df.to_parquet(data_path, index=False)

        episode_indices = df["episode_index"].to_numpy(dtype=np.int64)
        for ep_idx in np.unique(episode_indices):
            mask = episode_indices == ep_idx
            episode_state_chunks.setdefault(int(ep_idx), []).append(new_states[mask])
        all_state_chunks.append(new_states)

    episode_state_stats = {
        ep_idx: get_feature_stats(np.concatenate(chunks, axis=0), axis=0, keepdims=False)
        for ep_idx, chunks in episode_state_chunks.items()
    }
    global_state_stats = get_feature_stats(np.concatenate(all_state_chunks, axis=0), axis=0, keepdims=False)
    return episode_state_stats, global_state_stats


def _rewrite_info_with_phase_feature(output_root: Path, output_features: dict[str, dict]) -> None:
    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info.setdefault("features", {})["observation.state"] = copy.deepcopy(
        output_features["observation.state"]
    )
    write_info(info, output_root)


def _rewrite_stats_with_phase_state(output_root: Path, global_state_stats: dict[str, np.ndarray]) -> None:
    stats = load_stats(output_root)
    if stats is None:
        raise ValueError(f"Copied dataset is missing stats.json: {output_root}")
    stats["observation.state"] = global_state_stats
    write_stats(stats, output_root)


def _rewrite_episode_state_stats(
    output_root: Path,
    episode_state_stats: dict[int, dict[str, np.ndarray]],
) -> None:
    import pandas as pd

    episode_paths = sorted((output_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(
            f"No episode metadata parquet files found under {output_root / 'meta' / 'episodes'}"
        )

    for episode_path in episode_paths:
        df = pd.read_parquet(episode_path)
        episode_indices = df["episode_index"].to_numpy(dtype=np.int64)
        for stat_name in next(iter(episode_state_stats.values())).keys():
            col = f"stats/observation.state/{stat_name}"
            df[col] = [
                np.asarray(episode_state_stats[int(ep_idx)][stat_name]).tolist()
                for ep_idx in episode_indices
            ]
        df.to_parquet(episode_path, index=False)


def _copy_source_dataset_with_phase(
    source: LeRobotDataset,
    *,
    input_root: Path,
    output_root: Path,
    output_features: dict[str, dict],
    annotations: dict[int, EpisodeAnnotation],
    overwrite: bool,
) -> None:
    """Create a phase dataset without decoding or re-encoding videos.

    Phase annotation only changes tabular state data. Copying source videos
    byte-for-byte avoids introducing codec artifacts or changing video shard
    boundaries while preserving the original camera streams.
    """

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    shutil.copytree(input_root, output_root)
    episode_state_stats, global_state_stats = _append_phase_to_copied_data_files(
        source,
        output_root,
        annotations,
        output_features,
    )
    _rewrite_info_with_phase_feature(output_root, output_features)
    _rewrite_stats_with_phase_state(output_root, global_state_stats)
    _rewrite_episode_state_stats(output_root, episode_state_stats)


def _plot_episode(
    path: Path,
    annotation: EpisodeAnnotation,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is not available; skipping phase plots") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(annotation.phases))

    fig, ax = plt.subplots(figsize=(12, 4))
    for segment in _contiguous_segments(annotation.phases):
        color = "#d9ecff" if segment.label == PHASE_RIGHT else "#ffe2d8"
        ax.axvspan(segment.start, segment.end, color=color, alpha=0.55, linewidth=0)

    ax.plot(x, annotation.left_score, label="left_score", color="#d95f02", linewidth=1.2)
    ax.plot(x, annotation.right_score, label="right_score", color="#1b9e77", linewidth=1.2)
    ax.plot(
        x,
        annotation.smoothed_dominance,
        label="smoothed right-left dominance",
        color="#4b4b4b",
        linewidth=1.0,
        alpha=0.9,
    )
    boundary = annotation.report.get("boundary_frame")
    if boundary is not None:
        ax.axvline(int(boundary), color="#000000", linestyle="--", linewidth=1.0, label="boundary")
    ax.axhline(0.0, color="#777777", linewidth=0.6, alpha=0.5)
    ax.axhline(1.0, color="#999999", linewidth=0.6, alpha=0.4)
    ax.set_title(f"Episode {annotation.episode_index} phase annotation")
    ax.set_xlabel("frame")
    ax.set_ylabel("score")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_reports(
    report_dir: Path,
    report: dict[str, Any],
    *,
    save_report: bool,
) -> None:
    if not save_report:
        return
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "phase_annotation_report.json"
    json_path.write_text(json.dumps(report, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = report_dir / "phase_annotation_summary.csv"
    rows = report.get("episodes", [])
    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames and key != "warnings":
                    fieldnames.append(key)
        fieldnames.append("warnings")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                flat = dict(row)
                flat["warnings"] = "; ".join(row.get("warnings", []))
                writer.writerow(flat)
    else:
        csv_path.write_text("", encoding="utf-8")


def _validate_output_dataset(
    *,
    output_root: Path,
    output_repo_id: str,
    old_state_dim: int,
    expected_total_frames: int,
) -> list[str]:
    warnings: list[str] = []
    dataset = LeRobotDataset(output_repo_id, root=output_root)

    state_feature = dataset.features.get("observation.state")
    if state_feature is None:
        raise ValueError("Output dataset is missing observation.state")
    new_state_dim = int(tuple(state_feature["shape"])[0])
    if new_state_dim != old_state_dim + 2:
        raise ValueError(
            f"Output observation.state shape mismatch: {new_state_dim} != {old_state_dim + 2}"
        )
    names = state_feature.get("names") or []
    if list(names[-2:]) != list(PHASE_NAMES):
        raise ValueError(f"Output observation.state names do not end with {PHASE_NAMES}: {names[-2:]}")
    if int(dataset.meta.total_frames) != int(expected_total_frames):
        raise ValueError(
            f"Output total_frames mismatch: {dataset.meta.total_frames} != {expected_total_frames}"
        )
    if "observation.state" not in (dataset.meta.stats or {}):
        raise ValueError("Output stats.json is missing observation.state stats")

    raw = dataset.hf_dataset.with_format(None)
    total_left = 0
    for ep in dataset.meta.episodes:
        start = int(ep["dataset_from_index"])
        end = int(ep["dataset_to_index"])
        states = np.asarray(raw[start:end]["observation.state"], dtype=np.float32)
        phase = states[:, -2:]
        sums = phase.sum(axis=1)
        if not np.allclose(sums, 1.0):
            raise ValueError(f"Phase one-hot sum validation failed in episode {int(ep['episode_index'])}")
        if not np.all((phase == 0.0) | (phase == 1.0)):
            raise ValueError(f"Phase one-hot binary validation failed in episode {int(ep['episode_index'])}")
        right_frames = int((phase[:, 0] == 1.0).sum())
        left_frames = int((phase[:, 1] == 1.0).sum())
        total_left += left_frames
        if right_frames == 0:
            raise ValueError(f"Episode {int(ep['episode_index'])} has no right_arm phase frames")
    if total_left == 0:
        warnings.append("output dataset contains no left_arm phase frames")
    return warnings


def _report_totals(episode_reports: list[dict[str, Any]]) -> dict[str, Any]:
    total_frames = sum(int(row["length"]) for row in episode_reports)
    total_right = sum(int(row["right_frame_count"]) for row in episode_reports)
    total_left = sum(int(row["left_frame_count"]) for row in episode_reports)
    return {
        "episodes": len(episode_reports),
        "frames": total_frames,
        "right_frame_count": total_right,
        "left_frame_count": total_left,
        "right_frame_ratio": float(total_right / total_frames) if total_frames else 0.0,
        "left_frame_ratio": float(total_left / total_frames) if total_frames else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate a dual-arm LeRobot dataset with phase_right_arm/phase_left_arm "
            "features appended to observation.state."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Existing LeRobot dataset root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="New output dataset root. Required unless --dry-run and --report-dir are used.",
    )
    parser.add_argument("--repo-id", default=None, help="Repo id used for loading/writing local metadata.")
    parser.add_argument(
        "--phase-mode",
        default="two_phase_active_arm",
        choices=["two_phase_active_arm"],
        help="Phase annotation mode. Only two_phase_active_arm is implemented.",
    )
    parser.add_argument("--left-delta-slice", default="0:6")
    parser.add_argument("--right-delta-slice", default="6:12")
    parser.add_argument("--left-gripper-dim", type=int, default=None)
    parser.add_argument("--right-gripper-dim", type=int, default=None)
    parser.add_argument("--trans-threshold", type=float, default=0.001)
    parser.add_argument("--rot-threshold", type=float, default=0.005)
    parser.add_argument("--dominance-ratio", type=float, default=1.2)
    parser.add_argument("--fps", type=float, default=None, help="Override dataset fps. Defaults to metadata fps or 30.")
    parser.add_argument("--min-active-sec", type=float, default=0.3)
    parser.add_argument("--max-idle-gap-sec", type=float, default=0.5)
    parser.add_argument("--smooth-window", type=int, default=31, help="Odd moving-average window.")

    parser.set_defaults(use_main_boundary=True)
    parser.add_argument("--use-main-boundary", dest="use_main_boundary", action="store_true")
    parser.add_argument("--no-use-main-boundary", dest="use_main_boundary", action="store_false")

    parser.set_defaults(save_report=True)
    parser.add_argument("--save-report", dest="save_report", action="store_true")
    parser.add_argument("--no-save-report", dest="save_report", action="store_false")

    parser.set_defaults(save_plots=True)
    parser.add_argument("--save-plots", dest="save_plots", action="store_true")
    parser.add_argument("--no-save-plots", dest="save_plots", action="store_false")

    parser.add_argument("--overwrite", action="store_true", help="Replace an existing distinct output-root.")
    parser.add_argument("--dry-run", action="store_true", help="Only compute report/plots; do not write dataset.")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Report output directory. In export mode defaults to output-root.",
    )
    return parser.parse_args()


def _args_for_report(args: argparse.Namespace) -> dict[str, Any]:
    out = vars(args).copy()
    for key, value in list(out.items()):
        if isinstance(value, Path):
            out[key] = str(value)
    return out


def _phase_report_dir(args: argparse.Namespace) -> Path:
    if args.report_dir is not None:
        return args.report_dir.expanduser()
    if args.output_root is not None:
        return args.output_root.expanduser()
    return Path("phase_annotation_report").resolve()


def main() -> None:
    args = parse_args()
    _odd_window(int(args.smooth_window))

    input_root = args.input_root.expanduser()
    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset root does not exist: {input_root}")
    if not args.dry_run and args.output_root is None:
        raise ValueError("--output-root is required unless --dry-run is set")

    report_dir = _phase_report_dir(args)
    if _path_inside(report_dir, input_root):
        raise ValueError(
            "Refusing to write reports inside the input dataset. "
            f"input_root={_resolve_for_safety(input_root)} report_dir={_resolve_for_safety(report_dir)}"
        )

    _require_lerobot()

    source_repo_id = _infer_repo_id(input_root, args.repo_id)
    source = LeRobotDataset(source_repo_id, root=input_root)
    fps = float(args.fps if args.fps is not None else getattr(source, "fps", 30) or 30)
    output_repo_id = _output_repo_id(source.repo_id, args.repo_id)
    output_root = args.output_root.expanduser() if args.output_root is not None else None

    if not args.dry_run and output_root is not None and _paths_overlap(input_root, output_root):
        raise ValueError(
            "Refusing to write output inside or over the input dataset. "
            f"input_root={_resolve_for_safety(input_root)} output_root={_resolve_for_safety(output_root)}"
        )

    if "action" not in source.features:
        raise KeyError("Input dataset has no action feature.")
    action_dim = int(tuple(source.features["action"]["shape"])[0])
    left_indices = _parse_slice(args.left_delta_slice, action_dim, label="--left-delta-slice")
    right_indices = _parse_slice(args.right_delta_slice, action_dim, label="--right-delta-slice")
    output_features, old_state_dim, new_state_dim = _clone_features_with_phase(source)

    logger.info(
        "[LOAD] repo_id=%s root=%s episodes=%d frames=%d fps=%s action_dim=%d state_dim=%d->%d",
        source.repo_id,
        input_root,
        source.meta.total_episodes,
        source.meta.total_frames,
        fps,
        action_dim,
        old_state_dim,
        new_state_dim,
    )

    annotations: dict[int, EpisodeAnnotation] = {}
    episode_reports: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    total_frames = 0

    for ep in source.meta.episodes:
        ep_idx = int(ep["episode_index"])
        start = int(ep["dataset_from_index"])
        end = int(ep["dataset_to_index"])
        actions = _episode_actions(source, start, end)
        annotation = annotate_episode(
            episode_index=ep_idx,
            actions=actions,
            left_indices=left_indices,
            right_indices=right_indices,
            args=args,
            fps=fps,
        )
        annotations[ep_idx] = annotation
        episode_reports.append(annotation.report)
        total_frames += int(annotation.report["length"])

        for warning in annotation.warnings:
            all_warnings.append(f"episode {ep_idx}: {warning}")
        logger.info(
            "[EP %s] len=%d boundary=%s right=%d left=%d warnings=%d",
            ep_idx,
            annotation.report["length"],
            annotation.report["boundary_frame"],
            annotation.report["right_frame_count"],
            annotation.report["left_frame_count"],
            len(annotation.warnings),
        )

        if args.save_plots:
            try:
                _plot_episode(report_dir / "phase_plots" / f"episode_{ep_idx:06d}.png", annotation)
            except ModuleNotFoundError as exc:
                warning = str(exc)
                if warning not in all_warnings:
                    all_warnings.append(warning)
                logger.warning("[PLOT] %s", warning)
                args.save_plots = False

    validation_warnings: list[str] = []
    if not args.dry_run:
        if output_root is None:
            raise ValueError("--output-root is required when not using --dry-run")
        _copy_source_dataset_with_phase(
            source,
            input_root=input_root,
            output_root=output_root,
            output_features=output_features,
            annotations=annotations,
            overwrite=bool(args.overwrite),
        )
        logger.info("[WRITE] dataset copied with phase feature; videos preserved byte-for-byte")

        validation_warnings = _validate_output_dataset(
            output_root=output_root,
            output_repo_id=output_repo_id,
            old_state_dim=old_state_dim,
            expected_total_frames=total_frames,
        )
        all_warnings.extend(validation_warnings)

    report = {
        "phase_mode": args.phase_mode,
        "dry_run": bool(args.dry_run),
        "input_root": str(input_root),
        "output_root": str(output_root) if output_root is not None and not args.dry_run else None,
        "repo_id": output_repo_id,
        "fps": fps,
        "args": _args_for_report(args),
        "feature_update": {
            "observation_state_original_dim": old_state_dim,
            "observation_state_output_dim": new_state_dim,
            "appended_names": list(PHASE_NAMES),
            "stats_recomputed": not args.dry_run,
            "schema_updated": not args.dry_run,
        },
        "totals": _report_totals(episode_reports),
        "episodes": episode_reports,
        "warnings": all_warnings,
        "validation_warnings": validation_warnings,
    }
    _write_reports(report_dir, report, save_report=bool(args.save_report))

    logger.info(
        "[DONE] episodes=%d frames=%d%s report_dir=%s warnings=%d",
        len(episode_reports),
        total_frames,
        " dry-run" if args.dry_run else f" output_root={output_root}",
        report_dir,
        len(all_warnings),
    )


if __name__ == "__main__":
    main()
