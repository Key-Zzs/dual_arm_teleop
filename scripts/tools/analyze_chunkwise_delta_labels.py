#!/usr/bin/env python

"""Offline checks for ACT chunk-wise delta labels.

This script does not run a policy model and does not touch robot execution. It reads a LeRobot dataset,
verifies the chunk-wise training transform against inference decode on one sample, and summarizes the
right-arm z label distribution that the model sees.

python dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/tools/analyze_chunkwise_delta_labels.py \
  --repo-id nero_task3_step1/2mL_empty_right_merged_20260504_E175 \
  --root /home/geist/.cache/huggingface/lerobot/nero_task3_step1/2mL_empty_right_merged_20260504_E175 \
  --chunk-size 30 \
  --sample-index 0 \
  --max-samples 5000 \
  --observation-state-pose-axis-order x,y,z,rz,ry,rx
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

ACTION = "action"
OBS_STATE = "observation.state"


def main() -> None:
    args = _parse_args()
    _import_runtime_dependencies()
    axis_order = _parse_axis_order(args.observation_state_pose_axis_order)

    dataset = _load_dataset(args)
    action_feature_names = _resolve_feature_names(dataset, ACTION, args.action_feature_names)
    observation_state_feature_names = _resolve_feature_names(
        dataset, OBS_STATE, args.observation_state_feature_names
    )

    mapping = inspect_chunkwise_feature_mapping(
        action_feature_names,
        observation_state_feature_names,
        axis_order,
        require_right_arm=True,
    )
    right_z_action_index = mapping["arms"]["right"]["action"]["z"]["index"]
    right_z_state_index = mapping["arms"]["right"]["observation_state"]["z"]["index"]

    _print_header("Feature Mapping")
    print(f"action right z: index={right_z_action_index} name={action_feature_names[right_z_action_index]}")
    state_right_z = mapping["arms"]["right"]["observation_state"]["z"]
    print(
        "observation.state right z: "
        f"index={right_z_state_index} name={state_right_z['feature_name']} "
        f"effective={state_right_z.get('effective_feature_name', state_right_z['feature_name'])}"
    )
    print(f"observation_state_pose_axis_order={axis_order}")

    sample = _read_sample(dataset, args.sample_index)
    _run_inverse_checks(
        sample=sample,
        action_feature_names=action_feature_names,
        observation_state_feature_names=observation_state_feature_names,
        observation_state_pose_axis_order=axis_order,
        right_z_action_index=right_z_action_index,
        right_z_state_index=right_z_state_index,
        action_stats=dataset.meta.stats.get(ACTION, {}) if dataset.meta.stats is not None else {},
    )

    _run_distribution_stats(
        dataset=dataset,
        action_feature_names=action_feature_names,
        right_z_action_index=right_z_action_index,
        max_samples=args.max_samples,
        sample_stride=args.sample_stride,
        progress_every=args.progress_every,
    )

    _print_time_semantics(dataset, args.chunk_size)


def _import_runtime_dependencies() -> None:
    global ACTION
    global OBS_STATE
    global LeRobotDataset
    global LeRobotDatasetMetadata
    global check_chunkwise_train_decode_inverse
    global convert_stepwise_to_chunkwise_actions
    global decode_chunkwise_actions_to_absolute_actions
    global inspect_chunkwise_feature_mapping
    global _integrate_stepwise_actions_to_absolute_actions
    global _wrap_angle_delta
    global np
    global torch

    import numpy as _np
    import torch as _torch

    from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LeRobotDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata as _LeRobotDatasetMetadata
    from lerobot.utils.constants import ACTION as _ACTION
    from lerobot.utils.constants import OBS_STATE as _OBS_STATE

    action_delta_utils = _load_act_module_from_path(
        "lerobot.policies.act.action_delta_utils",
        SRC_ROOT / "lerobot" / "policies" / "act" / "action_delta_utils.py",
    )
    action_inference_utils = _load_act_module_from_path(
        "lerobot.policies.act.action_inference_utils",
        SRC_ROOT / "lerobot" / "policies" / "act" / "action_inference_utils.py",
    )

    np = _np
    torch = _torch
    LeRobotDataset = _LeRobotDataset
    LeRobotDatasetMetadata = _LeRobotDatasetMetadata
    convert_stepwise_to_chunkwise_actions = action_delta_utils.convert_stepwise_to_chunkwise_actions
    decode_chunkwise_actions_to_absolute_actions = (
        action_inference_utils.decode_chunkwise_actions_to_absolute_actions
    )
    inspect_chunkwise_feature_mapping = action_inference_utils.inspect_chunkwise_feature_mapping
    check_chunkwise_train_decode_inverse = action_inference_utils.check_chunkwise_train_decode_inverse
    _integrate_stepwise_actions_to_absolute_actions = (
        action_inference_utils._integrate_stepwise_actions_to_absolute_actions
    )
    _wrap_angle_delta = action_inference_utils._wrap_angle_delta
    ACTION = _ACTION
    OBS_STATE = _OBS_STATE


def _load_act_module_from_path(module_name: str, module_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]

    _ensure_package("lerobot.policies", SRC_ROOT / "lerobot" / "policies")
    _ensure_package("lerobot.policies.act", SRC_ROOT / "lerobot" / "policies" / "act")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_name} from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_package(package_name: str, package_path: Path) -> None:
    if package_name in sys.modules:
        return
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules[package_name] = package


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo id, for example user/dataset.")
    parser.add_argument("--root", type=Path, default=None, help="Optional local LeRobot dataset root.")
    parser.add_argument("--chunk-size", type=int, default=30, help="ACT action chunk size to analyze.")
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset frame index for the inverse check.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum frames to scan for distribution stats. Defaults to the whole dataset.",
    )
    parser.add_argument("--sample-stride", type=int, default=1, help="Scan every Nth dataset frame.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print distribution-scan progress every N scanned frames. Set <=0 to disable.",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Optional comma-separated episode indices, for example '0,1,2'.",
    )
    parser.add_argument(
        "--observation-state-pose-axis-order",
        default="x,y,z,rz,ry,rx",
        help=(
            "Stored ee-pose axis order inside observation.state. Nero legacy datasets usually use "
            "'x,y,z,rz,ry,rx'."
        ),
    )
    parser.add_argument(
        "--action-feature-names",
        default=None,
        help="Optional comma-separated action feature names if dataset metadata has no action names.",
    )
    parser.add_argument(
        "--observation-state-feature-names",
        default=None,
        help="Optional comma-separated observation.state feature names if dataset metadata has no state names.",
    )
    return parser.parse_args()


def _load_dataset(args: argparse.Namespace) -> LeRobotDataset:
    episodes = _parse_optional_int_list(args.episodes)
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    fps = metadata.fps
    delta_timestamps = {ACTION: [i / fps for i in range(args.chunk_size)]}
    return LeRobotDataset(
        args.repo_id,
        root=args.root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        download_videos=False,
    )


def _parse_optional_int_list(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_axis_order(value: str) -> tuple[str, ...]:
    axes = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(axes) != 6:
        raise ValueError(f"Expected 6 pose axes, got {axes}.")
    return axes


def _resolve_feature_names(dataset: LeRobotDataset, key: str, override: str | None) -> tuple[str, ...]:
    if override:
        return tuple(item.strip() for item in override.split(",") if item.strip())

    feature_spec = dataset.meta.features.get(key)
    if feature_spec is None:
        raise ValueError(f"Dataset metadata does not contain feature '{key}'.")

    names = tuple(_flatten_names(feature_spec.get("names")))
    if not names:
        raise ValueError(
            f"Dataset metadata feature '{key}' has no flattened names. Pass --{key.replace('.', '-')}-feature-names."
        )
    return names


def _flatten_names(names: Any) -> list[str]:
    if names is None:
        return []
    if isinstance(names, str):
        return [names]
    flattened: list[str] = []
    if isinstance(names, dict):
        iterable = names.values()
    else:
        iterable = names
    for item in iterable:
        flattened.extend(_flatten_names(item))
    return flattened


def _read_sample(dataset: LeRobotDataset, index: int) -> dict[str, torch.Tensor]:
    if index < 0 or index >= len(dataset):
        raise IndexError(f"sample index {index} is outside dataset length {len(dataset)}.")

    item = dataset.hf_dataset[index]
    ep_idx = _to_int(item["episode_index"])
    query_indices, padding = dataset._get_query_indices(index, ep_idx)
    action_chunk = dataset._query_hf_dataset({ACTION: query_indices[ACTION]})[ACTION].to(dtype=torch.float64)
    chunk_ref_state = _to_tensor(item[OBS_STATE]).to(dtype=torch.float64)
    action_is_pad = padding[f"{ACTION}_is_pad"].to(dtype=torch.bool)

    return {
        "action": action_chunk,
        "observation.state": chunk_ref_state,
        "action_is_pad": action_is_pad,
    }


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return torch.as_tensor(value)


def _to_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    if isinstance(value, np.ndarray):
        return int(value.item())
    return int(value)


def _run_inverse_checks(
    *,
    sample: dict[str, torch.Tensor],
    action_feature_names: tuple[str, ...],
    observation_state_feature_names: tuple[str, ...],
    observation_state_pose_axis_order: tuple[str, ...],
    right_z_action_index: int,
    right_z_state_index: int,
    action_stats: dict[str, Any],
) -> None:
    stepwise_actions = sample["action"].unsqueeze(0)
    chunk_ref_state = sample["observation.state"].unsqueeze(0)

    result = check_chunkwise_train_decode_inverse(
        stepwise_actions,
        chunk_ref_state,
        action_feature_names,
        observation_state_feature_names,
        observation_state_pose_axis_order,
    )

    chunkwise_delta = convert_stepwise_to_chunkwise_actions(stepwise_actions, action_feature_names)
    decoded_absolute = decode_chunkwise_actions_to_absolute_actions(
        chunkwise_delta,
        chunk_ref_state=chunk_ref_state,
        action_feature_names=action_feature_names,
        observation_state_feature_names=observation_state_feature_names,
        observation_state_pose_axis_order=observation_state_pose_axis_order,
    )
    reference_absolute = _integrate_stepwise_actions_to_absolute_actions(
        stepwise_actions,
        chunk_ref_state=chunk_ref_state,
        action_feature_names=action_feature_names,
        observation_state_feature_names=observation_state_feature_names,
        observation_state_pose_axis_order=observation_state_pose_axis_order,
    )
    error = _pose_error(decoded_absolute, reference_absolute, action_feature_names)

    _print_header("Raw-Space Inverse Check")
    print(f"ok={result['ok']} max_abs_error={result['max_abs_error']:.12g}")
    print(f"xyz_max_abs_error={_xyz_max_abs_error(error, action_feature_names):.12g}")
    print("right z per step:")
    _print_right_z_table(
        stepwise=stepwise_actions[0, :, right_z_action_index],
        chunkwise=chunkwise_delta[0, :, right_z_action_index],
        decoded=decoded_absolute[0, :, right_z_action_index],
        reference=reference_absolute[0, :, right_z_action_index],
        error=error[0, :, right_z_action_index],
    )
    if not result["ok"]:
        first_bad = _first_bad_step(error[0], atol=1e-6)
        print(f"FAILED: first_bad_step={first_bad}; step0_mismatch={first_bad == 0}")
    else:
        print("PASS: training transform and inference decode are inverse in raw physical units.")

    if "mean" in action_stats and "std" in action_stats:
        _run_training_order_normalization_check(
            stepwise_actions=stepwise_actions,
            chunk_ref_state=chunk_ref_state,
            action_feature_names=action_feature_names,
            observation_state_feature_names=observation_state_feature_names,
            observation_state_pose_axis_order=observation_state_pose_axis_order,
            right_z_action_index=right_z_action_index,
            right_z_state_index=right_z_state_index,
            action_stats=action_stats,
            raw_reference_absolute=reference_absolute,
        )
    else:
        _print_header("Training-Order Normalization Check")
        print("Skipped: dataset action mean/std stats were not found.")


def _run_training_order_normalization_check(
    *,
    stepwise_actions: torch.Tensor,
    chunk_ref_state: torch.Tensor,
    action_feature_names: tuple[str, ...],
    observation_state_feature_names: tuple[str, ...],
    observation_state_pose_axis_order: tuple[str, ...],
    right_z_action_index: int,
    right_z_state_index: int,
    action_stats: dict[str, Any],
    raw_reference_absolute: torch.Tensor,
) -> None:
    mean = _stats_tensor(action_stats["mean"], stepwise_actions)
    std = _stats_tensor(action_stats["std"], stepwise_actions).clamp_min(1e-12)

    normalized_stepwise = (stepwise_actions - mean) / std
    converted_normalized = convert_stepwise_to_chunkwise_actions(normalized_stepwise, action_feature_names)
    postprocessed_converted = converted_normalized * std + mean
    decoded_train_order = decode_chunkwise_actions_to_absolute_actions(
        postprocessed_converted,
        chunk_ref_state=chunk_ref_state,
        action_feature_names=action_feature_names,
        observation_state_feature_names=observation_state_feature_names,
        observation_state_pose_axis_order=observation_state_pose_axis_order,
    )
    error = _pose_error(decoded_train_order, raw_reference_absolute, action_feature_names)

    _print_header("Pre-Fix Normalization-Order Check")
    print(
        "This simulates the old buggy order: preprocessor normalizes action, "
        "ACT forward converts labels, postprocessor unnormalizes prediction."
    )
    print(f"right_z_action_stats.mean={float(mean.flatten()[right_z_action_index]):.12g}")
    print(f"right_z_action_stats.std={float(std.flatten()[right_z_action_index]):.12g}")
    print(f"right_z_chunk_ref={float(chunk_ref_state[0, right_z_state_index]):.12g}")
    print(f"max_abs_error_vs_raw_semantics={float(torch.max(torch.abs(error)).item()):.12g}")
    print(f"xyz_max_abs_error_vs_raw_semantics={_xyz_max_abs_error(error, action_feature_names):.12g}")
    print("right z per step after current train/postprocess order:")
    _print_right_z_table(
        stepwise=stepwise_actions[0, :, right_z_action_index],
        chunkwise=postprocessed_converted[0, :, right_z_action_index],
        decoded=decoded_train_order[0, :, right_z_action_index],
        reference=raw_reference_absolute[0, :, right_z_action_index],
        error=error[0, :, right_z_action_index],
    )

    raw_chunkwise = convert_stepwise_to_chunkwise_actions(stepwise_actions, action_feature_names)
    normalized_fixed = (raw_chunkwise - mean) / std
    postprocessed_fixed = normalized_fixed * std + mean
    decoded_fixed = decode_chunkwise_actions_to_absolute_actions(
        postprocessed_fixed,
        chunk_ref_state=chunk_ref_state,
        action_feature_names=action_feature_names,
        observation_state_feature_names=observation_state_feature_names,
        observation_state_pose_axis_order=observation_state_pose_axis_order,
    )
    fixed_error = _pose_error(decoded_fixed, raw_reference_absolute, action_feature_names)
    _print_header("Fixed Training-Order Check")
    print(
        "This simulates the fixed order: raw step-wise action -> chunk-wise conversion -> normalization -> "
        "postprocess -> decode."
    )
    print(f"max_abs_error_vs_raw_semantics={float(torch.max(torch.abs(fixed_error)).item()):.12g}")
    print(f"xyz_max_abs_error_vs_raw_semantics={_xyz_max_abs_error(fixed_error, action_feature_names):.12g}")
    print("right z per step after fixed train/postprocess order:")
    _print_right_z_table(
        stepwise=stepwise_actions[0, :, right_z_action_index],
        chunkwise=postprocessed_fixed[0, :, right_z_action_index],
        decoded=decoded_fixed[0, :, right_z_action_index],
        reference=raw_reference_absolute[0, :, right_z_action_index],
        error=fixed_error[0, :, right_z_action_index],
    )


def _pose_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
    action_feature_names: tuple[str, ...],
) -> torch.Tensor:
    error = actual - expected
    for offset in range(0, len(action_feature_names), 1):
        name = action_feature_names[offset]
        if name.endswith((".rx", ".ry", ".rz", ".roll", ".pitch", ".yaw")):
            error[:, :, offset] = _wrap_angle_delta(error[:, :, offset])
    return error


def _xyz_max_abs_error(error: torch.Tensor, action_feature_names: tuple[str, ...]) -> float:
    xyz_indices = [index for index, name in enumerate(action_feature_names) if name.endswith((".x", ".y", ".z"))]
    if not xyz_indices:
        return float("nan")
    return float(torch.max(torch.abs(error[:, :, xyz_indices])).item())


def _first_bad_step(error: torch.Tensor, atol: float) -> int | None:
    per_step = torch.max(torch.abs(error), dim=-1).values
    bad_steps = torch.nonzero(per_step > atol).flatten()
    if len(bad_steps) == 0:
        return None
    return int(bad_steps[0].item())


def _stats_tensor(values: Any, like: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(values), dtype=like.dtype, device=like.device).flatten()
    if tensor.numel() != like.shape[-1]:
        raise ValueError(f"stats dimension {tensor.numel()} does not match action dim {like.shape[-1]}.")
    return tensor.view(1, 1, -1)


def _print_right_z_table(
    *,
    stepwise: torch.Tensor,
    chunkwise: torch.Tensor,
    decoded: torch.Tensor,
    reference: torch.Tensor,
    error: torch.Tensor,
) -> None:
    print("  k | stepwise_delta | chunkwise_delta | decoded_abs | reference_abs | error")
    for k in range(stepwise.shape[0]):
        print(
            f"  {k:02d} | {float(stepwise[k]): .9f} | {float(chunkwise[k]): .9f} | "
            f"{float(decoded[k]): .9f} | {float(reference[k]): .9f} | {float(error[k]): .3e}"
        )


def _run_distribution_stats(
    *,
    dataset: LeRobotDataset,
    action_feature_names: tuple[str, ...],
    right_z_action_index: int,
    max_samples: int | None,
    sample_stride: int,
    progress_every: int,
) -> None:
    if sample_stride <= 0:
        raise ValueError("--sample-stride must be positive.")

    raw_step0_right_z: list[float] = []
    raw_chunkwise_right_z: list[list[float]] = [[] for _ in dataset.delta_indices[ACTION]]
    train_order_chunkwise_right_z: list[list[float]] = [[] for _ in dataset.delta_indices[ACTION]]

    action_stats = dataset.meta.stats.get(ACTION, {}) if dataset.meta.stats is not None else {}
    has_mean_std = "mean" in action_stats and "std" in action_stats

    scanned = 0
    for index in range(0, len(dataset), sample_stride):
        if max_samples is not None and scanned >= max_samples:
            break
        if progress_every > 0 and scanned > 0 and scanned % progress_every == 0:
            print(f"[distribution] scanned {scanned} frames...", file=sys.stderr, flush=True)
        sample = _read_sample(dataset, index)
        stepwise_actions = sample["action"].unsqueeze(0)
        valid_mask = ~sample["action_is_pad"]

        chunkwise_delta = convert_stepwise_to_chunkwise_actions(stepwise_actions, action_feature_names)
        if has_mean_std:
            mean = _stats_tensor(action_stats["mean"], stepwise_actions)
            std = _stats_tensor(action_stats["std"], stepwise_actions).clamp_min(1e-12)
            normalized = (stepwise_actions - mean) / std
            converted_normalized = convert_stepwise_to_chunkwise_actions(normalized, action_feature_names)
            train_order_chunkwise = converted_normalized * std + mean
        else:
            train_order_chunkwise = None

        if bool(valid_mask[0]):
            raw_step0_right_z.append(float(stepwise_actions[0, 0, right_z_action_index]))
        for step, is_valid in enumerate(valid_mask.tolist()):
            if not is_valid:
                continue
            raw_chunkwise_right_z[step].append(float(chunkwise_delta[0, step, right_z_action_index]))
            if train_order_chunkwise is not None:
                train_order_chunkwise_right_z[step].append(
                    float(train_order_chunkwise[0, step, right_z_action_index])
                )
        scanned += 1

    _print_header("Right-Z Label Distribution")
    print(f"scanned_frames={scanned} sample_stride={sample_stride}")
    print(f"raw step-wise right z step0: {_format_stats(_summarize(raw_step0_right_z))}")

    print("\nraw chunk-wise right z by step:")
    for step, values in enumerate(raw_chunkwise_right_z):
        print(f"  k={step:02d}: {_format_stats(_summarize(values))}")

    if has_mean_std:
        print("\npre-fix postprocessed chunk-wise right z by step:")
        print("  (normalize raw step-wise -> convert chunk-wise in normalized space -> unnormalize)")
        for step, values in enumerate(train_order_chunkwise_right_z):
            print(f"  k={step:02d}: {_format_stats(_summarize(values))}")

    _print_header("Action Normalization Stats")
    if not action_stats:
        print("No dataset action stats found.")
        return
    for key in ("mean", "std", "min", "max", "q01", "q50", "q99"):
        if key in action_stats:
            print(f"right_z action stats {key}={_stats_value(action_stats[key], right_z_action_index):.12g}")
    if "mean" in action_stats and "std" in action_stats:
        mean = _stats_value(action_stats["mean"], right_z_action_index)
        std = max(_stats_value(action_stats["std"], right_z_action_index), 1e-12)
        normalized_step0 = [(value - mean) / std for value in raw_step0_right_z]
        print(f"normalized raw step-wise right z step0: {_format_stats(_summarize(normalized_step0))}")


def _stats_value(values: Any, index: int) -> float:
    array = np.asarray(values).reshape(-1)
    if index >= len(array):
        raise IndexError(f"stats index {index} outside stats shape {np.asarray(values).shape}.")
    return float(array[index])


def _summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p1": float(np.percentile(array, 1)),
        "p50": float(np.percentile(array, 50)),
        "p99": float(np.percentile(array, 99)),
    }


def _format_stats(stats: dict[str, float | int]) -> str:
    if stats.get("count", 0) == 0:
        return "count=0"
    return (
        f"count={stats['count']} mean={stats['mean']:.9g} std={stats['std']:.9g} "
        f"min={stats['min']:.9g} max={stats['max']:.9g} "
        f"p1={stats['p1']:.9g} p50={stats['p50']:.9g} p99={stats['p99']:.9g}"
    )


def _print_time_semantics(dataset: LeRobotDataset, chunk_size: int) -> None:
    _print_header("Step0 Time Semantics")
    print(f"dataset_fps={dataset.fps}")
    print(f"ACT action_delta_indices=list(range(chunk_size)) -> {list(range(chunk_size))[:min(chunk_size, 12)]}...")
    print(
        "LeRobotDataset resolves action[k] from frame index idx + k. "
        "observation_delta_indices=None for ACT, so observation.state is frame idx."
    )
    print(
        "Therefore action_chunk[0] is paired with the same dataset frame as observation.state. "
        "In the recording loop, observation is read first, action is generated/executed next, and the pair is "
        "saved in the same frame. Any hardware latency is not represented as a separate delta index."
    )


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


if __name__ == "__main__":
    main()
