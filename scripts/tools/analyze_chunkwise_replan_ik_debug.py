#!/usr/bin/env python
"""Analyze chunk-wise replan consistency and IK joint6/joint7 drift debug logs.

The script expects log lines emitted by:
  [CHUNKWISE_REPLAN_DEBUG] {...}
  [IK_JOINT_DRIFT_DEBUG] {...}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CHUNKWISE_TAG = "[CHUNKWISE_REPLAN_DEBUG]"
IK_TAGS = ("[IK_JOINT_DRIFT_DEBUG]", "[IK_JOINT7_DRIFT_DEBUG]")
POSE_AXES = ("x", "y", "z", "rx", "ry", "rz")


def _parse_payload(line: str, tag: str) -> dict[str, Any] | None:
    if tag not in line:
        return None

    start = line.find("{", line.find(tag))
    if start < 0:
        return None

    try:
        payload, _ = json.JSONDecoder().raw_decode(line[start:].strip())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _load_debug_payloads(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunkwise_payloads: list[dict[str, Any]] = []
    ik_payloads: list[dict[str, Any]] = []

    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                chunkwise_payload = _parse_payload(line, CHUNKWISE_TAG)
                if chunkwise_payload is not None:
                    chunkwise_payloads.append(chunkwise_payload)
                    continue

                for ik_tag in IK_TAGS:
                    ik_payload = _parse_payload(line, ik_tag)
                    if ik_payload is not None:
                        ik_payloads.append(ik_payload)
                        break

    return chunkwise_payloads, ik_payloads


def _stats(values: list[float] | np.ndarray, *, absolute: bool = False) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "p95": None, "max": None}
    if absolute:
        array = np.abs(array)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def _print_stats(label: str, values: list[float] | np.ndarray, *, absolute: bool = False) -> None:
    stats = _stats(values, absolute=absolute)
    suffix = " abs" if absolute else ""
    print(
        f"{label}{suffix}: count={_fmt(stats['count'])} "
        f"mean={_fmt(stats['mean'])} std={_fmt(stats['std'])} "
        f"p95={_fmt(stats['p95'])} max={_fmt(stats['max'])}"
    )


def _extract_chunkwise_metrics(
    payloads: list[dict[str, Any]],
    *,
    arm: str,
) -> dict[str, list[float]]:
    metrics = {
        "pose_norm": [],
        "xyz_norm": [],
        "rpy_norm": [],
        "x": [],
        "y": [],
        "z": [],
        "rx": [],
        "ry": [],
        "rz": [],
    }
    for payload in payloads:
        arm_delta = payload.get("step0_delta", {}).get(arm)
        if not arm_delta:
            continue
        delta = np.asarray(arm_delta.get("step0_abs_delta", []), dtype=float)
        if delta.size != 6:
            continue
        metrics["pose_norm"].append(float(np.linalg.norm(delta)))
        metrics["xyz_norm"].append(float(arm_delta.get("step0_xyz_norm", np.linalg.norm(delta[:3]))))
        metrics["rpy_norm"].append(float(arm_delta.get("step0_rpy_norm", np.linalg.norm(delta[3:]))))
        for axis, value in zip(POSE_AXES, delta, strict=True):
            metrics[axis].append(float(value))
    return metrics


def _extract_ik_metrics(
    payloads: list[dict[str, Any]],
    *,
    robot_arm: str,
) -> dict[str, list[float]]:
    metrics = {
        "task_pose_norm": [],
        "task_xyz_norm": [],
        "task_rpy_norm": [],
        "joint6_delta": [],
        "joint6_q_cmd": [],
        "joint6_near_limit": [],
        "joint7_delta": [],
        "joint7_q_cmd": [],
        "joint7_near_limit": [],
    }
    for payload in payloads:
        if payload.get("robot_arm") != robot_arm:
            continue

        target_delta = payload.get("target_minus_current", {})
        joint6 = payload.get("joint6", {})
        joint7 = payload.get("joint7", {})
        joint6_delta = joint6.get("delta")
        joint7_delta = joint7.get("delta")
        if joint6_delta is None and joint7_delta is None:
            continue

        metrics["task_pose_norm"].append(float(target_delta.get("pose_norm", 0.0)))
        metrics["task_xyz_norm"].append(float(target_delta.get("xyz_norm", 0.0)))
        metrics["task_rpy_norm"].append(float(target_delta.get("rpy_norm", 0.0)))
        if joint6_delta is not None:
            metrics["joint6_delta"].append(float(joint6_delta))
        if joint6.get("q_cmd") is not None:
            metrics["joint6_q_cmd"].append(float(joint6["q_cmd"]))
        if joint6.get("near_limit") is not None:
            metrics["joint6_near_limit"].append(float(bool(joint6["near_limit"])))
        if joint7_delta is not None:
            metrics["joint7_delta"].append(float(joint7_delta))
        if joint7.get("q_cmd") is not None:
            metrics["joint7_q_cmd"].append(float(joint7["q_cmd"]))
        if joint7.get("near_limit") is not None:
            metrics["joint7_near_limit"].append(float(bool(joint7["near_limit"])))
    return metrics


def _corrcoef(xs: list[float], ys: list[float]) -> float | None:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return None
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _most_jittery_axis(metrics: dict[str, list[float]]) -> str | None:
    axis_scores = {
        axis: _stats(metrics[axis], absolute=True)["p95"]
        for axis in POSE_AXES
        if metrics.get(axis)
    }
    axis_scores = {axis: score for axis, score in axis_scores.items() if score is not None}
    if not axis_scores:
        return None
    return max(axis_scores, key=lambda axis: float(axis_scores[axis]))


def _print_joint_drift_block(
    *,
    joint_name: str,
    task_pose_norm: list[float],
    joint_delta: list[float],
    near_limit: list[float],
    small_task_space_norm: float,
    large_joint_delta: float,
) -> None:
    print(f"{joint_name} drift")
    _print_stats(f"{joint_name} delta", joint_delta, absolute=True)
    signed = _stats(joint_delta, absolute=False)
    print(f"{joint_name} delta signed mean: {_fmt(signed['mean'])}")
    if near_limit:
        near_limit_rate = float(np.mean(np.asarray(near_limit, dtype=float)))
        print(f"{joint_name} near-limit rate: {near_limit_rate:.3f}")

    correlation = _corrcoef(
        task_pose_norm,
        [abs(value) for value in joint_delta],
    )
    print(f"task-space delta norm vs abs({joint_name} delta) correlation: {_fmt(correlation)}")

    task_norm = np.asarray(task_pose_norm, dtype=float)
    joint_abs = np.abs(np.asarray(joint_delta, dtype=float))
    if task_norm.size != joint_abs.size:
        # Old logs may only contain joint7. Keep the script useful instead of failing the whole analysis.
        sample_count = min(task_norm.size, joint_abs.size)
        task_norm = task_norm[:sample_count]
        joint_abs = joint_abs[:sample_count]
    suspicious = (task_norm < small_task_space_norm) & (joint_abs > large_joint_delta)
    if suspicious.size > 0 and np.any(suspicious):
        count = int(np.sum(suspicious))
        print(
            "Likely IK redundancy / null-space amplification: "
            f"{count}/{suspicious.size} IK samples have task-space norm "
            f"< {small_task_space_norm:g} while abs({joint_name} delta) "
            f"> {large_joint_delta:g} rad."
        )


def analyze(args: argparse.Namespace) -> None:
    chunkwise_payloads, ik_payloads = _load_debug_payloads(args.logs)
    chunkwise_metrics = _extract_chunkwise_metrics(chunkwise_payloads, arm=args.arm)
    robot_arm = f"{args.arm}_robot"
    ik_metrics = _extract_ik_metrics(ik_payloads, robot_arm=robot_arm)

    print(f"Loaded chunkwise debug entries: {len(chunkwise_payloads)}")
    print(f"Loaded IK debug entries: {len(ik_payloads)}")
    print(f"Arm: {args.arm} | robot_arm: {robot_arm}")
    print("")

    print("Chunk-wise replan step0_abs_delta")
    _print_stats("step0 pose norm", chunkwise_metrics["pose_norm"])
    _print_stats("step0 xyz norm", chunkwise_metrics["xyz_norm"])
    _print_stats("step0 rpy norm", chunkwise_metrics["rpy_norm"])
    for axis in POSE_AXES:
        _print_stats(f"step0 {axis}", chunkwise_metrics[axis], absolute=True)
    most_jittery = _most_jittery_axis(chunkwise_metrics)
    if most_jittery is not None:
        print(f"Most jittery step0 component by p95 abs delta: {most_jittery}")
    print("")

    print("IK task-space target change")
    _print_stats("task-space pose norm", ik_metrics["task_pose_norm"])
    _print_stats("task-space xyz norm", ik_metrics["task_xyz_norm"])
    _print_stats("task-space rpy norm", ik_metrics["task_rpy_norm"])
    print("")
    _print_joint_drift_block(
        joint_name="joint6",
        task_pose_norm=ik_metrics["task_pose_norm"],
        joint_delta=ik_metrics["joint6_delta"],
        near_limit=ik_metrics["joint6_near_limit"],
        small_task_space_norm=args.small_task_space_norm,
        large_joint_delta=args.large_joint_delta,
    )
    print("")
    _print_joint_drift_block(
        joint_name="joint7",
        task_pose_norm=ik_metrics["task_pose_norm"],
        joint_delta=ik_metrics["joint7_delta"],
        near_limit=ik_metrics["joint7_near_limit"],
        small_task_space_norm=args.small_task_space_norm,
        large_joint_delta=args.large_joint_delta,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+", help="Log file(s) containing debug JSON lines.")
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument(
        "--small-task-space-norm",
        type=float,
        default=0.02,
        help="Threshold for a small target-current pose norm in mixed m/rad units.",
    )
    parser.add_argument(
        "--large-joint-delta",
        "--large-joint7-delta",
        dest="large_joint_delta",
        type=float,
        default=0.05,
        help="Threshold for a large per-command joint6/joint7 change in radians.",
    )
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
