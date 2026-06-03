#!/usr/bin/env python3
"""Create a left/right mirrored LeRobot v3 dataset.

The tool is intentionally conservative: it copies the source dataset, rewrites
numeric parquet columns from a configurable layout, flips/switches camera
videos, updates metadata, and emits debug summaries for manual inspection.

Usage:
  # Inspect schema and mirror mapping without writing data.
  python dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/tools/mirror_dataset.py --dry-run

  # Generate the mirrored dataset with defaults from the repo-relative config path.
  python dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/tools/mirror_dataset.py --overwrite

  # Validate an already generated mirrored dataset.
  python dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/tools/mirror_dataset.py --validate-only

  # Use a different config, or override src/dst from CLI.
  python dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/tools/mirror_dataset.py \
      --config dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/config/dataset_config/mirror_dataset_cfg.yaml \
      --src dual_arm_data_collection/lerobot_dual_arm_teleop/outputs/src_dataset \
      --dst dual_arm_data_collection/lerobot_dual_arm_teleop/outputs/src_dataset_mirrored \
      --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml
from scipy.spatial.transform import Rotation


INDEX_COLUMNS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
DEFAULT_SRC = Path(
    "dual_arm_data_collection/lerobot_dual_arm_teleop/"
    "outputs/train/task3_step1/dual_empty/act_20260602_E113_cleaned"
)
DEFAULT_DST = DEFAULT_SRC.with_name(DEFAULT_SRC.name + "_mirrored")
DEFAULT_CONFIG = Path(
    "dual_arm_data_collection/lerobot_dual_arm_teleop/scripts/config/dataset_config/mirror_dataset_cfg.yaml"
)


@dataclass(frozen=True)
class SliceSpec:
    name: str
    start: int
    end: int
    kind: str
    pose_format: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror a local LeRobot v3 dataset left/right.")
    parser.add_argument("--src", type=Path, default=None, help="Source dataset root. Overrides dataset.src in config.")
    parser.add_argument("--dst", type=Path, default=None, help="Output dataset root. Overrides dataset.dst in config.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Mirror YAML config.")
    parser.add_argument("--dry-run", action="store_true", default=None, help="Inspect schema/mapping without writing dataset.")
    parser.add_argument("--overwrite", action="store_true", default=None, help="Replace dst if it already exists.")
    parser.add_argument("--num-vis-episodes", type=int, default=None)
    parser.add_argument("--num-vis-frames-per-episode", type=int, default=None)
    parser.add_argument("--skip-video", action="store_true", default=None, help="Do not rewrite videos.")
    parser.add_argument("--skip-fk-check", action="store_true", default=None, help="Do not attempt FK validation.")
    parser.add_argument("--validate-only", action="store_true", default=None, help="Validate an already generated mirrored dataset.")
    return parser.parse_args()


def resolve_config_path(path: Path) -> Path:
    path = path.expanduser()
    if path.exists():
        return path
    migrated = path.parent / "dataset_config" / path.name
    if migrated.exists():
        return migrated
    if path.is_absolute():
        return path
    script_relative = Path(__file__).resolve().parents[1] / "config" / "dataset_config" / path.name
    if script_relative.exists():
        return script_relative
    return path


def config_path_value(cfg: dict[str, Any], section: str, key: str, default: Path) -> Path:
    value = (cfg.get(section, {}) or {}).get(key)
    return Path(value).expanduser() if value else default


def config_bool_value(cfg: dict[str, Any], section: str, key: str, default: bool = False) -> bool:
    return bool((cfg.get(section, {}) or {}).get(key, default))


def apply_config_defaults(args: argparse.Namespace, cfg: dict[str, Any]) -> argparse.Namespace:
    run_cfg = cfg.get("run", {}) or {}
    args.src = args.src if args.src is not None else config_path_value(cfg, "dataset", "src", DEFAULT_SRC)
    args.dst = args.dst if args.dst is not None else config_path_value(cfg, "dataset", "dst", DEFAULT_DST)
    args.dry_run = bool(args.dry_run) if args.dry_run is not None else config_bool_value(cfg, "run", "dry_run")
    args.overwrite = bool(args.overwrite) if args.overwrite is not None else config_bool_value(cfg, "run", "overwrite")
    args.skip_video = bool(args.skip_video) if args.skip_video is not None else config_bool_value(cfg, "run", "skip_video")
    args.skip_fk_check = (
        bool(args.skip_fk_check) if args.skip_fk_check is not None else config_bool_value(cfg, "run", "skip_fk_check")
    )
    args.validate_only = (
        bool(args.validate_only) if args.validate_only is not None else config_bool_value(cfg, "run", "validate_only")
    )
    if args.num_vis_episodes is None:
        args.num_vis_episodes = run_cfg.get("num_vis_episodes")
    if args.num_vis_frames_per_episode is None:
        args.num_vis_frames_per_episode = run_cfg.get("num_vis_frames_per_episode")
    return args


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return cfg


def list_parquets(root: Path, relative_dir: str) -> list[Path]:
    return sorted((root / relative_dir).glob("*/*.parquet"))


def data_parquets(root: Path) -> list[Path]:
    paths = list_parquets(root, "data")
    if not paths:
        raise FileNotFoundError(f"No data parquet files found under {root / 'data'}")
    return paths


def resolve_src(src: Path) -> Path:
    src = src.expanduser().resolve()
    if src.exists():
        return src
    hf_home = Path.home() / ".cache" / "huggingface" / "lerobot"
    candidates = [
        hf_home / "nero_task3_step1" / "empty_merged_E113_cleaned",
        hf_home / "nero_task3_step1" / "empty_merged_E113_cleaned_annotated",
    ]
    for candidate in candidates:
        if candidate.exists():
            print(f"[warn] Requested --src does not exist: {src}")
            print(f"[warn] Using detected local dataset instead: {candidate}")
            return candidate.resolve()
    raise FileNotFoundError(f"Source dataset does not exist: {src}")


def flatten_names(feature: dict[str, Any] | None) -> list[str] | None:
    if not feature:
        return None
    names = feature.get("names")
    if names is None:
        return None
    if names and all(isinstance(item, list) for item in names):
        return [str(part) for row in names for part in row]
    return [str(item) for item in names]


def schema_summary(root: Path) -> dict[str, Any]:
    info = read_json(root / "meta" / "info.json")
    parquet = data_parquets(root)[0]
    schema = pq.ParquetFile(parquet).schema_arrow
    return {
        "root": str(root),
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "parquet_columns": schema.names,
        "features": info.get("features", {}),
        "data_files": [str(p.relative_to(root)) for p in data_parquets(root)],
        "video_files": [str(p.relative_to(root)) for p in sorted((root / "videos").glob("**/*.mp4"))],
    }


def parse_layout(entries: list[dict[str, Any]], label: str) -> list[SliceSpec]:
    specs: list[SliceSpec] = []
    for item in entries or []:
        raw_slice = item.get("slice")
        if not isinstance(raw_slice, list) or len(raw_slice) != 2:
            raise ValueError(f"{label}.layout entry {item.get('name')!r} needs slice: [start, end]")
        specs.append(
            SliceSpec(
                name=str(item["name"]),
                start=int(raw_slice[0]),
                end=int(raw_slice[1]),
                kind=str(item["type"]),
                pose_format=item.get("pose_format"),
            )
        )
    return specs


def as_2d(values: Any, dim: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1 and dim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"Expected array [frames, {dim}], got {arr.shape}")
    return arr


def reflection_matrix(cfg: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(cfg.get("mirror", {}).get("reflection_matrix", [1, 0, 0, 0, -1, 0, 0, 0, 1]))
    matrix = matrix.astype(np.float64).reshape(3, 3)
    if not np.allclose(matrix @ matrix, np.eye(3)):
        raise ValueError(f"reflection_matrix must square to identity, got {matrix}")
    return matrix


def mirror_rotvec(rotvec: np.ndarray, s_mat: np.ndarray) -> np.ndarray:
    flat = rotvec.reshape(-1, 3)
    rot = Rotation.from_rotvec(flat)
    mirrored = s_mat @ rot.as_matrix() @ s_mat
    # For an improper reflection S, S exp([w]x) S = exp([det(S) S w]x).
    # This keeps the original axis-angle branch and makes double mirror exact.
    candidate = (np.linalg.det(s_mat) * (flat @ s_mat.T)).reshape(-1, 3)
    candidate_matrix = Rotation.from_rotvec(candidate).as_matrix()
    if not np.allclose(candidate_matrix, mirrored, atol=1e-7):
        candidate = Rotation.from_matrix(mirrored).as_rotvec()
    return candidate.reshape(rotvec.shape)


def mirror_pose(values: np.ndarray, pose_format: str, s_mat: np.ndarray) -> np.ndarray:
    if values.shape[1] != 6:
        raise ValueError(f"Only 6D pose slices are supported for {pose_format}, got {values.shape}")
    out = values.copy()
    out[:, 0:3] = values[:, 0:3] @ s_mat.T
    fmt = pose_format.lower()
    if fmt in {"rotvec", "axis-angle", "axis_angle"}:
        out[:, 3:6] = mirror_rotvec(values[:, 3:6], s_mat)
    elif fmt in {"rpy", "euler_xyz", "xyz"}:
        rot = Rotation.from_euler("xyz", values[:, 3:6])
        mirrored = s_mat @ rot.as_matrix() @ s_mat
        out[:, 3:6] = Rotation.from_matrix(mirrored).as_euler("xyz")
    else:
        raise ValueError(f"Unsupported pose_format={pose_format!r}; use rotvec or euler_xyz/rpy.")
    return out


def mirror_gripper(values: np.ndarray) -> np.ndarray:
    return values.copy()


def joint_params(cfg: dict[str, Any], dst_kind: str) -> tuple[np.ndarray, np.ndarray]:
    joint_cfg = cfg.get("joint_mirror", {}) or {}
    if dst_kind == "joint_left":
        sign = joint_cfg.get("left_from_right_sign")
        offset = joint_cfg.get("left_from_right_offset")
    elif dst_kind == "joint_right":
        sign = joint_cfg.get("right_from_left_sign")
        offset = joint_cfg.get("right_from_left_offset")
    else:
        raise ValueError(dst_kind)
    if sign is None or offset is None:
        raise ValueError("joint_mirror sign/offset must be configured before generating mirrored joints.")
    return np.asarray(sign, dtype=np.float64), np.asarray(offset, dtype=np.float64)


def transform_layout(values: np.ndarray, specs: list[SliceSpec], cfg: dict[str, Any]) -> np.ndarray:
    out = values.copy()
    by_kind = {spec.kind: spec for spec in specs}
    s_mat = reflection_matrix(cfg)

    def source_for(kind: str) -> SliceSpec:
        side_swaps = {
            "joint_left": "joint_right",
            "joint_right": "joint_left",
            "ee_pose_left": "ee_pose_right",
            "ee_pose_right": "ee_pose_left",
            "delta_ee_pose_left": "delta_ee_pose_right",
            "delta_ee_pose_right": "delta_ee_pose_left",
            "gripper_left": "gripper_right",
            "gripper_right": "gripper_left",
        }
        src_kind = side_swaps.get(kind, kind)
        if src_kind not in by_kind:
            raise ValueError(f"Cannot mirror {kind}: missing source slice {src_kind}")
        return by_kind[src_kind]

    for dst in specs:
        src = source_for(dst.kind)
        src_values = values[:, src.start : src.end]
        if dst.kind in {"joint_left", "joint_right"}:
            sign, offset = joint_params(cfg, dst.kind)
            if len(sign) != dst.end - dst.start or len(offset) != dst.end - dst.start:
                raise ValueError(f"joint_mirror length mismatch for {dst.kind}")
            transformed = src_values * sign + offset
        elif dst.kind in {"ee_pose_left", "ee_pose_right", "delta_ee_pose_left", "delta_ee_pose_right"}:
            pose_format = dst.pose_format or src.pose_format
            if not pose_format:
                raise ValueError(f"pose_format is required for {dst.name}")
            transformed = mirror_pose(src_values, pose_format, s_mat)
        elif dst.kind in {"gripper_left", "gripper_right"}:
            transformed = mirror_gripper(src_values)
        elif dst.kind == "copy":
            transformed = src_values.copy()
        else:
            raise ValueError(f"Unsupported layout type {dst.kind!r} in {dst.name}")
        out[:, dst.start : dst.end] = transformed.astype(out.dtype, copy=False)
    return out


def swap_left_right_annotation_columns(df: Any) -> Any:
    for col in list(df.columns):
        if ".left_" not in col:
            continue
        right = col.replace(".left_", ".right_", 1)
        if right in df.columns:
            left_values = df[col].copy()
            df[col] = df[right]
            df[right] = left_values
    return df


def rewrite_data_parquets(src: Path, dst: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    info = read_json(src / "meta" / "info.json")
    action_dim = int(np.prod(info["features"][cfg["action"]["key"]]["shape"]))
    state_dim = int(np.prod(info["features"][cfg["state"]["key"]]["shape"]))
    action_specs = parse_layout(cfg["action"]["layout"], "action")
    state_specs = parse_layout(cfg["state"]["layout"], "state")
    finite_columns: dict[str, dict[str, Any]] = {}
    row_count = 0

    for src_file in data_parquets(src):
        rel = src_file.relative_to(src)
        dst_file = dst / rel
        df = pd.read_parquet(src_file)
        if cfg["action"]["key"] in df:
            values = as_2d(df[cfg["action"]["key"]].tolist(), action_dim)
            mirrored = transform_layout(values, action_specs, cfg)
            if not np.isfinite(mirrored).all():
                raise ValueError(f"NaN/Inf after action mirror: {src_file}")
            df[cfg["action"]["key"]] = list(mirrored.astype(np.float32))
            finite_columns[cfg["action"]["key"]] = {"shape": list(mirrored.shape), "finite": True}
        if cfg["state"]["key"] in df:
            values = as_2d(df[cfg["state"]["key"]].tolist(), state_dim)
            mirrored = transform_layout(values, state_specs, cfg)
            if not np.isfinite(mirrored).all():
                raise ValueError(f"NaN/Inf after state mirror: {src_file}")
            df[cfg["state"]["key"]] = list(mirrored.astype(np.float32))
            finite_columns[cfg["state"]["key"]] = {"shape": list(mirrored.shape), "finite": True}
        df = swap_left_right_annotation_columns(df)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst_file, index=False)
        row_count += len(df)
    return {"rewritten_rows": row_count, "finite_columns": finite_columns}


def quantile_stats(values: np.ndarray) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "min": np.min(values, axis=0).astype(float).tolist(),
        "max": np.max(values, axis=0).astype(float).tolist(),
        "mean": np.mean(values, axis=0).astype(float).tolist(),
        "std": np.std(values, axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
    }
    for name, q in {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}.items():
        stats[name] = np.quantile(values, q, axis=0).astype(float).tolist()
    return stats


def recompute_numeric_stats(dst: Path, cfg: dict[str, Any]) -> None:
    import pandas as pd

    frames = [pd.read_parquet(path) for path in data_parquets(dst)]
    full = pd.concat(frames, ignore_index=True)
    stats_path = dst / "meta" / "stats.json"
    stats = read_json(stats_path) if stats_path.exists() else {}
    for key in [cfg["action"]["key"], cfg["state"]["key"]]:
        if key in full:
            stats[key] = quantile_stats(np.asarray(full[key].tolist(), dtype=np.float64))
    write_json(stats_path, stats)

    episode_files = list_parquets(dst, "meta/episodes")
    if not episode_files:
        return
    for ep_file in episode_files:
        ep_df = pd.read_parquet(ep_file)
        for row_idx, row in ep_df.iterrows():
            ep_index = int(row["episode_index"])
            ep_rows = full[full["episode_index"] == ep_index]
            for key in [cfg["action"]["key"], cfg["state"]["key"]]:
                if key not in ep_rows:
                    continue
                per_ep = quantile_stats(np.asarray(ep_rows[key].tolist(), dtype=np.float64))
                for stat_name, stat_value in per_ep.items():
                    col = f"stats/{key}/{stat_name}"
                    if col in ep_df.columns:
                        ep_df.at[row_idx, col] = stat_value
        ep_df = swap_left_right_episode_stats(ep_df)
        ep_df.to_parquet(ep_file, index=False)


def swap_left_right_episode_stats(df: Any) -> Any:
    for col in list(df.columns):
        if "/observation.images.left_" not in col:
            continue
        right = col.replace("/observation.images.left_", "/observation.images.right_", 1)
        if right in df.columns:
            left_values = df[col].copy()
            df[col] = df[right]
            df[right] = left_values
    return df


def update_metadata(src: Path, dst: Path, cfg_path: Path, cfg: dict[str, Any], summary: dict[str, Any]) -> None:
    info_path = dst / "meta" / "info.json"
    info = read_json(info_path)
    info["mirrored_from"] = str(src)
    info["mirror_config"] = str(cfg_path)
    info["mirror_config_digest"] = str(abs(hash(json.dumps(cfg, sort_keys=True))))
    info["mirror_created_at"] = datetime.now(timezone.utc).isoformat()
    info["mirror_type"] = "left_right_reflection_y"
    if "repo_id" in info:
        info["repo_id"] = f"{info['repo_id']}_mirrored"
    for feature in info.get("features", {}).values():
        if isinstance(feature, dict) and feature.get("dtype") == "video":
            feature.setdefault("info", {})["video.codec"] = "mp4v"
    write_json(info_path, info)
    write_json(dst / "mirror_debug" / "mirror_summary.json", summary)


def copy_dataset(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists; pass --overwrite to replace it: {dst}")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def camera_mapping(cfg: dict[str, Any]) -> dict[str, str]:
    cameras = cfg.get("cameras", {})
    return {
        str(cam["dst_key"]): str(cam.get("from_src_key") or cam["src_key"])
        for cam in cameras.values()
    }


def transform_videos(src: Path, dst: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    import cv2

    video_summary: dict[str, Any] = {"rewritten": [], "warnings": []}
    for dst_key, src_key in camera_mapping(cfg).items():
        src_dir = src / "videos" / src_key
        dst_dir = dst / "videos" / dst_key
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing source video dir for {src_key}: {src_dir}")
        for src_video in sorted(src_dir.glob("**/*.mp4")):
            rel = src_video.relative_to(src_dir)
            dst_video = dst_dir / rel
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            metadata = video_metadata(src_video, default_fps=float(cfg.get("fps", 30)))
            fps = metadata["fps"]
            width = metadata["width"]
            height = metadata["height"]
            expected = metadata["frames"]
            writer = cv2.VideoWriter(str(dst_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create video writer: {dst_video}")
            count = 0
            for frame in iter_video_frames_bgr(src_video):
                writer.write(cv2.flip(frame, 1))
                count += 1
            writer.release()
            if count == 0:
                raise RuntimeError(f"Decoded zero frames from video: {src_video}")
            if expected and count != expected:
                video_summary["warnings"].append(f"{src_video}: expected {expected} frames, wrote {count}")
            video_summary["rewritten"].append(
                {"dst_key": dst_key, "from_src_key": src_key, "file": str(dst_video.relative_to(dst)), "frames": count}
            )
    video_summary["codec_note"] = "Videos are re-encoded with OpenCV mp4v, so pixels may differ slightly."
    return video_summary


def video_metadata(path: Path, default_fps: float) -> dict[str, Any]:
    try:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.base_rate
            fps = float(rate) if rate is not None else float(default_fps)
            return {
                "fps": fps,
                "width": int(stream.width),
                "height": int(stream.height),
                "frames": int(stream.frames or 0),
                "codec": stream.codec_context.name,
            }
    except Exception:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        metadata = {
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or default_fps),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "codec": "opencv",
        }
        cap.release()
        return metadata


def iter_video_frames_bgr(path: Path):
    try:
        import av

        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                yield frame.to_ndarray(format="bgr24")
        return
    except Exception as av_error:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video with PyAV or OpenCV: {path}") from av_error
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()


def read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    try:
        import av

        with av.open(str(path)) as container:
            for idx, frame in enumerate(container.decode(video=0)):
                if idx == frame_index:
                    return frame.to_ndarray(format="bgr24")
    except Exception:
        pass

    import cv2

    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if ok:
            return frame
    else:
        cap.release()

    for idx, frame in enumerate(iter_video_frames_bgr(path)):
        if idx == frame_index:
            return frame
    raise RuntimeError(f"Cannot read frame {frame_index} from {path}")


def write_png(path: Path, image_bgr: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_bgr)


def make_debug_images(src: Path, dst: Path, cfg: dict[str, Any], frames_per_episode: int) -> list[str]:
    out_files: list[str] = []
    first_videos = {key: next((src / "videos" / key).glob("**/*.mp4")) for key in camera_mapping(cfg).values()}
    dst_videos = {key: next((dst / "videos" / key).glob("**/*.mp4")) for key in camera_mapping(cfg).keys()}
    for dst_key, src_key in camera_mapping(cfg).items():
        for idx in range(max(0, frames_per_episode)):
            before = read_video_frame(first_videos[src_key], idx)
            after = read_video_frame(dst_videos[dst_key], idx)
            combined = np.concatenate([before, after], axis=1)
            name = f"ep_000_frame_{idx:03d}_{dst_key.split('.')[-1]}_before_after.png"
            rel = Path("mirror_debug") / "sample_frames" / name
            write_png(dst / rel, combined)
            out_files.append(str(rel))
    return out_files


def make_trajectory_debug(dst: Path, cfg: dict[str, Any]) -> list[str]:
    import cv2
    import pandas as pd

    files: list[str] = []
    first = pd.read_parquet(data_parquets(dst)[0])
    ep0 = first[first["episode_index"] == first["episode_index"].iloc[0]]
    for key, specs in [
        (cfg["action"]["key"], parse_layout(cfg["action"]["layout"], "action")),
        (cfg["state"]["key"], parse_layout(cfg["state"]["layout"], "state")),
    ]:
        if key not in ep0:
            continue
        values = np.asarray(ep0[key].tolist(), dtype=np.float64)
        xyz_specs = [s for s in specs if "ee_pose" in s.kind]
        if not xyz_specs:
            continue
        canvas = np.full((420, 620, 3), 255, dtype=np.uint8)
        colors = [(30, 90, 220), (220, 90, 30), (40, 160, 70), (160, 40, 180)]
        for color, spec in zip(colors, xyz_specs):
            xyz = values[:, spec.start : spec.start + 3]
            xy = xyz[:, :2]
            xy = xy - xy.min(axis=0, keepdims=True)
            denom = np.maximum(xy.max(axis=0, keepdims=True), 1e-9)
            pts = (xy / denom * np.array([560, 360]) + np.array([30, 30])).astype(np.int32)
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(canvas, tuple(a), tuple(b), color, 1)
        rel = Path("mirror_debug") / "trajectories" / f"ep_000_{key.replace('.', '_')}_xyz_before_after.png"
        write_png(dst / rel, canvas)
        files.append(str(rel))
    return files


def validate_alignment(src: Path, dst: Path) -> dict[str, Any]:
    src_info = read_json(src / "meta" / "info.json")
    dst_info = read_json(dst / "meta" / "info.json")
    checks = {
        "episode_count_same": src_info.get("total_episodes") == dst_info.get("total_episodes"),
        "frame_count_same": src_info.get("total_frames") == dst_info.get("total_frames"),
        "data_file_count_same": len(data_parquets(src)) == len(data_parquets(dst)),
    }
    for src_file in data_parquets(src):
        dst_file = dst / src_file.relative_to(src)
        src_table = pq.read_table(src_file, columns=[c for c in INDEX_COLUMNS if c in pq.ParquetFile(src_file).schema_arrow.names])
        dst_table = pq.read_table(dst_file, columns=[c for c in INDEX_COLUMNS if c in pq.ParquetFile(dst_file).schema_arrow.names])
        checks[f"indices_same/{src_file.relative_to(src)}"] = src_table.equals(dst_table)
    if not all(checks.values()):
        raise ValueError(f"Alignment validation failed: {checks}")
    return checks


def double_mirror_check(src: Path, cfg: dict[str, Any]) -> dict[str, float]:
    import pandas as pd

    info = read_json(src / "meta" / "info.json")
    df = pd.read_parquet(data_parquets(src)[0]).head(64)
    result: dict[str, float] = {}
    for section in ["action", "state"]:
        key = cfg[section]["key"]
        dim = int(np.prod(info["features"][key]["shape"]))
        values = as_2d(df[key].tolist(), dim)
        specs = parse_layout(cfg[section]["layout"], section)
        restored = transform_layout(transform_layout(values, specs, cfg), specs, cfg)
        result[f"{key}_double_mirror_max_abs_error"] = float(np.max(np.abs(restored - values)))
    return result


def inspect_and_print(src: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    summary = schema_summary(src)
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:8000])
    print("\nCamera mapping:")
    for dst_key, src_key in camera_mapping(cfg).items():
        print(f"  {dst_key} <- horizontal_flip({src_key})")
    joint_cfg = cfg.get("joint_mirror", {}) or {}
    if not joint_cfg.get("verified", False):
        print("[warn] joint_mirror.verified=false; sign/offset map must be manually verified for Nero.")
    if (cfg.get("action", {}) or {}).get("mode") == "auto":
        raise ValueError("action.mode must be configured; this dataset uses delta_ee_pose actions.")
    return summary


def fk_check_summary(skip: bool) -> dict[str, str]:
    if skip:
        return {"status": "skipped", "reason": "--skip-fk-check was set"}
    return {"status": "skipped", "reason": "No Nero FK/URDF validation utility was found in this repository."}


def main() -> None:
    args = parse_args()
    args.config = resolve_config_path(args.config)
    cfg = load_config(args.config)
    args = apply_config_defaults(args, cfg)
    src = resolve_src(args.src)
    dst = args.dst.expanduser().resolve()

    src_schema = inspect_and_print(src, cfg)
    dry_check = double_mirror_check(src, cfg)
    print("\nDouble mirror check:", dry_check)

    if args.dry_run:
        debug_dir = Path("mirror_debug")
        print(f"[dry-run] Would write dataset to: {dst}")
        print(f"[dry-run] Would write debug summaries under: {dst / debug_dir}")
        return

    if args.validate_only:
        summary = {"alignment": validate_alignment(src, dst), "double_mirror_source": dry_check}
        write_json(dst / "mirror_debug" / "validate_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    copy_dataset(src, dst, args.overwrite)
    (dst / "mirror_debug").mkdir(parents=True, exist_ok=True)
    write_json(dst / "mirror_debug" / "schema_summary.json", src_schema)
    data_summary = rewrite_data_parquets(src, dst, cfg)
    recompute_numeric_stats(dst, cfg)
    if args.skip_video:
        video_summary = {"status": "skipped", "warning": "Videos were copied but not mirrored."}
    else:
        video_summary = transform_videos(src, dst, cfg)
    vis_frames = args.num_vis_frames_per_episode
    if vis_frames is None:
        vis_frames = int((cfg.get("validation", {}) or {}).get("num_frames_per_episode", 2))
    debug_files: list[str] = []
    if not args.skip_video:
        debug_files.extend(make_debug_images(src, dst, cfg, vis_frames))
    debug_files.extend(make_trajectory_debug(dst, cfg))
    summary = {
        "source": str(src),
        "destination": str(dst),
        "data": data_summary,
        "video": video_summary,
        "alignment": validate_alignment(src, dst),
        "double_mirror_source": dry_check,
        "fk_validation": fk_check_summary(args.skip_fk_check),
        "debug_files": debug_files,
        "risks": [
            "joint_mirror sign/offset is marked unverified in config"
            if not (cfg.get("joint_mirror", {}) or {}).get("verified", False)
            else "joint_mirror sign/offset marked verified by config",
        ],
    }
    update_metadata(src, dst, args.config, cfg, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
