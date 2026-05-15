#!/usr/bin/env python

from __future__ import annotations

import copy
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DAggerRoundsConfig:
    output_root: str | Path
    record_cfg_path: str | Path
    train_cfg_path: str | Path | None = None
    num_rounds: int | None = None
    episodes_per_round: int = 1
    seed_repo_id: str | None = None
    seed_dataset_path: str | Path | None = None
    seed_root: str | Path | None = None
    initial_pretrained_path: str | Path | None = None
    overwrite: bool = False
    train_round0_if_missing: bool = True
    export_min_segment_frames: int = 1
    min_episode_len_for_act: int | None = None
    pre_takeover_context: int = 0
    require_complete_expert_action: bool = True
    round_schedule: dict[str, Any] | None = None
    policy_backend: dict[str, Any] | None = None


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _clean_or_fail(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Path already exists: {path}")
    shutil.rmtree(path)


def _safe_repo_id(prefix: str, round_idx: int, suffix: str) -> str:
    return f"{prefix}_{suffix}_r{round_idx:03d}".replace("/", "_")


def _resolve_seed(cfg: DAggerRoundsConfig) -> tuple[str, Path]:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    seed_root = cfg.seed_dataset_path or cfg.seed_root
    if seed_root is not None:
        root = Path(seed_root)
    elif cfg.seed_repo_id is not None:
        root = HF_LEROBOT_HOME / cfg.seed_repo_id
    else:
        raise ValueError("One of seed_dataset_path, seed_root, or seed_repo_id is required.")

    repo_id = cfg.seed_repo_id or root.name
    return repo_id, root


def _require_checkpoint_exists(path: Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} checkpoint path is required.")
    if not path.is_dir():
        raise FileNotFoundError(f"{label} checkpoint path does not exist or is not a directory: {path}")
    missing = [name for name in ("config.json", "model.safetensors") if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{label} checkpoint is missing required file(s) {missing}: {path}")
    return path


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _clamp_steps(steps: int, min_steps: int | None, max_steps: int | None) -> int:
    if min_steps is not None:
        steps = max(int(min_steps), steps)
    if max_steps is not None:
        steps = min(int(max_steps), steps)
    return int(steps)


def _round_schedule_cfg(config: DAggerRoundsConfig) -> dict[str, Any]:
    return copy.deepcopy(config.round_schedule or {})


def _resolve_max_rounds(config: DAggerRoundsConfig, schedule_cfg: dict[str, Any]) -> int:
    max_rounds = schedule_cfg.get("max_rounds")
    if max_rounds is not None:
        return int(max_rounds)
    if config.num_rounds is None:
        raise ValueError("Either dagger_rounds.num_rounds or round_schedule.max_rounds is required.")
    return int(config.num_rounds)


def _train_steps_cfg(schedule_cfg: dict[str, Any] | None) -> dict[str, Any]:
    if not schedule_cfg:
        return {}
    train_steps = schedule_cfg.get("train_steps")
    if isinstance(train_steps, dict):
        return train_steps
    return schedule_cfg


def _early_stop_cfg(schedule_cfg: dict[str, Any] | None) -> dict[str, Any]:
    if not schedule_cfg:
        return {}
    early_stop = schedule_cfg.get("early_stop")
    if isinstance(early_stop, dict):
        return early_stop
    return {}


def _exported_counts(exported_summary: dict[str, Any] | None) -> tuple[int, int]:
    exported_dataset = (exported_summary or {}).get("exported_dataset", {})
    exported_frames = int(exported_dataset.get("total_frames", 0) or 0)
    exported_episodes = int(exported_dataset.get("total_episodes", 0) or 0)
    return exported_frames, exported_episodes


def _export_metrics(export_summary: dict[str, Any], run_mix_stats: dict[str, Any] | None = None) -> dict[str, Any]:
    exported_frames, exported_episodes = _exported_counts(export_summary)
    stats = export_summary.get("stats", {})
    run_mix_stats = run_mix_stats or {}
    expert_frame_ratio = stats.get("expert_frame_ratio", run_mix_stats.get("expert_frame_ratio", 0.0))
    intervention_count = stats.get("intervention_count", run_mix_stats.get("intervention_count", 0))
    return {
        "exported_frames": int(exported_frames),
        "exported_episodes": int(exported_episodes),
        "expert_frame_ratio": float(expert_frame_ratio or 0.0),
        "intervention_count": int(intervention_count or 0),
        "skipped_short_segments": int(stats.get("skipped_short_segments", 0) or 0),
        "incomplete_expert_label_frames": stats.get("incomplete_expert_label_frames"),
    }


def resolve_train_steps(
    round_id: int,
    exported_summary: dict[str, Any],
    base_train_steps: int,
    schedule_cfg: dict[str, Any] | None,
) -> int:
    """Resolve per-round train steps for the supported round DAgger controller.

    This helper only schedules the round-based `robot-dagger` path. It is not
    related to the removed offline DAgger trainer/pipeline.
    """

    train_steps_cfg = _train_steps_cfg(schedule_cfg)
    mode = str(train_steps_cfg.get("mode", "fixed"))
    exported_frames, exported_episodes = _exported_counts(exported_summary)
    if exported_frames <= 0 or exported_episodes <= 0:
        return 0

    fixed_steps = _optional_int(train_steps_cfg.get("fixed_steps"))
    min_steps = _optional_int(train_steps_cfg.get("min_steps"))
    max_steps = _optional_int(train_steps_cfg.get("max_steps"))
    round1_steps = _optional_int(train_steps_cfg.get("round1_steps"))
    decay_per_round = _optional_float(train_steps_cfg.get("decay_per_round"))
    if decay_per_round is None:
        decay_per_round = 1.0
    if decay_per_round <= 0:
        raise ValueError("round_schedule.train_steps.decay_per_round must be > 0.")

    if round_id == 1 and round1_steps is not None:
        resolved = round1_steps
        clamp_result = True
    elif mode == "fixed":
        resolved = fixed_steps if fixed_steps is not None else int(base_train_steps)
        clamp_result = decay_per_round != 1.0
    elif mode == "by_exported_frames":
        steps_per_frame = int(train_steps_cfg.get("steps_per_exported_frame", 1))
        resolved = exported_frames * steps_per_frame
        clamp_result = True
    elif mode == "by_exported_episodes":
        steps_per_episode = int(train_steps_cfg.get("steps_per_exported_episode", int(base_train_steps)))
        resolved = exported_episodes * steps_per_episode
        clamp_result = True
    elif mode == "hybrid":
        steps_per_frame = int(train_steps_cfg.get("steps_per_exported_frame", 1))
        steps_per_episode = int(train_steps_cfg.get("steps_per_exported_episode", int(base_train_steps)))
        frame_steps = exported_frames * steps_per_frame
        episode_steps = exported_episodes * steps_per_episode
        resolved = max(frame_steps, episode_steps)
        clamp_result = True
    else:
        raise ValueError(
            "round_schedule.train_steps.mode must be one of: "
            "fixed, by_exported_frames, by_exported_episodes, hybrid."
        )

    if decay_per_round != 1.0:
        decay_factor = decay_per_round ** max(0, int(round_id) - 1)
        resolved = int(round(float(resolved) * decay_factor))
        clamp_result = True

    if clamp_result:
        resolved = _clamp_steps(int(resolved), min_steps, max_steps)
    return int(resolved)


def _dagger_rounds_with_current(
    round_history: list[dict[str, Any]],
    current_round_state: dict[str, Any],
) -> list[dict[str, Any]]:
    rounds = [state for state in round_history if int(state.get("round_id", 0) or 0) > 0]
    rounds.append(current_round_state)
    return rounds


def _tail_count(rounds: list[dict[str, Any]], predicate) -> int:
    count = 0
    for state in reversed(rounds):
        if predicate(state):
            count += 1
        else:
            break
    return count


def should_stop_dagger(
    round_history: list[dict[str, Any]],
    current_round_state: dict[str, Any],
    schedule_cfg: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    """Evaluate early stopping for the supported round DAgger controller only."""

    early_stop_cfg = _early_stop_cfg(schedule_cfg)
    if not bool(early_stop_cfg.get("enabled", False)):
        return False, None

    exported_frames = int(current_round_state.get("exported_frames", 0) or 0)
    exported_episodes = int(current_round_state.get("exported_episodes", 0) or 0)
    intervention_count = int(current_round_state.get("intervention_count", 0) or 0)

    if bool(early_stop_cfg.get("stop_if_no_new_data", True)) and (
        exported_frames <= 0 or exported_episodes <= 0
    ):
        if intervention_count == 0 and exported_frames <= 0:
            return (
                True,
                "no exported DAgger data: intervention_count=0 and exported_frames=0 "
                "(policy may already be good, or this round had no effective takeover)",
            )
        return (
            True,
            f"no exported DAgger data: exported_frames={exported_frames}, "
            f"exported_episodes={exported_episodes}",
        )

    patience = max(1, int(early_stop_cfg.get("patience", 1) or 1))
    rounds = _dagger_rounds_with_current(round_history, current_round_state)

    if bool(early_stop_cfg.get("stop_if_export_too_small", True)):
        min_exported_frames = int(early_stop_cfg.get("min_exported_frames", 1) or 0)
        min_exported_episodes = int(early_stop_cfg.get("min_exported_episodes", 1) or 0)
        small_count = _tail_count(
            rounds,
            lambda state: int(state.get("exported_frames", 0) or 0) < min_exported_frames
            or int(state.get("exported_episodes", 0) or 0) < min_exported_episodes,
        )
        if small_count >= patience:
            return (
                True,
                f"export too small for {small_count} consecutive round(s): "
                f"min_exported_frames={min_exported_frames}, "
                f"min_exported_episodes={min_exported_episodes}",
            )

    if bool(early_stop_cfg.get("stop_if_low_expert_ratio", False)):
        min_expert_frame_ratio = float(early_stop_cfg.get("min_expert_frame_ratio", 0.0) or 0.0)
        low_ratio_count = _tail_count(
            rounds,
            lambda state: float(state.get("expert_frame_ratio", 0.0) or 0.0) < min_expert_frame_ratio,
        )
        if low_ratio_count >= patience:
            return (
                True,
                f"expert_frame_ratio too low for {low_ratio_count} consecutive round(s): "
                f"min_expert_frame_ratio={min_expert_frame_ratio}",
            )

    return False, None


def _make_record_section(
    base_record_cfg: dict[str, Any],
    round_idx: int,
    round_dir: Path,
    checkpoint_path: Path,
    episodes_per_round: int,
    repo_prefix: str,
    policy_runner,
    round_cfg: dict[str, Any],
) -> dict[str, Any]:
    record_section = copy.deepcopy(base_record_cfg)
    raw_root = round_dir / "raw_run_mix"
    raw_repo_id = _safe_repo_id(repo_prefix, round_idx, "raw_run_mix")

    record_section["repo_id"] = raw_repo_id
    record_section["dataset_name"] = raw_repo_id
    record_section["dataset_root"] = str(raw_root)
    record_section["dataset_path"] = str(raw_root)
    record_section["run_mode"] = "run_mix"
    record_section.setdefault("task", {})
    record_section["task"]["num_episodes"] = episodes_per_round
    record_section["task"]["resume"] = False
    record_section.setdefault("storage", {})
    record_section["storage"]["push_to_hub"] = False
    return policy_runner.prepare_record_cfg(record_section, checkpoint_path, round_cfg)


def _train_round(
    trainer,
    base_train_cfg: dict[str, Any],
    round_idx: int,
    round_dir: Path,
    checkpoint_path: Path | None,
    aggregated_repo_id: str,
    aggregated_root: Path,
    seed_repo_id: str,
    seed_root: Path,
    train_steps: int | None = None,
) -> Path:
    train_output_dir = round_dir / "train"
    train_section = trainer.prepare_train_cfg(
        base_train_cfg=base_train_cfg,
        aggregated_dataset_path=aggregated_root,
        repo_id=aggregated_repo_id,
        checkpoint_in=checkpoint_path,
        output_dir=train_output_dir,
        round_id=round_idx,
        resolved_steps=train_steps,
        seed_repo_id=seed_repo_id,
        seed_root=seed_root,
    )
    logging.info(
        "[round %03d] train output: %s steps=%s",
        round_idx,
        train_output_dir,
        train_section.get("steps"),
    )
    return Path(trainer.train(train_section))


def run_dagger_rounds(config: DAggerRoundsConfig | dict[str, Any]) -> dict[str, Any]:
    from scripts.core.dagger_backends import make_policy_backend
    from scripts.core.run_dagger_export import assert_dataset_roots_schema_compatible
    from scripts.core.run_record import RecordConfig, run_record

    if isinstance(config, dict):
        config = DAggerRoundsConfig(**config)
    round_cfg = copy.deepcopy(config.__dict__)
    policy_backend = make_policy_backend(config.policy_backend)
    policy_backend_info = policy_backend.metadata()

    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    seed_repo_id, seed_root = _resolve_seed(config)
    repo_prefix = seed_repo_id.replace("/", "_")
    base_record_cfg = _load_yaml(config.record_cfg_path)["record"]
    train_cfg_path = config.train_cfg_path or policy_backend.trainer.train_cfg_path
    if train_cfg_path is None:
        raise ValueError("dagger_rounds.train_cfg_path or policy_backend.trainer.train_cfg_path is required.")
    base_train_cfg = _load_yaml(train_cfg_path)["train"]
    round_schedule_cfg = _round_schedule_cfg(config)
    max_rounds = _resolve_max_rounds(config, round_schedule_cfg)
    base_train_steps = int(base_train_cfg.get("steps", 10_000))
    train_steps_cfg = _train_steps_cfg(round_schedule_cfg)
    train_steps_mode = str(train_steps_cfg.get("mode", "fixed"))
    warm_start_from_previous = bool(train_steps_cfg.get("warm_start_from_previous", True))

    current_checkpoint = Path(config.initial_pretrained_path) if config.initial_pretrained_path else None
    previous_aggregated_repo_id = seed_repo_id
    previous_aggregated_root = seed_root
    rounds: list[dict[str, Any]] = []
    stop_state: dict[str, Any] = {
        "stopped_early": False,
        "stop_round": None,
        "reason": None,
    }

    if current_checkpoint is None:
        if not config.train_round0_if_missing:
            raise ValueError("initial_pretrained_path is required when train_round0_if_missing=False.")
        round0_dir = output_root / "round_000"
        _clean_or_fail(round0_dir, config.overwrite)
        round0_dir.mkdir(parents=True, exist_ok=True)
        logging.info("[round 000] training initial %s policy from seed dataset: %s", policy_backend.backend_type, seed_root)
        current_checkpoint = _train_round(
            trainer=policy_backend.trainer,
            base_train_cfg=base_train_cfg,
            round_idx=0,
            round_dir=round0_dir,
            checkpoint_path=None,
            aggregated_repo_id=seed_repo_id,
            aggregated_root=seed_root,
            seed_repo_id=seed_repo_id,
            seed_root=seed_root,
        )
        round0_state = {
            "round_id": 0,
            "round": 0,
            "mode": "seed_train",
            "raw_dataset_path": None,
            "exported_dataset_path": None,
            "aggregated_dataset_path": str(seed_root),
            "seed_dataset": {"repo_id": seed_repo_id, "root": str(seed_root)},
            "train_output": str(round0_dir / "train"),
            "train_output_dir": str(round0_dir / "train"),
            "checkpoint": str(current_checkpoint),
            "selected_checkpoint_path": str(current_checkpoint),
            "policy_backend": policy_backend_info,
        }
        _write_json(round0_dir / "round_state.json", round0_state)
        rounds.append(round0_state)
    else:
        current_checkpoint = _require_checkpoint_exists(current_checkpoint, "initial")
        logging.info("[round 000] reuse initial checkpoint: %s", current_checkpoint)

    for round_idx in range(1, max_rounds + 1):
        round_dir = output_root / f"round_{round_idx:03d}"
        _clean_or_fail(round_dir, config.overwrite)
        round_dir.mkdir(parents=True, exist_ok=True)

        logging.info("========== [DAgger round %03d] ==========", round_idx)
        previous_state_path = output_root / f"round_{round_idx - 1:03d}" / "round_state.json"
        if previous_state_path.is_file():
            previous_state = _load_json(previous_state_path)
            previous_checkpoint = Path(previous_state["selected_checkpoint_path"])
            _require_checkpoint_exists(previous_checkpoint, f"round {round_idx - 1:03d} selected")
            if previous_checkpoint != current_checkpoint:
                raise RuntimeError(
                    f"Checkpoint handoff mismatch before round {round_idx:03d}: "
                    f"state file has {previous_checkpoint}, controller has {current_checkpoint}"
                )
            logging.info(
                "[round %03d] previous selected checkpoint from metadata: %s",
                round_idx,
                previous_checkpoint,
            )
        current_checkpoint = _require_checkpoint_exists(current_checkpoint, f"round {round_idx:03d} input")
        logging.info("[round %03d] checkpoint: %s", round_idx, current_checkpoint)

        record_section = _make_record_section(
            base_record_cfg=base_record_cfg,
            round_idx=round_idx,
            round_dir=round_dir,
            checkpoint_path=current_checkpoint,
            episodes_per_round=int(config.episodes_per_round),
            repo_prefix=repo_prefix,
            policy_runner=policy_backend.runner,
            round_cfg=round_cfg,
        )
        record_result = run_record(RecordConfig(record_section))
        raw_repo_id = record_result["dataset_name"]
        raw_root = Path(record_result["dataset_root"])
        logging.info("[round %03d] raw run_mix logs: %s", round_idx, raw_root)

        exported_root = round_dir / "exported_dagger_dataset"
        exported_repo_id = _safe_repo_id(repo_prefix, round_idx, "dagger_export")
        export_section = policy_backend.export_profile.prepare_export_cfg(
            {
                "raw_repo_id": raw_repo_id,
                "raw_root": raw_root,
                "output_repo_id": exported_repo_id,
                "output_root": exported_root,
                "seed_repo_id": seed_repo_id,
                "seed_root": seed_root,
                "min_segment_frames": int(config.export_min_segment_frames),
                "overwrite": config.overwrite,
            },
            base_train_cfg,
            round_cfg,
        )
        export_summary = policy_backend.export_profile.export(export_section)
        logging.info("[round %03d] exported dagger dataset: %s", round_idx, exported_root)

        run_mix_stats = record_result.get("run_mix_stats", {})
        export_metrics = _export_metrics(export_summary, run_mix_stats)
        exported_frames = export_metrics["exported_frames"]
        exported_episodes = export_metrics["exported_episodes"]
        expert_frame_ratio = export_metrics["expert_frame_ratio"]
        intervention_count = export_metrics["intervention_count"]
        logging.info(
            "[round %03d] exported frames=%d, episodes=%d, expert_ratio=%.4f, interventions=%d",
            round_idx,
            exported_frames,
            exported_episodes,
            expert_frame_ratio,
            intervention_count,
        )

        resolved_train_steps = resolve_train_steps(
            round_id=round_idx,
            exported_summary=export_summary,
            base_train_steps=base_train_steps,
            schedule_cfg=round_schedule_cfg,
        )
        logging.info(
            "[round %03d] train steps resolved: base=%d, exported_frames=%d, "
            "exported_episodes=%d, mode=%s, resolved=%d",
            round_idx,
            base_train_steps,
            exported_frames,
            exported_episodes,
            train_steps_mode,
            resolved_train_steps,
        )
        logging.info("[round %03d] resolved train steps=%d", round_idx, resolved_train_steps)

        skipped_train = exported_frames <= 0 or exported_episodes <= 0
        skipped_train_reason = None
        train_checkpoint_in = None
        if skipped_train:
            skipped_train_reason = (
                f"empty exported DAgger dataset: exported_frames={exported_frames}, "
                f"exported_episodes={exported_episodes}"
            )
            logging.warning(
                "[round %03d] %s; skipping aggregate/train and keeping previous checkpoint.",
                round_idx,
                skipped_train_reason,
            )
            aggregated_repo_id = previous_aggregated_repo_id
            aggregated_root = previous_aggregated_root
            train_checkpoint = current_checkpoint
        else:
            from lerobot.datasets.aggregate import aggregate_datasets

            if resolved_train_steps <= 0:
                raise ValueError(
                    f"[round {round_idx:03d}] resolved_train_steps must be > 0 when export is non-empty; "
                    f"got {resolved_train_steps}."
                )
            aggregated_root = round_dir / "aggregated_dataset"
            aggregated_repo_id = _safe_repo_id(repo_prefix, round_idx, "aggregated")
            _clean_or_fail(aggregated_root, config.overwrite)
            logging.info(
                "[round %03d] aggregate previous=%s current=%s -> %s",
                round_idx,
                previous_aggregated_root,
                exported_root,
                aggregated_root,
            )
            assert_dataset_roots_schema_compatible(
                previous_aggregated_repo_id,
                previous_aggregated_root,
                exported_repo_id,
                exported_root,
            )
            aggregate_datasets(
                repo_ids=[previous_aggregated_repo_id, exported_repo_id],
                roots=[previous_aggregated_root, exported_root],
                aggr_repo_id=aggregated_repo_id,
                aggr_root=aggregated_root,
            )
            assert_dataset_roots_schema_compatible(
                seed_repo_id,
                seed_root,
                aggregated_repo_id,
                aggregated_root,
            )

            logging.info("[round %03d] aggregated dataset: %s", round_idx, aggregated_root)
            assert_dataset_roots_schema_compatible(seed_repo_id, seed_root, aggregated_repo_id, aggregated_root)
            train_checkpoint_in = current_checkpoint if warm_start_from_previous else None
            train_checkpoint = _train_round(
                trainer=policy_backend.trainer,
                base_train_cfg=base_train_cfg,
                round_idx=round_idx,
                round_dir=round_dir,
                checkpoint_path=train_checkpoint_in,
                aggregated_repo_id=aggregated_repo_id,
                aggregated_root=aggregated_root,
                seed_repo_id=seed_repo_id,
                seed_root=seed_root,
                train_steps=resolved_train_steps,
            )
            logging.info("[round %03d] next checkpoint: %s", round_idx, train_checkpoint)

        round_state = {
            "round_id": round_idx,
            "round": round_idx,
            "checkpoint_in": str(current_checkpoint),
            "checkpoint_in_path": str(current_checkpoint),
            "train_checkpoint_in": str(train_checkpoint_in) if not skipped_train and train_checkpoint_in else None,
            "raw_dataset_path": str(raw_root),
            "exported_dataset_path": str(exported_root),
            "aggregated_dataset_path": str(aggregated_root),
            "train_output_dir": str(round_dir / "train"),
            "selected_checkpoint_path": str(train_checkpoint),
            "exported_frames": exported_frames,
            "exported_episodes": exported_episodes,
            "expert_frame_ratio": expert_frame_ratio,
            "intervention_count": intervention_count,
            "skipped_short_segments": export_metrics["skipped_short_segments"],
            "incomplete_expert_label_frames": export_metrics["incomplete_expert_label_frames"],
            "base_train_steps": base_train_steps,
            "resolved_train_steps": resolved_train_steps,
            "train_steps_mode": train_steps_mode,
            "skipped_train": skipped_train,
            "skipped_train_reason": skipped_train_reason,
            "early_stop_triggered": False,
            "early_stop_reason": None,
            "raw_run_mix": {"repo_id": raw_repo_id, "root": str(raw_root)},
            "exported_dagger": {
                "repo_id": exported_repo_id,
                "root": str(exported_root),
                "summary": export_summary,
            },
            "aggregated_dataset": {
                "repo_id": aggregated_repo_id,
                "root": str(aggregated_root),
            },
            "train_output": str(round_dir / "train"),
            "checkpoint_out": str(train_checkpoint),
            "policy_backend": policy_backend_info,
            "run_mix_stats": run_mix_stats,
            "schedule": {
                "resolved_train_steps": resolved_train_steps,
                "base_train_steps": base_train_steps,
                "train_steps_mode": train_steps_mode,
                "warm_start_from_previous": warm_start_from_previous,
                "early_stop_checked": True,
                "early_stop_triggered": False,
                "early_stop_reason": None,
            },
        }
        early_stop_triggered, early_stop_reason = should_stop_dagger(rounds, round_state, round_schedule_cfg)
        logging.info(
            "[round %03d] early stop check: triggered=%s, reason=%s",
            round_idx,
            early_stop_triggered,
            early_stop_reason,
        )
        round_state["early_stop_triggered"] = early_stop_triggered
        round_state["early_stop_reason"] = early_stop_reason
        round_state["schedule"]["early_stop_triggered"] = early_stop_triggered
        round_state["schedule"]["early_stop_reason"] = early_stop_reason
        _write_json(round_dir / "round_state.json", round_state)
        rounds.append(round_state)

        current_checkpoint = train_checkpoint
        previous_aggregated_repo_id = aggregated_repo_id
        previous_aggregated_root = aggregated_root

        if early_stop_triggered:
            stop_state = {
                "stopped_early": True,
                "stop_round": round_idx,
                "reason": early_stop_reason,
            }
            break

    final_state = {
        "output_root": str(output_root),
        "policy_backend": policy_backend_info,
        "seed_dataset": {"repo_id": seed_repo_id, "root": str(seed_root)},
        "final_checkpoint": str(current_checkpoint),
        "final_aggregated_dataset": {
            "repo_id": previous_aggregated_repo_id,
            "root": str(previous_aggregated_root),
        },
        "total_rounds_completed": sum(1 for state in rounds if int(state.get("round_id", 0) or 0) > 0),
        "stopped_early": bool(stop_state["stopped_early"]),
        "stop_reason": stop_state["reason"],
        "stop": stop_state,
        "round_schedule": round_schedule_cfg,
        "round_schedules": [
            {
                "round": state.get("round_id"),
                "exported_frames": state.get("exported_frames"),
                "exported_episodes": state.get("exported_episodes"),
                "resolved_train_steps": state.get("resolved_train_steps"),
                "base_train_steps": state.get("base_train_steps"),
                "train_steps_mode": state.get("train_steps_mode"),
                "skipped_train": state.get("skipped_train"),
                "early_stop_triggered": state.get("early_stop_triggered"),
                "early_stop_reason": state.get("early_stop_reason"),
            }
            for state in rounds
            if int(state.get("round_id", 0) or 0) > 0
        ],
        "rounds": rounds,
    }
    _write_json(output_root / "dagger_rounds_state.json", final_state)
    return final_state


def main() -> None:
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "dagger_rounds_cfg.yaml"
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    cfg = _load_yaml(cfg_path)
    rounds_cfg = cfg.get("dagger_rounds", cfg)
    run_dagger_rounds(rounds_cfg)


if __name__ == "__main__":
    main()
