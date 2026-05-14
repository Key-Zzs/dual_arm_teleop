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

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.utils.constants import HF_LEROBOT_HOME

from scripts.core.run_dagger_export import assert_dataset_roots_schema_compatible, export_dagger_dataset
from scripts.core.run_record import RecordConfig, run_record
from scripts.core.run_train import run_act_dagger_from_train_cfg


@dataclass
class DAggerRoundsConfig:
    output_root: str | Path
    record_cfg_path: str | Path
    train_cfg_path: str | Path
    num_rounds: int
    episodes_per_round: int
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
    seed_root = cfg.seed_dataset_path or cfg.seed_root
    if seed_root is not None:
        root = Path(seed_root)
    elif cfg.seed_repo_id is not None:
        root = HF_LEROBOT_HOME / cfg.seed_repo_id
    else:
        raise ValueError("One of seed_dataset_path, seed_root, or seed_repo_id is required.")

    repo_id = cfg.seed_repo_id or root.name
    return repo_id, root


def _resolve_latest_pretrained(train_output_dir: Path) -> Path:
    # Default checkpoint selection is intentionally conservative: use the
    # training loop's `last` symlink. A future best-checkpoint hook can be added
    # here once an evaluation/selection signal exists for real-robot ACT.
    last = train_output_dir / "checkpoints" / "last" / "pretrained_model"
    if last.is_dir():
        return last

    raise FileNotFoundError(
        f"No selected training checkpoint found: {last}. "
        "Expected train output to contain checkpoints/last/pretrained_model."
    )


def _require_checkpoint_exists(path: Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} checkpoint path is required.")
    if not path.is_dir():
        raise FileNotFoundError(f"{label} checkpoint path does not exist or is not a directory: {path}")
    missing = [name for name in ("config.json", "model.safetensors") if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{label} checkpoint is missing required file(s) {missing}: {path}")
    return path


def _policy_chunk_size(train_cfg: dict[str, Any]) -> int:
    return int(train_cfg.get("policy", {}).get("chunk_size", 1))


def _make_record_section(
    base_record_cfg: dict[str, Any],
    round_idx: int,
    round_dir: Path,
    checkpoint_path: Path,
    episodes_per_round: int,
    repo_prefix: str,
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
    record_section.setdefault("policy", {})
    if record_section["policy"].get("type") == "act_dagger":
        record_section["policy"]["type"] = "act"
    if record_section["policy"].get("type") != "act":
        raise ValueError("run_mix DAgger rounds require record.policy.type to be ACT-compatible.")
    record_section["policy"]["pretrained_path"] = str(checkpoint_path)

    return record_section


def _make_train_section(
    base_train_cfg: dict[str, Any],
    round_idx: int,
    train_output_dir: Path,
    checkpoint_path: Path | None,
    aggregated_repo_id: str,
    aggregated_root: Path,
    seed_repo_id: str,
    seed_root: Path,
) -> dict[str, Any]:
    train_section = copy.deepcopy(base_train_cfg)
    train_section["output_dir"] = str(train_output_dir)
    train_section["job_name"] = f"act_dagger_round_{round_idx:03d}"
    train_section["resume"] = False
    train_section.setdefault("dataset", {})
    train_section["dataset"]["repo_id"] = aggregated_repo_id
    train_section["dataset"]["root"] = str(aggregated_root)

    train_section.setdefault("policy", {})
    train_section["policy"]["type"] = "act_dagger"
    if checkpoint_path is None:
        train_section["policy"].pop("pretrained_path", None)
    else:
        train_section["policy"]["pretrained_path"] = str(checkpoint_path)

    dagger_section = train_section.setdefault("dagger", {})
    dagger_dataset = dagger_section.setdefault("dataset", {})
    dagger_dataset.update(
        {
            "aggregated_repo_id": aggregated_repo_id,
            "aggregated_root": str(aggregated_root),
            "seed_repo_id": seed_repo_id,
            "seed_root": str(seed_root),
            "resume_aggregation": True,
            "copy_seed_if_missing": False,
        }
    )

    dagger_training = dagger_section.setdefault("training", {})
    dagger_training["rounds"] = 1
    dagger_training.setdefault("steps_per_round", train_section.get("steps", 10_000))
    dagger_training.setdefault("batch_size", train_section.get("batch_size", 8))
    dagger_training.setdefault("num_workers", train_section.get("num_workers", 4))
    dagger_training.setdefault("log_freq", train_section.get("log_freq", 100))
    dagger_training["save_checkpoint"] = True
    train_section["save_checkpoint"] = True

    return train_section


def _train_round(
    base_train_cfg: dict[str, Any],
    round_idx: int,
    round_dir: Path,
    checkpoint_path: Path | None,
    aggregated_repo_id: str,
    aggregated_root: Path,
    seed_repo_id: str,
    seed_root: Path,
) -> Path:
    train_output_dir = round_dir / "train"
    train_section = _make_train_section(
        base_train_cfg=base_train_cfg,
        round_idx=round_idx,
        train_output_dir=train_output_dir,
        checkpoint_path=checkpoint_path,
        aggregated_repo_id=aggregated_repo_id,
        aggregated_root=aggregated_root,
        seed_repo_id=seed_repo_id,
        seed_root=seed_root,
    )
    logging.info("[round %03d] train output: %s", round_idx, train_output_dir)
    run_act_dagger_from_train_cfg(train_section)
    return _resolve_latest_pretrained(train_output_dir)


def run_dagger_rounds(config: DAggerRoundsConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        config = DAggerRoundsConfig(**config)

    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    seed_repo_id, seed_root = _resolve_seed(config)
    repo_prefix = seed_repo_id.replace("/", "_")
    base_record_cfg = _load_yaml(config.record_cfg_path)["record"]
    base_train_cfg = _load_yaml(config.train_cfg_path)["train"]
    act_chunk_size = _policy_chunk_size(base_train_cfg)
    effective_min_episode_len = int(config.min_episode_len_for_act or act_chunk_size)

    current_checkpoint = Path(config.initial_pretrained_path) if config.initial_pretrained_path else None
    previous_aggregated_repo_id = seed_repo_id
    previous_aggregated_root = seed_root
    rounds: list[dict[str, Any]] = []

    if current_checkpoint is None:
        if not config.train_round0_if_missing:
            raise ValueError("initial_pretrained_path is required when train_round0_if_missing=False.")
        round0_dir = output_root / "round_000"
        _clean_or_fail(round0_dir, config.overwrite)
        round0_dir.mkdir(parents=True, exist_ok=True)
        logging.info("[round 000] training initial ACT from seed dataset: %s", seed_root)
        current_checkpoint = _train_round(
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
        }
        _write_json(round0_dir / "round_state.json", round0_state)
        rounds.append(round0_state)
    else:
        current_checkpoint = _require_checkpoint_exists(current_checkpoint, "initial")
        logging.info("[round 000] reuse initial checkpoint: %s", current_checkpoint)

    for round_idx in range(1, int(config.num_rounds) + 1):
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
        )
        record_result = run_record(RecordConfig(record_section))
        raw_repo_id = record_result["dataset_name"]
        raw_root = Path(record_result["dataset_root"])
        logging.info("[round %03d] raw run_mix logs: %s", round_idx, raw_root)

        exported_root = round_dir / "exported_dagger_dataset"
        exported_repo_id = _safe_repo_id(repo_prefix, round_idx, "dagger_export")
        export_summary = export_dagger_dataset(
            raw_repo_id=raw_repo_id,
            raw_root=raw_root,
            output_repo_id=exported_repo_id,
            output_root=exported_root,
            seed_repo_id=seed_repo_id,
            seed_root=seed_root,
            min_segment_frames=int(config.export_min_segment_frames),
            min_episode_len_for_act=effective_min_episode_len,
            pre_takeover_context=int(config.pre_takeover_context),
            require_complete_expert_action=bool(config.require_complete_expert_action),
            overwrite=config.overwrite,
        )
        logging.info("[round %03d] exported dagger dataset: %s", round_idx, exported_root)

        exported_frames = int(export_summary["exported_dataset"]["total_frames"])
        if exported_frames > 0:
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
        else:
            logging.warning(
                "[round %03d] no exported DAgger frames; reusing previous aggregated dataset.",
                round_idx,
            )
            aggregated_repo_id = previous_aggregated_repo_id
            aggregated_root = previous_aggregated_root

        logging.info("[round %03d] aggregated dataset: %s", round_idx, aggregated_root)
        assert_dataset_roots_schema_compatible(seed_repo_id, seed_root, aggregated_repo_id, aggregated_root)
        train_checkpoint = _train_round(
            base_train_cfg=base_train_cfg,
            round_idx=round_idx,
            round_dir=round_dir,
            checkpoint_path=current_checkpoint,
            aggregated_repo_id=aggregated_repo_id,
            aggregated_root=aggregated_root,
            seed_repo_id=seed_repo_id,
            seed_root=seed_root,
        )
        logging.info("[round %03d] next checkpoint: %s", round_idx, train_checkpoint)

        round_state = {
            "round_id": round_idx,
            "round": round_idx,
            "checkpoint_in": str(current_checkpoint),
            "checkpoint_in_path": str(current_checkpoint),
            "raw_dataset_path": str(raw_root),
            "exported_dataset_path": str(exported_root),
            "aggregated_dataset_path": str(aggregated_root),
            "train_output_dir": str(round_dir / "train"),
            "selected_checkpoint_path": str(train_checkpoint),
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
            "run_mix_stats": record_result.get("run_mix_stats", {}),
        }
        _write_json(round_dir / "round_state.json", round_state)
        rounds.append(round_state)

        current_checkpoint = train_checkpoint
        previous_aggregated_repo_id = aggregated_repo_id
        previous_aggregated_root = aggregated_root

    final_state = {
        "output_root": str(output_root),
        "seed_dataset": {"repo_id": seed_repo_id, "root": str(seed_root)},
        "final_checkpoint": str(current_checkpoint),
        "final_aggregated_dataset": {
            "repo_id": previous_aggregated_repo_id,
            "root": str(previous_aggregated_root),
        },
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
