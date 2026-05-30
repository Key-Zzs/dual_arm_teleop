from __future__ import annotations

import copy
import csv
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from lerobot.datasets.utils import build_dataset_frame
from lerobot.utils.constants import OBS_STR


OBS_STATE_KEY = f"{OBS_STR}.state"
PHASE_RIGHT_ARM = "right_arm"
PHASE_LEFT_ARM = "left_arm"
PHASE_NAMES = ("phase_right_arm", "phase_left_arm")
VALID_PHASES = {PHASE_RIGHT_ARM, PHASE_LEFT_ARM}
PHASE_MODE_TWO_PHASE_ACTIVE_ARM = "two_phase_active_arm"
VALID_PHASE_MODES = {PHASE_MODE_TWO_PHASE_ACTIVE_ARM}
STABLE_SOURCE_MEASURED_EE_POSE = "measured_ee_pose"
STABLE_SOURCE_SENT_ACTION = "sent_action"
STABLE_SOURCE_BOTH = "both"
VALID_STABLE_SOURCES = {
    STABLE_SOURCE_MEASURED_EE_POSE,
    STABLE_SOURCE_SENT_ACTION,
    STABLE_SOURCE_BOTH,
}
PHASE_LOG_EVERY_FRAMES = 30
DEFAULT_PHASE_CONDITIONING_CONFIG = "phase_conditioning_cfg.yaml"


def default_phase_conditioning_config_path(scripts_dir: Path) -> Path:
    return Path(scripts_dir) / "config" / DEFAULT_PHASE_CONDITIONING_CONFIG


def _cfg_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "0", "no", "n", "off", "none", "null", ""}:
            return False
    return bool(value)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip_gripper_cmd(value: float) -> float:
    return min(1.0, max(0.0, value))


def _flatten_feature_names(names: Any) -> list[str]:
    if names is None:
        return []
    if isinstance(names, str):
        return [names]
    if isinstance(names, dict):
        flattened: list[str] = []
        for value in names.values():
            flattened.extend(_flatten_feature_names(value))
        return flattened
    if isinstance(names, (list, tuple)):
        flattened = []
        for value in names:
            flattened.extend(_flatten_feature_names(value))
        return flattened
    return [str(names)]


def _nested_mapping_value(source: Any, dotted_key: str) -> Any:
    if not isinstance(source, dict) or "." not in dotted_key:
        return None
    current: Any = source
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _source_float(source: dict[str, Any] | None, key: str) -> float | None:
    if source is None:
        return None
    value = source.get(key)
    if value is None:
        value = _nested_mapping_value(source, key)
    return _float_or_none(value)


@dataclass
class PhaseSwitchRightToLeftConfig:
    enabled: bool = True
    require_right_gripper_open: bool = True
    right_gripper_open_threshold: float = 0.8
    require_right_gripper_closed_once: bool = False
    right_gripper_closed_threshold: float = 0.2
    require_right_arm_stable: bool = True
    stable_source: str = STABLE_SOURCE_MEASURED_EE_POSE
    right_delta_trans_threshold: float = 0.001
    right_delta_rot_threshold: float = 0.005
    right_ee_speed_threshold: float = 0.01
    stable_window_frames: int = 10
    dwell_frames: int = 15
    min_phase_frames: int = 30

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any] | None) -> "PhaseSwitchRightToLeftConfig":
        cfg = cfg or {}
        if not isinstance(cfg, dict):
            raise ValueError("`phase_conditioning.switch_right_to_left` must be a mapping.")

        stable_source = str(
            cfg.get("stable_source", STABLE_SOURCE_MEASURED_EE_POSE)
        ).strip().lower()
        if stable_source not in VALID_STABLE_SOURCES:
            raise ValueError(
                "`phase_conditioning.switch_right_to_left.stable_source` "
                f"must be one of {sorted(VALID_STABLE_SOURCES)}. Got: {stable_source!r}"
            )

        stable_window_frames = max(1, int(cfg.get("stable_window_frames", 10)))
        dwell_frames = max(1, int(cfg.get("dwell_frames", 15)))
        min_phase_frames = max(0, int(cfg.get("min_phase_frames", 30)))
        return cls(
            enabled=_cfg_bool(cfg.get("enabled", True), True),
            require_right_gripper_open=_cfg_bool(
                cfg.get("require_right_gripper_open", True), True
            ),
            right_gripper_open_threshold=float(cfg.get("right_gripper_open_threshold", 0.8)),
            require_right_gripper_closed_once=_cfg_bool(
                cfg.get("require_right_gripper_closed_once", False), False
            ),
            right_gripper_closed_threshold=float(cfg.get("right_gripper_closed_threshold", 0.2)),
            require_right_arm_stable=_cfg_bool(cfg.get("require_right_arm_stable", True), True),
            stable_source=stable_source,
            right_delta_trans_threshold=float(cfg.get("right_delta_trans_threshold", 0.001)),
            right_delta_rot_threshold=float(cfg.get("right_delta_rot_threshold", 0.005)),
            right_ee_speed_threshold=float(cfg.get("right_ee_speed_threshold", 0.01)),
            stable_window_frames=stable_window_frames,
            dwell_frames=dwell_frames,
            min_phase_frames=min_phase_frames,
        )


@dataclass
class PhaseActionGateConfig:
    """Optional deployment guard that zeros inactive-arm policy deltas.

    Phase conditioning should ideally make the policy output near-zero actions for
    the inactive arm by itself. This gate is a runtime safety guard for deployment:
    it is disabled by default, and when enabled it only changes executed policy
    deltas, not the policy network or training.
    """

    enabled: bool = False
    zero_inactive_arm_delta_pose: bool = True
    zero_inactive_arm_gripper: bool = False
    log_action_norms: bool = True

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any] | None) -> "PhaseActionGateConfig":
        cfg = cfg or {}
        if not isinstance(cfg, dict):
            raise ValueError("`phase_conditioning.active_arm_action_gate` must be a mapping.")
        return cls(
            enabled=_cfg_bool(cfg.get("enabled", False), False),
            zero_inactive_arm_delta_pose=_cfg_bool(
                cfg.get("zero_inactive_arm_delta_pose", True), True
            ),
            zero_inactive_arm_gripper=_cfg_bool(
                cfg.get("zero_inactive_arm_gripper", False), False
            ),
            log_action_norms=_cfg_bool(cfg.get("log_action_norms", True), True),
        )


@dataclass
class PhaseConditioningConfig:
    enabled: bool = False
    mode: str = PHASE_MODE_TWO_PHASE_ACTIVE_ARM
    initial_phase: str = PHASE_RIGHT_ARM
    switch_right_to_left: PhaseSwitchRightToLeftConfig = field(
        default_factory=PhaseSwitchRightToLeftConfig
    )
    reset_policy_on_switch: bool = True
    log_phase_state: bool = True
    debug_csv: Path | None = None
    log_every_frames: int = PHASE_LOG_EVERY_FRAMES
    active_arm_action_gate: PhaseActionGateConfig = field(default_factory=PhaseActionGateConfig)

    @classmethod
    def from_dict(
        cls,
        cfg: Dict[str, Any] | None,
        *,
        project_root: Path,
    ) -> "PhaseConditioningConfig":
        cfg = cfg or {}
        if not isinstance(cfg, dict):
            raise ValueError("`phase_conditioning` must be a mapping.")

        mode = str(cfg.get("mode", PHASE_MODE_TWO_PHASE_ACTIVE_ARM)).strip().lower()
        if mode not in VALID_PHASE_MODES:
            raise ValueError(
                "`phase_conditioning.mode` must be "
                f"{PHASE_MODE_TWO_PHASE_ACTIVE_ARM!r}. Got: {mode!r}"
            )

        initial_phase = str(cfg.get("initial_phase", PHASE_RIGHT_ARM)).strip().lower()
        if initial_phase not in VALID_PHASES:
            raise ValueError(
                "`phase_conditioning.initial_phase` must be one of "
                f"{sorted(VALID_PHASES)}. Got: {initial_phase!r}"
            )

        debug_csv = cfg.get("debug_csv")
        debug_csv_path = None
        if debug_csv not in (None, "", "null", "None"):
            debug_csv_path = Path(str(debug_csv)).expanduser()
            if not debug_csv_path.is_absolute():
                debug_csv_path = project_root / debug_csv_path

        return cls(
            enabled=_cfg_bool(cfg.get("enabled", False), False),
            mode=mode,
            initial_phase=initial_phase,
            switch_right_to_left=PhaseSwitchRightToLeftConfig.from_dict(
                cfg.get("switch_right_to_left", {})
            ),
            reset_policy_on_switch=_cfg_bool(cfg.get("reset_policy_on_switch", True), True),
            log_phase_state=_cfg_bool(cfg.get("log_phase_state", True), True),
            debug_csv=debug_csv_path,
            log_every_frames=max(1, int(cfg.get("log_every_frames", PHASE_LOG_EVERY_FRAMES))),
            active_arm_action_gate=PhaseActionGateConfig.from_dict(
                cfg.get("active_arm_action_gate", {})
            ),
        )


def load_phase_conditioning_config(
    *,
    scripts_dir: Path,
    project_root: Path,
    config_path: Path | None = None,
) -> PhaseConditioningConfig:
    path = Path(config_path) if config_path is not None else default_phase_conditioning_config_path(scripts_dir)
    if not path.is_file():
        return PhaseConditioningConfig()

    with open(path, "r") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Phase conditioning config must be a mapping: {path}")
    cfg = loaded.get("phase_conditioning", loaded)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"`phase_conditioning` section must be a mapping: {path}")
    return PhaseConditioningConfig.from_dict(cfg, project_root=project_root)


@dataclass
class PhaseUpdate:
    frame_idx: int
    timestamp_s: float
    current_phase: str
    phase_one_hot: np.ndarray
    right_gripper_value: float | None
    right_gripper_open: bool | None
    right_gripper_closed_seen: bool
    right_ee_speed: float | None
    right_delta_trans_norm: float | None
    right_delta_rot_norm: float | None
    right_arm_stable: bool | None
    done_counter: int
    current_phase_frames: int
    switched_this_frame: bool


def phase_one_hot(phase: str) -> np.ndarray:
    if phase == PHASE_RIGHT_ARM:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    if phase == PHASE_LEFT_ARM:
        return np.asarray([0.0, 1.0], dtype=np.float32)
    raise ValueError(f"Unsupported phase: {phase!r}. Expected one of {sorted(VALID_PHASES)}.")


def _fmt_optional_float(value: float | None, precision: int = 6) -> str:
    if value is None:
        return "None"
    return f"{float(value):.{precision}f}"


def _value_from_state_frame(
    observation_frame: dict[str, Any] | None,
    state_names: list[str],
    candidates: tuple[str, ...],
) -> float | None:
    if not observation_frame or OBS_STATE_KEY not in observation_frame:
        return None
    state = np.asarray(observation_frame[OBS_STATE_KEY], dtype=np.float32).reshape(-1)
    for candidate in candidates:
        if candidate not in state_names:
            continue
        idx = state_names.index(candidate)
        if idx >= state.shape[0]:
            continue
        return _float_or_none(state[idx])
    return None


def _candidate_gripper_keys(arm: str, gripper_keys: dict[str, str]) -> list[str]:
    candidates = {
        "left": ("left_gripper_cmd", "left_gripper_cmd_bin"),
        "right": ("right_gripper_cmd", "right_gripper_cmd_bin"),
    }
    preferred = gripper_keys.get(arm)
    keys = [preferred] if preferred else []
    keys.extend(candidates.get(arm, ()))
    seen: set[str] = set()
    result: list[str] = []
    for key in keys:
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _gripper_command_value(
    arm: str,
    source: dict[str, Any] | None,
    gripper_keys: dict[str, str],
) -> float | None:
    if source is None:
        return None
    for key in _candidate_gripper_keys(arm, gripper_keys):
        value = _float_or_none(source.get(key))
        if value is not None:
            return _clip_gripper_cmd(value)
    return None


def _right_gripper_value_from_sources(
    *,
    raw_obs: dict[str, Any],
    obs_processed: dict[str, Any],
    observation_frame: dict[str, Any] | None,
    state_names: list[str],
    last_sent_action: dict[str, Any] | None,
    gripper_keys: dict[str, str],
    right_gripper_max_open: float | None,
) -> float | None:
    normalized_candidates = (
        "right_gripper_state_norm",
        "right_gripper_open_fraction",
        "right_gripper_open",
        "right_gripper_position",
        "right_gripper_cmd",
        "right_gripper_cmd_bin",
    )
    width_candidates = (
        "right_gripper_width_norm",
        "right_gripper_state.width_norm",
        "right_gripper.width_norm",
    )
    raw_width_candidates = (
        "right_gripper_width",
        "right_gripper_state.width",
        "right_gripper.width",
    )

    for source in (raw_obs, obs_processed):
        for key in normalized_candidates + width_candidates:
            value = _source_float(source, key)
            if value is not None:
                return _clip_gripper_cmd(value)
        for key in raw_width_candidates:
            value = _source_float(source, key)
            if value is None:
                continue
            if right_gripper_max_open is not None and right_gripper_max_open > 0:
                return _clip_gripper_cmd(value / float(right_gripper_max_open))
            return _clip_gripper_cmd(value)

    value = _value_from_state_frame(observation_frame, state_names, normalized_candidates)
    if value is not None:
        return _clip_gripper_cmd(value)

    return _gripper_command_value("right", last_sent_action, gripper_keys)


def _right_ee_xyz_from_source(source: dict[str, Any] | None) -> np.ndarray | None:
    values = [_source_float(source, f"right_ee_pose.{axis}") for axis in ("x", "y", "z")]
    if any(value is None for value in values):
        return None
    xyz = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(xyz)):
        return None
    return xyz


def _right_delta_norms_from_action(action: dict[str, Any] | None) -> tuple[float | None, float | None]:
    if action is None:
        return None, None

    trans_values = [_source_float(action, f"right_delta_ee_pose.{axis}") for axis in ("x", "y", "z")]
    rot_values = [_source_float(action, f"right_delta_ee_pose.{axis}") for axis in ("rx", "ry", "rz")]
    if any(value is None for value in trans_values + rot_values):
        return None, None

    trans = np.asarray(trans_values, dtype=np.float64)
    rot = np.asarray(rot_values, dtype=np.float64)
    if not np.all(np.isfinite(trans)) or not np.all(np.isfinite(rot)):
        return None, None
    return float(np.linalg.norm(trans)), float(np.linalg.norm(rot))


def _arm_delta_norms_from_action(action: dict[str, Any] | None, arm: str) -> tuple[float | None, float | None]:
    if action is None:
        return None, None

    trans_values = [_source_float(action, f"{arm}_delta_ee_pose.{axis}") for axis in ("x", "y", "z")]
    rot_values = [_source_float(action, f"{arm}_delta_ee_pose.{axis}") for axis in ("rx", "ry", "rz")]
    if any(value is None for value in trans_values + rot_values):
        return None, None

    trans = np.asarray(trans_values, dtype=np.float64)
    rot = np.asarray(rot_values, dtype=np.float64)
    if not np.all(np.isfinite(trans)) or not np.all(np.isfinite(rot)):
        return None, None
    return float(np.linalg.norm(trans)), float(np.linalg.norm(rot))


def _zero_arm_delta_pose(action: dict[str, Any], arm: str) -> None:
    for axis in ("x", "y", "z", "rx", "ry", "rz"):
        key = f"{arm}_delta_ee_pose.{axis}"
        if key in action:
            action[key] = 0.0


def _zero_arm_gripper(action: dict[str, Any], arm: str, gripper_keys: dict[str, str]) -> None:
    key = gripper_keys.get(arm)
    if key is not None and key in action:
        action[key] = 0.0


def _inactive_arm_for_phase(phase: str) -> str | None:
    if phase == PHASE_RIGHT_ARM:
        return "left"
    if phase == PHASE_LEFT_ARM:
        return "right"
    return None


def apply_phase_action_gate(
    action: dict[str, Any],
    *,
    phase_machine: "PhaseStateMachine | None",
    phase_conditioning: PhaseConditioningConfig,
    gripper_keys: dict[str, str],
    frame_idx: int,
    source: str,
) -> dict[str, Any]:
    gate_cfg = phase_conditioning.active_arm_action_gate
    if phase_machine is None or not phase_conditioning.enabled or not gate_cfg.enabled:
        return action

    inactive_arm = _inactive_arm_for_phase(phase_machine.current_phase)
    if inactive_arm is None:
        return action

    gated_action = dict(action)
    before_left_trans, before_left_rot = _arm_delta_norms_from_action(gated_action, "left")
    before_right_trans, before_right_rot = _arm_delta_norms_from_action(gated_action, "right")

    if gate_cfg.zero_inactive_arm_delta_pose:
        _zero_arm_delta_pose(gated_action, inactive_arm)
    if gate_cfg.zero_inactive_arm_gripper:
        _zero_arm_gripper(gated_action, inactive_arm, gripper_keys)

    after_left_trans, after_left_rot = _arm_delta_norms_from_action(gated_action, "left")
    after_right_trans, after_right_rot = _arm_delta_norms_from_action(gated_action, "right")

    inactive_trans = before_left_trans if inactive_arm == "left" else before_right_trans
    inactive_rot = before_left_rot if inactive_arm == "left" else before_right_rot
    should_log = (
        gate_cfg.log_action_norms
        and (
            frame_idx <= 5
            or frame_idx % phase_conditioning.log_every_frames == 0
            or (inactive_trans is not None and inactive_trans > 1e-6)
            or (inactive_rot is not None and inactive_rot > 1e-6)
        )
    )
    if should_log:
        logging.info(
            "[phase_action_gate] frame=%d phase=%s source=%s inactive_arm=%s "
            "left_trans=%s->%s left_rot=%s->%s right_trans=%s->%s right_rot=%s->%s",
            frame_idx,
            phase_machine.current_phase,
            source,
            inactive_arm,
            _fmt_optional_float(before_left_trans),
            _fmt_optional_float(after_left_trans),
            _fmt_optional_float(before_left_rot),
            _fmt_optional_float(after_left_rot),
            _fmt_optional_float(before_right_trans),
            _fmt_optional_float(after_right_trans),
            _fmt_optional_float(before_right_rot),
            _fmt_optional_float(after_right_rot),
        )

    return gated_action


def _state_names_from_features(features: dict[str, Any] | None) -> list[str]:
    if not features or OBS_STATE_KEY not in features:
        return []
    names = features[OBS_STATE_KEY].get("names")
    return [str(name) for name in _flatten_feature_names(names)]


def _phase_observation_shape_error(
    actual_shape: tuple[int, ...],
    expected_shape: tuple[int, ...] | None,
    *,
    enabled: bool,
) -> ValueError:
    return ValueError(
        "phase_conditioning observation.state shape mismatch.\n"
        f"- phase_conditioning.enabled: {enabled}\n"
        f"- current observation.state shape: {actual_shape}\n"
        f"- policy expected observation.state shape: {expected_shape}\n"
        "- If enabled=true, verify that the checkpoint was trained on a dataset "
        "after scripts/debug/annotate_dataset_phase.py appended "
        "phase_right_arm and phase_left_arm.\n"
        "- If the checkpoint expects phase features, verify phase_conditioning_cfg.yaml has "
        "`phase_conditioning.enabled: true`.\n"
        "- If the checkpoint is an old non-phase checkpoint, keep "
        "`phase_conditioning.enabled: false`."
    )


def _policy_expected_state_shape(policy) -> tuple[int, ...] | None:
    feature = getattr(getattr(policy, "config", None), "robot_state_feature", None)
    if feature is None:
        return None
    return tuple(int(dim) for dim in feature.shape)


def _policy_expected_state_dim(policy) -> int | None:
    shape = _policy_expected_state_shape(policy)
    if shape is None or len(shape) != 1:
        return None
    return int(shape[0])


def _validate_observation_state_shape_for_policy(
    observation_frame: dict[str, Any],
    *,
    policy,
    phase_enabled: bool,
) -> None:
    if OBS_STATE_KEY not in observation_frame:
        raise ValueError(
            "phase_conditioning enabled but observation.state is missing from the policy observation."
        )

    expected_shape = _policy_expected_state_shape(policy)
    if expected_shape is None:
        return
    actual_shape = tuple(np.asarray(observation_frame[OBS_STATE_KEY]).shape)
    if actual_shape != expected_shape:
        raise _phase_observation_shape_error(
            actual_shape,
            expected_shape,
            enabled=phase_enabled,
        )


def try_reset_policy_after_phase_switch(policy) -> None:
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()
        logging.info("[phase_conditioning] policy.reset() called after phase switch.")
        return
    logging.warning(
        "[phase_conditioning] phase switched but policy has no reset() method; "
        "continuing without clearing policy queues."
    )


class PhaseStateMachine:
    """Minimal online phase switcher for a single phase-conditioned policy.

    This stage intentionally uses a small engineering condition:
    right_gripper_open + right_arm_stable + dwell time. That only means the right
    arm has stopped and the right gripper is open; it is not proof that the vial
    was successfully inserted. Later stages should replace or augment it with a
    more reliable success signal such as target pose completion, visual insertion
    detection, VLM/rule checks, manual confirmation, or a multi-condition gate.
    If the episode starts with the right gripper already open, this condition can
    fire too early; enable require_right_gripper_closed_once to require a close
    event before treating a later open state as right-arm completion.
    """

    def __init__(
        self,
        cfg: PhaseConditioningConfig,
        *,
        fps: int | float,
        state_names: list[str],
        right_gripper_max_open: float | None = None,
    ):
        if cfg.mode != PHASE_MODE_TWO_PHASE_ACTIVE_ARM:
            raise ValueError(
                "`phase_conditioning.mode` currently only supports "
                f"{PHASE_MODE_TWO_PHASE_ACTIVE_ARM!r}. Got: {cfg.mode!r}"
            )
        if cfg.initial_phase not in VALID_PHASES:
            raise ValueError(
                "`phase_conditioning.initial_phase` must be one of "
                f"{sorted(VALID_PHASES)}. Got: {cfg.initial_phase!r}"
            )

        self.cfg = cfg
        self.switch_cfg = cfg.switch_right_to_left
        self.fps = max(1.0, float(fps))
        self.current_phase = cfg.initial_phase
        self.current_phase_frames = 0
        self.done_counter = 0
        self.right_gripper_closed_seen = False
        self.state_names = list(state_names)
        self.right_gripper_max_open = right_gripper_max_open
        self.right_ee_history: deque[tuple[float, np.ndarray]] = deque(
            maxlen=self.switch_cfg.stable_window_frames
        )
        self._warned: set[str] = set()
        self._csv_file = None
        self._csv_writer: csv.DictWriter | None = None
        if cfg.debug_csv is not None:
            cfg.debug_csv.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(cfg.debug_csv, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=[
                    "frame_idx",
                    "timestamp",
                    "current_phase",
                    "phase_right_arm",
                    "phase_left_arm",
                    "right_gripper_value",
                    "right_gripper_open",
                    "right_gripper_closed_seen",
                    "right_ee_speed",
                    "right_delta_trans_norm",
                    "right_delta_rot_norm",
                    "right_arm_stable",
                    "done_counter",
                    "current_phase_frames",
                    "switched_this_frame",
                ],
            )
            self._csv_writer.writeheader()

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def one_hot(self) -> np.ndarray:
        return phase_one_hot(self.current_phase)

    def update(
        self,
        *,
        frame_idx: int,
        timestamp_s: float,
        raw_obs: dict[str, Any],
        obs_processed: dict[str, Any],
        last_sent_action: dict[str, Any] | None,
        gripper_keys: dict[str, str],
        observation_frame: dict[str, Any] | None = None,
    ) -> PhaseUpdate:
        self.current_phase_frames += 1

        right_xyz = _right_ee_xyz_from_source(obs_processed)
        if right_xyz is None:
            right_xyz = _right_ee_xyz_from_source(raw_obs)
        if right_xyz is not None:
            self.right_ee_history.append((float(timestamp_s), right_xyz))

        right_gripper_value = _right_gripper_value_from_sources(
            raw_obs=raw_obs,
            obs_processed=obs_processed,
            observation_frame=observation_frame,
            state_names=self.state_names,
            last_sent_action=last_sent_action,
            gripper_keys=gripper_keys,
            right_gripper_max_open=self.right_gripper_max_open,
        )
        if (
            right_gripper_value is not None
            and float(right_gripper_value) < self.switch_cfg.right_gripper_closed_threshold
        ):
            self.right_gripper_closed_seen = True
        right_gripper_open = self._right_gripper_open(right_gripper_value)
        right_ee_speed = self._right_ee_speed()
        right_delta_trans_norm, right_delta_rot_norm = _right_delta_norms_from_action(last_sent_action)
        right_arm_stable = self._right_arm_stable(
            right_ee_speed=right_ee_speed,
            right_delta_trans_norm=right_delta_trans_norm,
            right_delta_rot_norm=right_delta_rot_norm,
            sent_action_available=last_sent_action is not None,
        )

        switched = False
        if self.current_phase == PHASE_RIGHT_ARM:
            if self._right_to_left_ready(right_gripper_open, right_arm_stable):
                self.done_counter += 1
            else:
                self.done_counter = 0

            if (
                self.switch_cfg.enabled
                and self.done_counter >= self.switch_cfg.dwell_frames
                and self.current_phase_frames >= self.switch_cfg.min_phase_frames
            ):
                self.current_phase = PHASE_LEFT_ARM
                self.current_phase_frames = 0
                self.done_counter = 0
                switched = True
        else:
            self.done_counter = 0

        update = PhaseUpdate(
            frame_idx=frame_idx,
            timestamp_s=float(timestamp_s),
            current_phase=self.current_phase,
            phase_one_hot=self.one_hot(),
            right_gripper_value=right_gripper_value,
            right_gripper_open=right_gripper_open,
            right_gripper_closed_seen=self.right_gripper_closed_seen,
            right_ee_speed=right_ee_speed,
            right_delta_trans_norm=right_delta_trans_norm,
            right_delta_rot_norm=right_delta_rot_norm,
            right_arm_stable=right_arm_stable,
            done_counter=self.done_counter,
            current_phase_frames=self.current_phase_frames,
            switched_this_frame=switched,
        )
        self._log_update(update)
        self._write_csv(update)
        return update

    def _right_gripper_open(self, value: float | None) -> bool | None:
        if not self.switch_cfg.require_right_gripper_open:
            return True
        if value is None:
            self._warn_once(
                "missing_right_gripper",
                "[phase_conditioning] Cannot read right gripper value from raw observation, "
                "observation.state, or last sent action; right->left switch condition fails.",
            )
            return None
        return float(value) > self.switch_cfg.right_gripper_open_threshold

    def _right_ee_speed(self) -> float | None:
        if len(self.right_ee_history) < self.switch_cfg.stable_window_frames:
            return None
        first_t, first_xyz = self.right_ee_history[0]
        last_t, last_xyz = self.right_ee_history[-1]
        dt = max(float(last_t - first_t), (len(self.right_ee_history) - 1) / self.fps)
        if dt <= 0:
            return None
        return float(np.linalg.norm(last_xyz - first_xyz) / dt)

    def _right_arm_stable(
        self,
        *,
        right_ee_speed: float | None,
        right_delta_trans_norm: float | None,
        right_delta_rot_norm: float | None,
        sent_action_available: bool,
    ) -> bool | None:
        if not self.switch_cfg.require_right_arm_stable:
            return True

        measured_available = right_ee_speed is not None
        sent_available = right_delta_trans_norm is not None and right_delta_rot_norm is not None
        measured_stable = (
            measured_available and right_ee_speed < self.switch_cfg.right_ee_speed_threshold
        )
        sent_stable = (
            sent_available
            and right_delta_trans_norm < self.switch_cfg.right_delta_trans_threshold
            and right_delta_rot_norm < self.switch_cfg.right_delta_rot_threshold
        )

        if self.switch_cfg.stable_source == STABLE_SOURCE_MEASURED_EE_POSE:
            if measured_available:
                return measured_stable
            self._warn_once(
                "missing_measured_ee_pose",
                "[phase_conditioning] measured_ee_pose stability requested but measured "
                "right_ee_pose history is unavailable; falling back to sent_action.",
            )
            return self._sent_action_stable_or_warn(sent_stable, sent_available, sent_action_available)

        if self.switch_cfg.stable_source == STABLE_SOURCE_SENT_ACTION:
            return self._sent_action_stable_or_warn(sent_stable, sent_available, sent_action_available)

        if measured_available and sent_available:
            return measured_stable and sent_stable
        if not measured_available and sent_available:
            self._warn_once(
                "missing_measured_ee_pose_both",
                "[phase_conditioning] stable_source=both but measured right_ee_pose is "
                "unavailable; falling back to sent_action.",
            )
            return sent_stable
        if measured_available and not sent_available:
            self._warn_once(
                "missing_sent_action_both",
                "[phase_conditioning] stable_source=both but sent right_delta_ee_pose is "
                "unavailable; switch condition fails until both sources are readable.",
            )
            return False
        self._warn_once(
            "missing_all_stability",
            "[phase_conditioning] Cannot compute right_arm_stable from measured_ee_pose "
            "or sent_action; right->left switch condition fails.",
        )
        return None

    def _sent_action_stable_or_warn(
        self,
        sent_stable: bool,
        sent_available: bool,
        sent_action_available: bool,
    ) -> bool | None:
        if sent_available:
            return sent_stable
        warning = (
            "[phase_conditioning] sent_action stability requested but no previous sent action "
            "is available yet; switch condition fails."
            if not sent_action_available
            else "[phase_conditioning] sent_action stability requested but right_delta_ee_pose "
            "is missing from the sent action; switch condition fails."
        )
        self._warn_once("missing_sent_action_stability", warning)
        return None

    def _right_to_left_ready(
        self,
        right_gripper_open: bool | None,
        right_arm_stable: bool | None,
    ) -> bool:
        if not self.switch_cfg.enabled:
            return False
        gripper_ok = True if not self.switch_cfg.require_right_gripper_open else bool(right_gripper_open)
        stable_ok = True if not self.switch_cfg.require_right_arm_stable else bool(right_arm_stable)
        closed_once_ok = True
        if self.switch_cfg.require_right_gripper_closed_once:
            closed_once_ok = self.right_gripper_closed_seen
            if gripper_ok and stable_ok and not closed_once_ok:
                self._warn_once(
                    "right_gripper_not_closed_once",
                    "[phase_conditioning] right gripper is open and right arm is stable, "
                    "but require_right_gripper_closed_once=true and no closed state has "
                    "been observed yet; right->left switch is held.",
                )
        return gripper_ok and stable_ok and closed_once_ok

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        logging.warning(message)

    def _log_update(self, update: PhaseUpdate) -> None:
        if not self.cfg.log_phase_state:
            return
        if (
            not update.switched_this_frame
            and update.frame_idx > 5
            and update.frame_idx % self.cfg.log_every_frames != 0
        ):
            return
        logging.info(
            "[phase_conditioning] frame=%d phase=%s one_hot=%s right_gripper_value=%s "
            "right_gripper_open=%s right_gripper_closed_seen=%s right_arm_stable=%s "
            "right_ee_speed=%s "
            "right_delta_trans=%s right_delta_rot=%s done_counter=%d phase_frames=%d switched=%s",
            update.frame_idx,
            update.current_phase,
            update.phase_one_hot.tolist(),
            _fmt_optional_float(update.right_gripper_value, 3),
            update.right_gripper_open,
            update.right_gripper_closed_seen,
            update.right_arm_stable,
            _fmt_optional_float(update.right_ee_speed),
            _fmt_optional_float(update.right_delta_trans_norm),
            _fmt_optional_float(update.right_delta_rot_norm),
            update.done_counter,
            update.current_phase_frames,
            update.switched_this_frame,
        )

    def _write_csv(self, update: PhaseUpdate) -> None:
        if self._csv_writer is None:
            return
        self._csv_writer.writerow(
            {
                "frame_idx": update.frame_idx,
                "timestamp": update.timestamp_s,
                "current_phase": update.current_phase,
                "phase_right_arm": float(update.phase_one_hot[0]),
                "phase_left_arm": float(update.phase_one_hot[1]),
                "right_gripper_value": update.right_gripper_value,
                "right_gripper_open": update.right_gripper_open,
                "right_gripper_closed_seen": update.right_gripper_closed_seen,
                "right_ee_speed": update.right_ee_speed,
                "right_delta_trans_norm": update.right_delta_trans_norm,
                "right_delta_rot_norm": update.right_delta_rot_norm,
                "right_arm_stable": update.right_arm_stable,
                "done_counter": update.done_counter,
                "current_phase_frames": update.current_phase_frames,
                "switched_this_frame": update.switched_this_frame,
            }
        )
        if self._csv_file is not None:
            self._csv_file.flush()


def extend_dataset_features_for_phase_conditioning(
    dataset_features: dict[str, Any],
    phase_cfg: PhaseConditioningConfig,
) -> None:
    if not phase_cfg.enabled:
        return
    state_feature = dataset_features.get(OBS_STATE_KEY)
    if state_feature is None:
        raise ValueError(
            "phase_conditioning.enabled=true but observation.state is missing from "
            "dataset features. Phase one-hot must be appended to observation.state."
        )
    if state_feature.get("dtype") != "float32" or len(tuple(state_feature.get("shape", ()))) != 1:
        raise ValueError(
            "phase_conditioning appends to a 1-D float32 observation.state. "
            f"Found feature: {state_feature!r}"
        )

    old_dim = int(tuple(state_feature["shape"])[0])
    names = state_feature.get("names")
    if names is None:
        names = [f"state_{idx}" for idx in range(old_dim)]
    else:
        names = [str(name) for name in _flatten_feature_names(names)]
    if len(names) != old_dim:
        raise ValueError(
            "observation.state names length does not match shape before phase append: "
            f"len(names)={len(names)} shape={state_feature['shape']}"
        )

    if old_dim >= 2 and tuple(names[-2:]) == PHASE_NAMES:
        return

    dataset_features[OBS_STATE_KEY] = {
        **state_feature,
        "shape": (old_dim + 2,),
        "names": names + list(PHASE_NAMES),
    }


def dataset_features_without_phase(
    dataset_features: dict[str, Any],
    phase_cfg: PhaseConditioningConfig,
) -> dict[str, Any]:
    if not phase_cfg.enabled:
        return dataset_features
    state_feature = dataset_features.get(OBS_STATE_KEY)
    if state_feature is None:
        raise ValueError(
            "phase_conditioning.enabled=true but observation.state is missing from dataset features."
        )
    names = [str(name) for name in _flatten_feature_names(state_feature.get("names"))]
    shape = tuple(state_feature.get("shape", ()))
    if len(shape) != 1 or len(names) != int(shape[0]):
        raise ValueError(
            "Invalid observation.state feature while preparing phase-conditioned observation: "
            f"{state_feature!r}"
        )
    if len(names) < 2 or tuple(names[-2:]) != PHASE_NAMES:
        raise ValueError(
            "phase_conditioning.enabled=true but observation.state feature names do not end "
            f"with {PHASE_NAMES}: {names[-2:]}"
        )

    base_features = dict(dataset_features)
    base_features[OBS_STATE_KEY] = {
        **state_feature,
        "shape": (int(shape[0]) - 2,),
        "names": names[:-2],
    }
    return base_features


def append_phase_to_observation_frame(
    observation_frame: dict[str, Any],
    *,
    current_phase: str,
    expected_state_dim: int | None,
) -> dict[str, Any]:
    if OBS_STATE_KEY not in observation_frame:
        raise ValueError(
            "phase_conditioning.enabled=true but observation.state is missing from "
            "the constructed policy observation."
        )

    original_state = np.asarray(observation_frame[OBS_STATE_KEY], dtype=np.float32).reshape(-1)
    phase = phase_one_hot(current_phase)
    appended_state = np.concatenate([original_state, phase], axis=0).astype(np.float32)
    if expected_state_dim is not None and appended_state.shape != (expected_state_dim,):
        raise _phase_observation_shape_error(
            tuple(appended_state.shape),
            (expected_state_dim,),
            enabled=True,
        )

    frame = dict(observation_frame)
    frame[OBS_STATE_KEY] = appended_state
    return frame


def build_policy_observation_frame(
    *,
    dataset_features: dict[str, Any],
    base_observation_features: dict[str, Any],
    obs_processed: dict[str, Any],
    phase_machine: PhaseStateMachine | None,
    policy,
) -> dict[str, Any]:
    if phase_machine is None:
        observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
        return observation_frame

    observation_frame = build_dataset_frame(
        base_observation_features,
        obs_processed,
        prefix=OBS_STR,
    )
    observation_frame = append_phase_to_observation_frame(
        observation_frame,
        current_phase=phase_machine.current_phase,
        expected_state_dim=_policy_expected_state_dim(policy),
    )
    _validate_observation_state_shape_for_policy(
        observation_frame,
        policy=policy,
        phase_enabled=True,
    )
    return observation_frame


def local_checkpoint_expected_state_dim(pretrained_path: str | Path | None) -> int | None:
    if not pretrained_path:
        return None

    raw_path = str(pretrained_path)
    path = Path(raw_path).expanduser()
    if not (path.is_absolute() or raw_path.startswith(("~", ".")) or path.exists()):
        return None

    config_path = path / "config.json"
    if not config_path.is_file():
        return None
    try:
        with open(config_path, "r") as f:
            checkpoint_cfg = json.load(f)
    except Exception as exc:
        logging.warning(
            "[phase_conditioning] Could not read checkpoint config for state-dim check: %s",
            exc,
        )
        return None

    input_features = checkpoint_cfg.get("input_features")
    if not isinstance(input_features, dict):
        return None
    state_feature = input_features.get(OBS_STATE_KEY)
    if not isinstance(state_feature, dict):
        return None
    shape = state_feature.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 1:
        return None
    return int(shape[0])


def validate_dataset_state_dim_against_checkpoint(
    *,
    dataset_features: dict[str, Any],
    policy_cfg,
    phase_cfg: PhaseConditioningConfig,
) -> None:
    expected_dim = local_checkpoint_expected_state_dim(getattr(policy_cfg, "pretrained_path", None))
    if expected_dim is None:
        return

    state_feature = dataset_features.get(OBS_STATE_KEY)
    if state_feature is None:
        raise ValueError(
            "Policy checkpoint expects observation.state but dataset features do not contain it."
        )
    actual_dim = int(tuple(state_feature["shape"])[0])
    if actual_dim == expected_dim:
        return

    raise ValueError(
        "observation.state dimension does not match the local checkpoint config.\n"
        f"- phase_conditioning.enabled: {phase_cfg.enabled}\n"
        f"- dataset/online observation.state dim after config: {actual_dim}\n"
        f"- checkpoint expected observation.state dim: {expected_dim}\n"
        "- If this is a phase-conditioned checkpoint, enable "
        "`phase_conditioning.enabled: true` in scripts/config/phase_conditioning_cfg.yaml "
        "and make sure the training dataset was produced by "
        "scripts/debug/annotate_dataset_phase.py.\n"
        "- If this is an old non-phase checkpoint, keep "
        "`phase_conditioning.enabled: false`.\n"
        "- If enabled=true and this still fails, the checkpoint may not have been trained "
        "on the phase-annotated dataset."
    )


def make_phase_state_machine(
    phase_cfg: PhaseConditioningConfig,
    *,
    dataset_features: dict[str, Any],
    fps: int | float,
    right_gripper_max_open: float | None,
) -> PhaseStateMachine | None:
    if not phase_cfg.enabled:
        return None
    return PhaseStateMachine(
        phase_cfg,
        fps=fps,
        state_names=_state_names_from_features(dataset_features),
        right_gripper_max_open=right_gripper_max_open,
    )


def self_test_phase_conditioning() -> None:
    project_root = Path(__file__).resolve().parents[2]
    phase_cfg = PhaseConditioningConfig.from_dict(
        {
            "enabled": True,
            "mode": PHASE_MODE_TWO_PHASE_ACTIVE_ARM,
            "initial_phase": PHASE_RIGHT_ARM,
            "log_phase_state": False,
            "switch_right_to_left": {
                "enabled": True,
                "require_right_gripper_open": True,
                "right_gripper_open_threshold": 0.8,
                "require_right_arm_stable": True,
                "stable_source": STABLE_SOURCE_SENT_ACTION,
                "right_delta_trans_threshold": 0.001,
                "right_delta_rot_threshold": 0.005,
                "stable_window_frames": 1,
                "dwell_frames": 2,
                "min_phase_frames": 1,
            },
        },
        project_root=project_root,
    )
    gripper_keys = {"right": "right_gripper_cmd"}
    stable_sent_action = {
        "right_delta_ee_pose.x": 0.0,
        "right_delta_ee_pose.y": 0.0,
        "right_delta_ee_pose.z": 0.0,
        "right_delta_ee_pose.rx": 0.0,
        "right_delta_ee_pose.ry": 0.0,
        "right_delta_ee_pose.rz": 0.0,
        "right_gripper_cmd": 1.0,
    }
    moving_sent_action = dict(stable_sent_action)
    moving_sent_action["right_delta_ee_pose.x"] = 0.01

    sm = PhaseStateMachine(
        phase_cfg,
        fps=30,
        state_names=[],
        right_gripper_max_open=1.0,
    )
    if sm.current_phase != PHASE_RIGHT_ARM:
        raise AssertionError("initial phase should be right_arm")

    update = sm.update(
        frame_idx=0,
        timestamp_s=0.0,
        raw_obs={"right_gripper_cmd": 0.0},
        obs_processed={"right_gripper_cmd": 0.0},
        last_sent_action=stable_sent_action,
        gripper_keys=gripper_keys,
    )
    if update.switched_this_frame or sm.current_phase != PHASE_RIGHT_ARM:
        raise AssertionError("closed right gripper must not switch phase")

    update = sm.update(
        frame_idx=1,
        timestamp_s=1 / 30,
        raw_obs={"right_gripper_cmd": 1.0},
        obs_processed={"right_gripper_cmd": 1.0},
        last_sent_action=moving_sent_action,
        gripper_keys=gripper_keys,
    )
    if update.switched_this_frame or update.right_arm_stable:
        raise AssertionError("open gripper with unstable right arm must not switch phase")

    update = sm.update(
        frame_idx=2,
        timestamp_s=2 / 30,
        raw_obs={"right_gripper_cmd": 1.0},
        obs_processed={"right_gripper_cmd": 1.0},
        last_sent_action=stable_sent_action,
        gripper_keys=gripper_keys,
    )
    if update.switched_this_frame or update.done_counter != 1:
        raise AssertionError("first stable dwell frame must not switch yet")

    update = sm.update(
        frame_idx=3,
        timestamp_s=3 / 30,
        raw_obs={"right_gripper_cmd": 1.0},
        obs_processed={"right_gripper_cmd": 1.0},
        last_sent_action=stable_sent_action,
        gripper_keys=gripper_keys,
    )
    if not update.switched_this_frame or sm.current_phase != PHASE_LEFT_ARM:
        raise AssertionError("second stable dwell frame should switch to left_arm")

    update = sm.update(
        frame_idx=4,
        timestamp_s=4 / 30,
        raw_obs={"right_gripper_cmd": 1.0},
        obs_processed={"right_gripper_cmd": 1.0},
        last_sent_action=moving_sent_action,
        gripper_keys=gripper_keys,
    )
    if update.switched_this_frame or sm.current_phase != PHASE_LEFT_ARM:
        raise AssertionError("left_arm phase must not switch back in this two-phase state machine")
    sm.close()

    base_features = {
        OBS_STATE_KEY: {
            "dtype": "float32",
            "shape": (3,),
            "names": ["a", "b", "c"],
        }
    }
    disabled_cfg = PhaseConditioningConfig.from_dict({"enabled": False}, project_root=project_root)
    disabled_frame = build_dataset_frame(
        dataset_features_without_phase(base_features, disabled_cfg),
        {"a": 1.0, "b": 2.0, "c": 3.0},
        prefix=OBS_STR,
    )
    if disabled_frame[OBS_STATE_KEY].shape != (3,):
        raise AssertionError("enabled=false should preserve observation.state dim")

    enabled_features = copy.deepcopy(base_features)
    extend_dataset_features_for_phase_conditioning(enabled_features, phase_cfg)
    frame = build_dataset_frame(
        dataset_features_without_phase(enabled_features, phase_cfg),
        {"a": 1.0, "b": 2.0, "c": 3.0},
        prefix=OBS_STR,
    )
    frame = append_phase_to_observation_frame(
        frame,
        current_phase=PHASE_LEFT_ARM,
        expected_state_dim=5,
    )
    if frame[OBS_STATE_KEY].shape != (5,) or not np.allclose(frame[OBS_STATE_KEY][-2:], [0.0, 1.0]):
        raise AssertionError("enabled=true should append left_arm phase one-hot to observation.state")

    gate_cfg = PhaseConditioningConfig.from_dict(
        {
            "enabled": True,
            "active_arm_action_gate": {"enabled": True, "log_action_norms": False},
        },
        project_root=project_root,
    )
    gate_sm = PhaseStateMachine(gate_cfg, fps=30, state_names=[])
    action = {
        **{f"left_delta_ee_pose.{axis}": 1.0 for axis in ("x", "y", "z", "rx", "ry", "rz")},
        **{f"right_delta_ee_pose.{axis}": 2.0 for axis in ("x", "y", "z", "rx", "ry", "rz")},
    }
    gated = apply_phase_action_gate(
        action,
        phase_machine=gate_sm,
        phase_conditioning=gate_cfg,
        gripper_keys={},
        frame_idx=0,
        source="self_test",
    )
    if any(gated[f"left_delta_ee_pose.{axis}"] != 0.0 for axis in ("x", "y", "z", "rx", "ry", "rz")):
        raise AssertionError("right_arm phase should zero left-arm deltas when action gate is enabled")
    if any(gated[f"right_delta_ee_pose.{axis}"] != 2.0 for axis in ("x", "y", "z", "rx", "ry", "rz")):
        raise AssertionError("right_arm phase should preserve right-arm deltas")

    logging.info("====== [PHASE CONDITIONING SELF-TEST] OK ======")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    self_test_phase_conditioning()
