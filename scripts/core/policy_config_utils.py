from __future__ import annotations

import logging
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

import yaml

ACT_POLICY_TYPES = {"act", "act_dagger"}
DIFFUSION_POLICY_TYPES = {"diffusion", "dp", "diffusion_policy"}
POLICY_CONFIG_META_KEYS = {"type", "config_path", "name"}


def default_policy_config_path(policy_type: str) -> str:
    policy_type = str(policy_type).strip().lower()
    if policy_type in ACT_POLICY_TYPES:
        return "scripts/policy_config/act_config.yaml"
    if policy_type in DIFFUSION_POLICY_TYPES:
        return "scripts/policy_config/diffusion_policy.yaml"
    raise ValueError(
        f"No config for policy type: {policy_type}. "
        "Supported policy types: act | diffusion | dp | diffusion_policy"
    )


def resolve_policy_config_path(
    policy_cfg: Dict[str, Any],
    scripts_dir: Path,
    project_root: Path,
) -> Path:
    """Resolve policy config paths without depending on the process cwd.

    Relative paths are checked in this order:
    1. Project root, so `scripts/policy_config/act_config.yaml` works from the
       lerobot_dual_arm_teleop project root.
    2. The scripts directory, so `policy_config/act_config.yaml` also works.
    """
    policy_type = str(policy_cfg.get("type", "")).strip().lower()
    raw_path = policy_cfg.get("config_path") or default_policy_config_path(policy_type)

    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        if path.is_file():
            return path
        raise FileNotFoundError(f"Policy config file does not exist: {path}")

    candidates = [project_root / path, scripts_dir / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    candidate_text = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Policy config file '{raw_path}' was not found. Tried:\n{candidate_text}"
    )


def load_policy_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Policy config yaml must contain a mapping at the top level: {path}")
    return cfg


def get_act_config_field_names() -> set[str]:
    from lerobot.policies import ACTConfig

    return {config_field.name for config_field in fields(ACTConfig)}


def _normalization_mode_from_config(value: Any) -> Any:
    from lerobot.configs.types import NormalizationMode

    if isinstance(value, NormalizationMode):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return NormalizationMode[text]
        except KeyError:
            try:
                return NormalizationMode(text)
            except ValueError as exc:
                valid = [mode.name for mode in NormalizationMode]
                raise ValueError(
                    f"Unknown normalization mode {value!r}. Expected one of: {valid}"
                ) from exc
    raise ValueError(
        "normalization_mapping values must be NormalizationMode names or strings. "
        f"Got {type(value).__name__}: {value!r}"
    )


def convert_normalization_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            "`normalization_mapping` must be a mapping such as "
            "{VISUAL: MEAN_STD, STATE: MEAN_STD, ACTION: MEAN_STD}."
        )
    return {str(key): _normalization_mode_from_config(mode) for key, mode in value.items()}


def _feature_type_from_config(value: Any) -> Any:
    from lerobot.configs.types import FeatureType

    if isinstance(value, FeatureType):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return FeatureType[text]
        except KeyError:
            try:
                return FeatureType(text)
            except ValueError as exc:
                valid = [feature_type.name for feature_type in FeatureType]
                raise ValueError(f"Unknown feature type {value!r}. Expected one of: {valid}") from exc
    raise ValueError(f"Feature type must be a string. Got {type(value).__name__}: {value!r}")


def _convert_policy_features(value: Any, field_name: str) -> dict[str, Any]:
    from lerobot.configs.types import PolicyFeature

    if not isinstance(value, dict):
        raise ValueError(f"`{field_name}` must be a mapping of feature names to feature specs.")

    converted: dict[str, Any] = {}
    for name, feature in value.items():
        if isinstance(feature, PolicyFeature):
            converted[str(name)] = feature
            continue
        if not isinstance(feature, dict):
            raise ValueError(
                f"`{field_name}.{name}` must be a mapping with `type` and `shape` fields."
            )
        if "type" not in feature or "shape" not in feature:
            raise ValueError(f"`{field_name}.{name}` must include `type` and `shape`.")
        shape = feature["shape"]
        if not isinstance(shape, (list, tuple)):
            raise ValueError(f"`{field_name}.{name}.shape` must be a list or tuple.")
        converted[str(name)] = PolicyFeature(
            type=_feature_type_from_config(feature["type"]),
            shape=tuple(shape),
        )
    return converted


def normalize_temporal_ensemble_coeff(value: Any) -> float | None:
    """Treat non-positive and None-like values as disabled temporal ensembling."""
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null", "~"}:
            return None
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(
                "`temporal_ensemble_coeff` must be a number, null, or None-like string. "
                f"Got: {value!r}"
            ) from exc

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if value > 0 else None

    raise ValueError(
        "`temporal_ensemble_coeff` must be numeric or null-like. "
        f"Got type: {type(value).__name__}"
    )


def _validate_optional_string_field(cfg: Dict[str, Any], field_name: str) -> None:
    value = cfg.get(field_name)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"`{field_name}` must be a string or null. Got {type(value).__name__}.")


def _validate_tags(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, list) or not all(isinstance(tag, str) for tag in value):
        raise ValueError("`tags` must be null or a list of strings.")


def validate_unknown_fields(
    policy_yaml_dict: Dict[str, Any],
    allowed_fields: set[str],
    config_path: Path | None = None,
) -> None:
    unknown_fields = sorted(set(policy_yaml_dict) - allowed_fields)
    if unknown_fields:
        source = f" in {config_path}" if config_path is not None else ""
        raise ValueError(
            f"Unknown ACT policy config field(s){source}: {unknown_fields}. "
            f"Allowed fields are: {sorted(allowed_fields)}"
        )


def merge_legacy_policy_fields(
    policy_yaml_dict: Dict[str, Any],
    legacy_policy_dict: Dict[str, Any] | None,
    allowed_fields: set[str],
    source_name: str,
) -> dict[str, Any]:
    merged = dict(policy_yaml_dict)
    legacy_policy_dict = legacy_policy_dict or {}
    legacy_overrides = {
        key: value
        for key, value in legacy_policy_dict.items()
        if key not in POLICY_CONFIG_META_KEYS and key in allowed_fields
    }
    if legacy_overrides:
        logging.warning(
            "[DEPRECATED] ACT field(s) still defined in %s: %s. "
            "Please move them to scripts/policy_config/*.yaml. "
            "For backward compatibility, these values override the policy yaml for this run.",
            source_name,
            sorted(legacy_overrides),
        )
        merged.update(legacy_overrides)
    return merged


def _act_init_device_and_restore_device(device: Any) -> tuple[Any, str | None]:
    """Allow indexed CUDA devices while keeping PreTrainedConfig's availability checks.

    This LeRobot version validates only base device strings like `cuda`, but PyTorch
    and downstream `.to(...)` calls accept `cuda:0`. We initialize ACTConfig with the
    base device, then restore the indexed string if that CUDA device is available.
    """
    if device is None:
        return None, None
    if not isinstance(device, str):
        raise ValueError(f"`device` must be a string or null. Got {type(device).__name__}.")

    text = device.strip()
    if text.startswith("cuda:"):
        return "cuda", text
    return text, None


def _restore_indexed_device_if_available(policy_cfg: Any, indexed_device: str | None) -> None:
    if indexed_device is None:
        return

    try:
        import torch

        torch_device = torch.device(indexed_device)
    except Exception as exc:
        raise ValueError(f"Invalid torch device string: {indexed_device!r}") from exc

    if torch_device.type != "cuda":
        policy_cfg.device = indexed_device
        return
    if not torch.cuda.is_available():
        logging.warning(
            "Device '%s' is not available. Keeping LeRobot-selected device '%s'.",
            indexed_device,
            policy_cfg.device,
        )
        return
    if torch_device.index is not None and torch_device.index >= torch.cuda.device_count():
        raise ValueError(
            f"Configured device '{indexed_device}' is not available. "
            f"CUDA device count: {torch.cuda.device_count()}."
        )
    policy_cfg.device = indexed_device


def build_act_config(
    policy_yaml_dict: Dict[str, Any],
    legacy_policy_dict: Dict[str, Any] | None = None,
    *,
    legacy_source_name: str = "config",
    runtime_overrides: Dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> PreTrainedConfig:
    from lerobot.policies import ACTConfig

    allowed_fields = get_act_config_field_names()
    validate_unknown_fields(policy_yaml_dict, allowed_fields, config_path=config_path)

    act_kwargs = merge_legacy_policy_fields(
        policy_yaml_dict,
        legacy_policy_dict,
        allowed_fields,
        source_name=legacy_source_name,
    )
    if runtime_overrides:
        unknown_runtime_fields = sorted(set(runtime_overrides) - allowed_fields)
        if unknown_runtime_fields:
            raise ValueError(f"Unknown ACT runtime override field(s): {unknown_runtime_fields}")
        act_kwargs.update({key: value for key, value in runtime_overrides.items() if value is not None})

    for feature_field in ("input_features", "output_features"):
        if feature_field in act_kwargs:
            if act_kwargs[feature_field] == {}:
                act_kwargs.pop(feature_field)
            else:
                act_kwargs[feature_field] = _convert_policy_features(
                    act_kwargs[feature_field], feature_field
                )

    if "normalization_mapping" in act_kwargs:
        act_kwargs["normalization_mapping"] = convert_normalization_mapping(
            act_kwargs["normalization_mapping"]
        )

    if "temporal_ensemble_coeff" in act_kwargs:
        act_kwargs["temporal_ensemble_coeff"] = normalize_temporal_ensemble_coeff(
            act_kwargs["temporal_ensemble_coeff"]
        )

    if (
        act_kwargs.get("temporal_ensemble_coeff") is not None
        and act_kwargs.get("n_action_steps", 100) != 1
    ):
        raise ValueError(
            "temporal_ensemble_coeff is enabled, so n_action_steps must be 1 "
            "for ACT temporal ensembling."
        )

    for field_name in (
        "device",
        "repo_id",
        "license",
        "pretrained_path",
        "pretrained_backbone_weights",
    ):
        _validate_optional_string_field(act_kwargs, field_name)
    if "tags" in act_kwargs:
        _validate_tags(act_kwargs["tags"])

    init_device, indexed_device = _act_init_device_and_restore_device(act_kwargs.get("device"))
    act_kwargs["device"] = init_device
    act_config = ACTConfig(**act_kwargs)
    _restore_indexed_device_if_available(act_config, indexed_device)
    return act_config


def build_policy_config(
    policy_type: str,
    policy_yaml_dict: Dict[str, Any],
    legacy_policy_dict: Dict[str, Any] | None = None,
    *,
    legacy_source_name: str = "config",
    runtime_overrides: Dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> PreTrainedConfig:
    policy_type = str(policy_type).strip().lower()
    if policy_type in ACT_POLICY_TYPES:
        return build_act_config(
            policy_yaml_dict,
            legacy_policy_dict=legacy_policy_dict,
            legacy_source_name=legacy_source_name,
            runtime_overrides=runtime_overrides,
            config_path=config_path,
        )
    if policy_type in DIFFUSION_POLICY_TYPES:
        raise NotImplementedError(
            "Diffusion policy config loading is reserved but not fully implemented yet."
        )
    raise ValueError(
        f"No config for policy type: {policy_type}. "
        "Supported policy types: act | diffusion | dp | diffusion_policy"
    )


def self_test_policy_config_loader() -> None:
    from lerobot.configs.types import NormalizationMode

    def expect_error(case_name: str, cfg: Dict[str, Any], expected: str) -> None:
        try:
            build_act_config(cfg)
        except Exception as exc:
            if expected not in str(exc):
                raise AssertionError(
                    f"{case_name}: expected error containing {expected!r}, got {exc!r}"
                ) from exc
            return
        raise AssertionError(f"{case_name}: expected an error but config loaded successfully.")

    cfg = build_act_config(
        {
            "temporal_ensemble_coeff": None,
            "n_action_steps": 100,
            "normalization_mapping": {
                "VISUAL": "MEAN_STD",
                "STATE": "MEAN_STD",
                "ACTION": "MEAN_STD",
            },
        }
    )
    if cfg.temporal_ensemble_coeff is not None or cfg.n_action_steps != 100:
        raise AssertionError("temporal_ensemble_coeff: null with n_action_steps: 100 should pass.")
    if cfg.normalization_mapping["VISUAL"] is not NormalizationMode.MEAN_STD:
        raise AssertionError("normalization_mapping VISUAL was not converted to NormalizationMode.")

    cfg = build_act_config({"temporal_ensemble_coeff": 0.01, "n_action_steps": 1})
    if cfg.temporal_ensemble_coeff != 0.01 or cfg.n_action_steps != 1:
        raise AssertionError("temporal_ensemble_coeff: 0.01 with n_action_steps: 1 should pass.")

    expect_error(
        "temporal ensemble n_action_steps guard",
        {"temporal_ensemble_coeff": 0.01, "n_action_steps": 10},
        "n_action_steps must be 1",
    )
    expect_error(
        "unknown ACT field guard",
        {"temporal_ensemble_coef": 0.01},
        "Unknown ACT policy config field",
    )
    logging.info("====== [POLICY CONFIG SELF-TEST] OK ======")
