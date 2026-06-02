from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.policies.act.configuration_act import ACTConfig, ACTLossWeightingConfig
from lerobot.policies.act.modeling_act import (
    compute_unweighted_action_l1_loss,
    compute_weighted_action_l1_loss,
    infer_gripper_dim_indices,
)

ANNOTATION_WEIGHT = "annotation.keyframe_weight"
ANNOTATION_EVENT = "annotation.gripper_event"


def _config(enabled: bool = True) -> ACTConfig:
    config = ACTConfig(device="cpu", push_to_hub=False)
    config.loss_weighting.enabled = enabled
    return config


def test_act_loss_weighting_config_accepts_dict() -> None:
    config = ACTConfig(
        device="cpu",
        push_to_hub=False,
        loss_weighting={"enabled": True, "gripper_dim_indices": [1], "gripper_dim_weight": 3.0},
    )

    assert isinstance(config.loss_weighting, ACTLossWeightingConfig)
    assert config.loss_weighting.enabled is True
    assert config.loss_weighting.gripper_dim_indices == [1]
    assert config.loss_weighting.gripper_dim_weight == 3.0


def test_infer_gripper_dim_indices_from_feature_names() -> None:
    indices = infer_gripper_dim_indices(
        4,
        action_dim_names=["shoulder", "left_gripper_cmd", "wrist", "right_gripper_cmd_bin"],
    )

    assert indices == [1, 3]


def test_disabled_path_matches_original_act_l1_formula() -> None:
    pred_action = torch.tensor(
        [
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
        ]
    )
    target_action = torch.ones_like(pred_action)
    action_is_pad = torch.tensor([[False, True, False], [False, False, True]])

    new_loss = compute_unweighted_action_l1_loss(pred_action, target_action, action_is_pad)
    old_loss = (
        F.l1_loss(pred_action, target_action, reduction="none") * (~action_is_pad).unsqueeze(-1)
    ).mean()

    torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)


def test_enabled_weighted_loss_matches_hand_calculation() -> None:
    config = _config(enabled=True)
    config.loss_weighting.gripper_dim_indices = [2]
    config.loss_weighting.gripper_dim_weight = 3.0
    config.loss_weighting.log_weighted_loss_breakdown = False
    pred_action = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    target_action = torch.zeros_like(pred_action)
    action_is_pad = torch.tensor([[False, False]])
    batch = {ANNOTATION_WEIGHT: torch.tensor([[1.0, 2.0]])}

    loss, _ = compute_weighted_action_l1_loss(
        pred_action,
        target_action,
        action_is_pad,
        batch,
        config,
    )

    loss_per_dim = F.l1_loss(pred_action, target_action, reduction="none")
    valid_mask = (~action_is_pad).unsqueeze(-1)
    timestep_weight = batch[ANNOTATION_WEIGHT].unsqueeze(-1)
    action_dim_weight = torch.tensor([[[1.0, 1.0, 3.0]]])
    effective_weight = valid_mask * timestep_weight * action_dim_weight
    expected = (loss_per_dim * effective_weight).sum() / effective_weight.sum()

    torch.testing.assert_close(loss, expected)


def test_enabled_missing_annotation_falls_back_to_unit_weights() -> None:
    config = _config(enabled=True)
    config.loss_weighting.infer_gripper_dim_from_feature_names = False
    config.loss_weighting.log_weighted_loss_breakdown = False
    pred_action = torch.tensor([[[1.0, 2.0], [100.0, 100.0]]])
    target_action = torch.zeros_like(pred_action)
    action_is_pad = torch.tensor([[False, True]])

    loss, _ = compute_weighted_action_l1_loss(
        pred_action,
        target_action,
        action_is_pad,
        batch={},
        config=config,
    )

    expected = torch.tensor((1.0 + 2.0) / 2.0)
    torch.testing.assert_close(loss, expected)


def test_gripper_dim_weight_scales_explicit_gripper_index() -> None:
    config = _config(enabled=True)
    config.loss_weighting.gripper_dim_indices = [2]
    config.loss_weighting.gripper_dim_weight = 3.0
    config.loss_weighting.log_weighted_loss_breakdown = False
    pred_action = torch.tensor([[[1.0, 1.0, 2.0]]])
    target_action = torch.zeros_like(pred_action)
    action_is_pad = torch.tensor([[False]])
    batch = {ANNOTATION_WEIGHT: torch.ones(1, 1)}

    loss, _ = compute_weighted_action_l1_loss(
        pred_action,
        target_action,
        action_is_pad,
        batch,
        config,
    )

    expected = torch.tensor((1.0 + 1.0 + 2.0 * 3.0) / (1.0 + 1.0 + 3.0))
    torch.testing.assert_close(loss, expected)


def test_padding_is_excluded_from_enabled_weighted_denominator() -> None:
    config = _config(enabled=True)
    config.loss_weighting.use_action_dim_weight = False
    config.loss_weighting.log_weighted_loss_breakdown = False
    pred_action = torch.tensor([[[1.0], [100.0]]])
    target_action = torch.zeros_like(pred_action)
    action_is_pad = torch.tensor([[False, True]])
    batch = {ANNOTATION_WEIGHT: torch.tensor([[1.0, 10.0]])}

    loss, _ = compute_weighted_action_l1_loss(
        pred_action,
        target_action,
        action_is_pad,
        batch,
        config,
    )

    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_event_breakdown_adds_opening_and_closing_metrics() -> None:
    config = _config(enabled=True)
    config.loss_weighting.gripper_dim_indices = [1]
    pred_action = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    target_action = torch.zeros_like(pred_action)
    action_is_pad = torch.tensor([[False, False, False]])
    batch = {
        ANNOTATION_WEIGHT: torch.tensor([[2.0, 2.0, 1.0]]),
        ANNOTATION_EVENT: torch.tensor([[2, 5, 0]]),
    }

    _, metrics = compute_weighted_action_l1_loss(
        pred_action,
        target_action,
        action_is_pad,
        batch,
        config,
    )

    assert "loss/closing_l1" in metrics
    assert "loss/opening_l1" in metrics
    assert "loss/gripper_l1" in metrics
    assert "loss/pose_l1" in metrics
    assert "loss/keyframe_ratio" in metrics
    assert "loss/normal_ratio" in metrics
    assert "loss/mean_annotation_weight" in metrics
    assert "loss/max_effective_weight" in metrics
    assert "loss/act_action_l1_unweighted" in metrics
    assert "loss/act_action_l1_weighted" in metrics
    assert "loss/act_keyframe_l1" in metrics
    assert "loss/act_normal_l1" in metrics
    assert "loss/act_gripper_l1" in metrics
    assert "loss/act_pose_l1" in metrics
    assert "loss/act_closing_l1" in metrics
    assert "loss/act_opening_l1" in metrics
