from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.policies.diffusion.configuration_diffusion import (
    DiffusionConfig,
    DiffusionLossWeightingConfig,
)
from lerobot.policies.diffusion.modeling_diffusion import (
    compute_unweighted_denoising_mse_loss,
    compute_weighted_denoising_mse_loss,
    infer_gripper_dim_indices,
)

ANNOTATION_WEIGHT = "annotation.keyframe_weight"
ANNOTATION_EVENT = "annotation.gripper_event"


def _config(
    *,
    enabled: bool = True,
    do_mask_loss_for_padding: bool = True,
) -> DiffusionConfig:
    config = DiffusionConfig(
        device="cpu",
        push_to_hub=False,
        do_mask_loss_for_padding=do_mask_loss_for_padding,
    )
    config.loss_weighting.enabled = enabled
    return config


def test_diffusion_loss_weighting_config_accepts_dict() -> None:
    config = DiffusionConfig(
        device="cpu",
        push_to_hub=False,
        loss_weighting={"enabled": True, "gripper_dim_indices": [1], "gripper_dim_weight": 3.0},
    )

    assert isinstance(config.loss_weighting, DiffusionLossWeightingConfig)
    assert config.loss_weighting.enabled is True
    assert config.loss_weighting.gripper_dim_indices == [1]
    assert config.loss_weighting.gripper_dim_weight == 3.0


def test_infer_gripper_dim_indices_from_feature_names() -> None:
    indices = infer_gripper_dim_indices(
        4,
        action_dim_names=["shoulder", "left_gripper_cmd", "wrist", "right_gripper_cmd_bin"],
    )

    assert indices == [1, 3]


@pytest.mark.parametrize("do_mask_loss_for_padding", [True, False])
def test_disabled_path_matches_original_diffusion_mse_formula(do_mask_loss_for_padding: bool) -> None:
    config = _config(enabled=False, do_mask_loss_for_padding=do_mask_loss_for_padding)
    pred = torch.tensor(
        [
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
        ]
    )
    target = torch.ones_like(pred)
    action_is_pad = torch.tensor([[False, True, False], [False, False, True]])

    new_loss = compute_unweighted_denoising_mse_loss(pred, target, action_is_pad, config)
    old_loss = F.mse_loss(pred, target, reduction="none")
    if do_mask_loss_for_padding:
        old_loss = old_loss * (~action_is_pad).unsqueeze(-1)
    old_loss = old_loss.mean()

    torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)


def test_enabled_weighted_loss_matches_hand_calculation() -> None:
    config = _config(enabled=True, do_mask_loss_for_padding=True)
    config.loss_weighting.gripper_dim_indices = [2]
    config.loss_weighting.gripper_dim_weight = 3.0
    config.loss_weighting.log_weighted_loss_breakdown = False
    pred = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    target = torch.zeros_like(pred)
    action_is_pad = torch.tensor([[False, False]])
    batch = {ANNOTATION_WEIGHT: torch.tensor([[1.0, 2.0]])}

    loss, _ = compute_weighted_denoising_mse_loss(pred, target, action_is_pad, batch, config)

    mse_per_dim = F.mse_loss(pred, target, reduction="none")
    valid_mask = (~action_is_pad).unsqueeze(-1)
    horizon_weight = batch[ANNOTATION_WEIGHT].unsqueeze(-1)
    action_dim_weight = torch.tensor([[[1.0, 1.0, 3.0]]])
    effective_weight = valid_mask * horizon_weight * action_dim_weight
    expected = (mse_per_dim * effective_weight).sum() / effective_weight.sum()

    torch.testing.assert_close(loss, expected)


def test_enabled_missing_annotation_falls_back_to_unit_weights() -> None:
    config = _config(enabled=True, do_mask_loss_for_padding=True)
    config.loss_weighting.infer_gripper_dim_from_feature_names = False
    config.loss_weighting.log_weighted_loss_breakdown = False
    pred = torch.tensor([[[1.0, 2.0], [100.0, 100.0]]])
    target = torch.zeros_like(pred)
    action_is_pad = torch.tensor([[False, True]])

    loss, _ = compute_weighted_denoising_mse_loss(
        pred,
        target,
        action_is_pad,
        batch={},
        config=config,
    )

    expected = torch.tensor((1.0 + 4.0) / 2.0)
    torch.testing.assert_close(loss, expected)


def test_gripper_dim_weight_scales_explicit_gripper_index() -> None:
    config = _config(enabled=True, do_mask_loss_for_padding=True)
    config.loss_weighting.gripper_dim_indices = [2]
    config.loss_weighting.gripper_dim_weight = 3.0
    config.loss_weighting.log_weighted_loss_breakdown = False
    pred = torch.tensor([[[1.0, 1.0, 2.0]]])
    target = torch.zeros_like(pred)
    action_is_pad = torch.tensor([[False]])
    batch = {ANNOTATION_WEIGHT: torch.ones(1, 1)}

    loss, _ = compute_weighted_denoising_mse_loss(pred, target, action_is_pad, batch, config)

    expected = torch.tensor((1.0 + 1.0 + 4.0 * 3.0) / (1.0 + 1.0 + 3.0))
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize(
    ("do_mask_loss_for_padding", "expected"),
    [
        (True, torch.tensor(1.0)),
        (False, torch.tensor((1.0 + 10000.0 * 10.0) / (1.0 + 10.0))),
    ],
)
def test_enabled_padding_mask_semantics(do_mask_loss_for_padding: bool, expected: torch.Tensor) -> None:
    config = _config(enabled=True, do_mask_loss_for_padding=do_mask_loss_for_padding)
    config.loss_weighting.use_action_dim_weight = False
    config.loss_weighting.log_weighted_loss_breakdown = False
    pred = torch.tensor([[[1.0], [100.0]]])
    target = torch.zeros_like(pred)
    action_is_pad = torch.tensor([[False, True]])
    batch = {ANNOTATION_WEIGHT: torch.tensor([[1.0, 10.0]])}

    loss, _ = compute_weighted_denoising_mse_loss(pred, target, action_is_pad, batch, config)

    torch.testing.assert_close(loss, expected)


def test_event_breakdown_adds_opening_and_closing_metrics() -> None:
    config = _config(enabled=True, do_mask_loss_for_padding=True)
    config.loss_weighting.gripper_dim_indices = [1]
    pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    target = torch.zeros_like(pred)
    action_is_pad = torch.tensor([[False, False, False]])
    batch = {
        ANNOTATION_WEIGHT: torch.tensor([[2.0, 2.0, 1.0]]),
        ANNOTATION_EVENT: torch.tensor([[2, 5, 0]]),
    }

    _, metrics = compute_weighted_denoising_mse_loss(pred, target, action_is_pad, batch, config)

    assert "loss/closing_mse" in metrics
    assert "loss/opening_mse" in metrics
    assert "loss/gripper_mse" in metrics
    assert "loss/pose_mse" in metrics
    assert "loss/keyframe_ratio" in metrics


def test_diffusion_timestep_shaped_weight_is_not_used_as_horizon_weight() -> None:
    config = _config(enabled=True, do_mask_loss_for_padding=False)
    config.loss_weighting.use_action_dim_weight = False
    config.loss_weighting.log_weighted_loss_breakdown = False
    pred = torch.tensor([[[1.0], [1.0], [1.0]], [[2.0], [2.0], [2.0]]])
    target = torch.zeros_like(pred)
    batch = {ANNOTATION_WEIGHT: torch.tensor([1.0, 10.0])}

    loss, _ = compute_weighted_denoising_mse_loss(
        pred,
        target,
        action_is_pad=None,
        batch=batch,
        config=config,
    )

    expected = F.mse_loss(pred, target)
    torch.testing.assert_close(loss, expected)
