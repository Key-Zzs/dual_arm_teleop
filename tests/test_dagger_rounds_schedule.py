from __future__ import annotations

import sys
from pathlib import Path

TELEOP_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(TELEOP_ROOT))

from scripts.core.run_dagger_rounds import should_stop_dagger


def _state(round_id: int, frames: int, episodes: int, interventions: int = 1) -> dict:
    return {
        "round_id": round_id,
        "exported_frames": frames,
        "exported_episodes": episodes,
        "intervention_count": interventions,
    }


def _schedule(patience: int = 2) -> dict:
    return {
        "early_stop": {
            "enabled": True,
            "patience": patience,
            "stop_if_no_new_data": True,
            "stop_if_export_too_small": False,
            "stop_if_low_expert_ratio": False,
        }
    }


def test_no_exported_data_respects_patience() -> None:
    triggered, reason = should_stop_dagger(
        [],
        _state(1, 0, 0),
        _schedule(patience=2),
    )

    assert not triggered
    assert reason is None


def test_no_exported_data_stops_after_patience_rounds() -> None:
    triggered, reason = should_stop_dagger(
        [_state(1, 0, 0)],
        _state(2, 0, 0),
        _schedule(patience=2),
    )

    assert triggered
    assert "2 consecutive round(s)" in reason
