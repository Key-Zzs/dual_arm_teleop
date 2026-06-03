from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TELEOP_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = TELEOP_ROOT / "scripts" / "tools" / "annotate_gripper_transition.py"

spec = importlib.util.spec_from_file_location("annotate_gripper_transition", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
agt = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = agt
spec.loader.exec_module(agt)


def _frames_and_events(transitions: list) -> list[tuple[int, int]]:
    return [(transition.frame, transition.event) for transition in transitions]


def _hysteresis(values: list[float], *, event_frame: str = "reached_state") -> list:
    return agt._detect_transitions_hysteresis(
        np.asarray(values, dtype=np.float32),
        mode="continuous",
        open_high=True,
        binary_threshold=0.5,
        open_threshold=0.8,
        close_threshold=0.2,
        event_frame=event_frame,
        min_transition_gap=0,
    )


def test_hysteresis_continuous_slope_emits_one_event_per_state_change() -> None:
    values = [1.0, 0.95, 0.70, 0.55, 0.40, 0.25, 0.15, 0.10, 0.15, 0.35, 0.60, 0.82, 0.90]

    transitions = _hysteresis(values)

    assert _frames_and_events(transitions) == [
        (6, agt.EVENT_CLOSING),
        (11, agt.EVENT_OPENING),
    ]


def test_hysteresis_event_frame_modes() -> None:
    values = [1.0, 0.95, 0.70, 0.55, 0.40, 0.25, 0.15, 0.10, 0.15, 0.35, 0.60, 0.82, 0.90]

    assert _frames_and_events(_hysteresis(values, event_frame="start")) == [
        (2, agt.EVENT_CLOSING),
        (9, agt.EVENT_OPENING),
    ]
    assert _frames_and_events(_hysteresis(values, event_frame="midpoint")) == [
        (3, agt.EVENT_CLOSING),
        (9, agt.EVENT_OPENING),
    ]


def test_hysteresis_binary_mode_uses_state_changes_not_delta_retriggering() -> None:
    transitions = agt._detect_transitions_hysteresis(
        np.asarray([1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        mode="binary",
        open_high=True,
        binary_threshold=0.5,
        open_threshold=0.8,
        close_threshold=0.2,
        event_frame="reached_state",
        min_transition_gap=0,
    )

    assert _frames_and_events(transitions) == [
        (2, agt.EVENT_CLOSING),
        (4, agt.EVENT_OPENING),
    ]


def test_derivative_detector_remains_available() -> None:
    transitions = agt._detect_transitions(
        np.asarray([1.0, 0.9, 0.8, 0.7], dtype=np.float32),
        detector="derivative",
        mode="continuous",
        open_high=True,
        delta_threshold=0.05,
        binary_threshold=0.5,
        open_threshold=0.8,
        close_threshold=0.2,
        event_frame="reached_state",
        min_transition_gap=0,
    )

    assert _frames_and_events(transitions) == [
        (1, agt.EVENT_CLOSING),
        (2, agt.EVENT_CLOSING),
        (3, agt.EVENT_CLOSING),
    ]


def test_expected_count_warning_is_reported_without_changing_transitions() -> None:
    left = agt._empty_side_detection(agt.SIDE_LEFT, 5)
    right = agt._empty_side_detection(agt.SIDE_RIGHT, 5)
    left.transitions = [
        agt.Transition(frame=1, event=agt.EVENT_CLOSING, strength=1.0),
        agt.Transition(frame=3, event=agt.EVENT_OPENING, strength=1.0),
    ]
    right.transitions = [
        agt.Transition(frame=2, event=agt.EVENT_CLOSING, strength=1.0),
        agt.Transition(frame=4, event=agt.EVENT_OPENING, strength=1.0),
    ]
    detection = agt.EpisodeDetection(
        episode_index=0,
        frame_indices=np.arange(5, dtype=np.int64),
        global_indices=np.arange(5, dtype=np.int64),
        left=left,
        right=right,
        combined_events=np.full(5, agt.EVENT_NORMAL, dtype=np.int64),
        combined_weights=np.full(5, agt.EVENT_WEIGHTS[agt.EVENT_NORMAL], dtype=np.float32),
        warnings=[],
    )
    args = SimpleNamespace(
        expected_left_opening=2,
        expected_left_closing=1,
        expected_right_opening=1,
        expected_right_closing=1,
    )

    agt._apply_expected_count_check(detection, args)

    assert detection.expected_count_ok is False
    assert "left_opening=1 expected=2" in detection.unexpected_count_warning
    assert len(left.transitions) == 2


def test_annotation_columns_are_prefixed_scalar_features() -> None:
    columns = agt._annotation_columns("annotation")
    info = {"features": {"action": {"dtype": "float32", "shape": [2], "names": ["a", "b"]}}}

    updated = agt._add_annotation_features_to_info_dict(info, columns)

    assert columns["gripper_event"] == "annotation.gripper_event"
    assert updated["features"]["annotation.gripper_event"] == {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    }
    assert updated["features"]["annotation.keyframe_weight"] == {
        "dtype": "float32",
        "shape": [1],
        "names": None,
    }
    assert "annotation.gripper_event" not in info["features"]


def test_export_path_safety_rejects_overlapping_roots() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        source = Path(tmpdir) / "source"
        source.mkdir()

        for output in (source, source / "child", Path(tmpdir)):
            try:
                agt._validate_output_dataset_path(source, output)
            except ValueError:
                pass
            else:  # pragma: no cover - explicit assertion path for script-style tests
                raise AssertionError(f"Expected overlapping path to be rejected: {output}")


def test_build_annotation_arrays_uses_global_indices() -> None:
    left = agt._empty_side_detection(agt.SIDE_LEFT, 2)
    right = agt._empty_side_detection(agt.SIDE_RIGHT, 2)
    left.events = np.asarray([agt.EVENT_CLOSING, agt.EVENT_NORMAL], dtype=np.int64)
    right.events = np.asarray([agt.EVENT_NORMAL, agt.EVENT_OPENING], dtype=np.int64)
    left.weights = np.asarray([6.0, 1.0], dtype=np.float32)
    right.weights = np.asarray([1.0, 6.0], dtype=np.float32)
    detection = agt.EpisodeDetection(
        episode_index=0,
        frame_indices=np.asarray([0, 1], dtype=np.int64),
        global_indices=np.asarray([2, 4], dtype=np.int64),
        left=left,
        right=right,
        combined_events=np.asarray([agt.EVENT_CLOSING, agt.EVENT_OPENING], dtype=np.int64),
        combined_weights=np.asarray([6.0, 6.0], dtype=np.float32),
        warnings=[],
    )
    columns = agt._annotation_columns("annotation")

    arrays = agt._build_annotation_arrays(total_frames=5, detections=[detection], columns=columns)

    assert arrays["annotation.gripper_event"].tolist() == [
        agt.EVENT_NORMAL,
        agt.EVENT_NORMAL,
        agt.EVENT_CLOSING,
        agt.EVENT_NORMAL,
        agt.EVENT_OPENING,
    ]
    assert arrays["annotation.keyframe_weight"].tolist() == [1.0, 1.0, 6.0, 1.0, 6.0]


if __name__ == "__main__":
    test_hysteresis_continuous_slope_emits_one_event_per_state_change()
    test_hysteresis_event_frame_modes()
    test_hysteresis_binary_mode_uses_state_changes_not_delta_retriggering()
    test_derivative_detector_remains_available()
    test_expected_count_warning_is_reported_without_changing_transitions()
    test_annotation_columns_are_prefixed_scalar_features()
    test_export_path_safety_rejects_overlapping_roots()
    test_build_annotation_arrays_uses_global_indices()
    print("gripper transition hysteresis tests passed")
