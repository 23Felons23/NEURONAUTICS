"""Tests for hand tracking gesture mapper."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.tracking.gesture_mapper import (
    CameraCommand,
    EMA_ALPHA,
    GestureMapper,
    INDEX_MCP,
    INDEX_TIP,
    PINCH_THRESHOLD,
    THUMB_TIP,
    ZOOM_SCALE,
)


@dataclass
class MockLandmark:
    """Minimal landmark mock matching MediaPipe's interface."""
    x: float = 0.5
    y: float = 0.5
    z: float = 0.0


def _make_landmarks(**overrides: dict) -> list:
    """Create a list of 21 default landmarks with optional overrides.

    overrides maps landmark index → dict of {attr: value}.
    Example: _make_landmarks(**{4: {"x": 0.3, "y": 0.3}})
    """
    lms = [MockLandmark() for _ in range(21)]
    # Default thumb tip and index tip to be far apart to prevent accidental pinch
    lms[THUMB_TIP].x = 0.1
    lms[INDEX_TIP].x = 0.9
    # Extend index tip by default (y = 0.3 < pip y = 0.5) to make it a "point" gesture
    lms[INDEX_TIP].y = 0.3
    for idx, attrs in overrides.items():
        for attr, val in attrs.items():
            setattr(lms[int(idx)], attr, val)
    return lms


def _make_open_palm() -> list:
    """Create landmarks where all fingers are extended (tips above PIPs)."""
    lms = [MockLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    # Separate thumb and index to prevent pinch
    lms[THUMB_TIP].x = 0.1
    lms[INDEX_TIP].x = 0.9
    # PIP joints at y=0.6 (lower on screen)
    for pip_idx in [6, 10, 14, 18]:
        lms[pip_idx].y = 0.6
    # Fingertips at y=0.3 (higher on screen)
    for tip_idx in [8, 12, 16, 20]:
        lms[tip_idx].y = 0.3
    # Thumb: IP at 0.6, tip at 0.3
    lms[3].y = 0.6
    lms[4].y = 0.3
    return lms


def _make_fist() -> list:
    """Create landmarks where all fingers are curled (tips below PIPs)."""
    lms = [MockLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    # Separate thumb and index to prevent pinch
    lms[THUMB_TIP].x = 0.1
    lms[INDEX_TIP].x = 0.9
    # PIP joints at y=0.4 (higher)
    for pip_idx in [6, 10, 14, 18]:
        lms[pip_idx].y = 0.4
    # Fingertips at y=0.7 (lower = curled)
    for tip_idx in [8, 12, 16, 20]:
        lms[tip_idx].y = 0.7
    # Thumb: IP at 0.4, tip at 0.7
    lms[3].y = 0.4
    lms[4].y = 0.7
    return lms


def _make_pinch() -> list:
    """Create landmarks where thumb and index tips are close together."""
    lms = _make_landmarks()
    lms[THUMB_TIP].x = 0.5  # thumb tip
    lms[THUMB_TIP].y = 0.5
    lms[INDEX_TIP].x = 0.5 + PINCH_THRESHOLD * 0.5  # index tip very close
    lms[INDEX_TIP].y = 0.5
    return lms


class TestGestureClassification:
    """Test gesture recognition logic."""

    def test_pinch_detected(self):
        mapper = GestureMapper()
        lms = _make_pinch()
        cmd = mapper.process(lms)
        assert cmd.gesture == "pinch"

    def test_open_palm_detected_only_with_two_hands(self):
        mapper = GestureMapper()
        right = _make_open_palm()
        left = _make_open_palm()
        
        # Single hand should map to point (hovering)
        cmd_single = mapper.process(right)
        assert cmd_single.gesture == "point"
        
        # Two hands should map to open_palm (resetting)
        mapper.process_two_hands(right, left)
        cmd_two = mapper.process(right)
        assert cmd_two.gesture == "open_palm"

    def test_fist_detected(self):
        mapper = GestureMapper()
        lms = _make_fist()
        cmd = mapper.process(lms)
        assert cmd.gesture == "fist"

    def test_default_is_point(self):
        mapper = GestureMapper()
        # Default landmarks (all at 0.5, 0.5) — not pinch, not open palm, not fist
        lms = _make_landmarks()
        cmd = mapper.process(lms)
        assert cmd.gesture == "point"
        assert cmd.azimuth_delta == 0.0
        assert cmd.elevation_delta == 0.0

    def test_fist_returns_noop(self):
        """Fist gesture should return zero deltas (pause)."""
        mapper = GestureMapper()
        lms = _make_fist()
        cmd = mapper.process(lms)
        assert cmd.azimuth_delta == 0.0
        assert cmd.elevation_delta == 0.0
        assert cmd.dolly_factor == 1.0
        assert cmd.reset_camera is False


class TestTwoHandOpenPalmReset:
    """Test that both hands showing open palm triggers camera reset after sustained hold."""

    def test_single_hand_open_palm_does_not_reset(self):
        mapper = GestureMapper()
        lms = _make_open_palm()
        cmd = mapper.process(lms)
        assert cmd.gesture == "point"
        assert cmd.reset_camera is False
        assert cmd.reset_progress == 0.0
        assert mapper._open_palm_start_time is None

    def test_two_hands_open_palm_needs_sustained_hold(self):
        from unittest.mock import patch
        mapper = GestureMapper()
        right = _make_open_palm()
        left = _make_open_palm()
        
        with patch("time.time") as mock_time:
            mock_time.return_value = 1000.0
            # Frame 1: start hold
            mapper.process_two_hands(right, left)
            cmd1 = mapper.process(right)
            assert cmd1.reset_camera is False
            assert cmd1.reset_progress == 0.0

            # Frame 2: 0.75 seconds later
            mock_time.return_value = 1000.75
            mapper.process_two_hands(right, left)
            cmd2 = mapper.process(right)
            assert cmd2.reset_camera is False
            assert cmd2.reset_progress == pytest.approx(0.5)

    def test_two_hands_open_palm_triggers_after_threshold(self):
        from unittest.mock import patch
        mapper = GestureMapper()
        right = _make_open_palm()
        left = _make_open_palm()

        with patch("time.time") as mock_time:
            mock_time.return_value = 1000.0
            # Start hold
            mapper.process_two_hands(right, left)
            mapper.process(right)

            # Advance time by 1.5 seconds
            mock_time.return_value = 1001.5
            mapper.process_two_hands(right, left)
            cmd = mapper.process(right)
            assert cmd.reset_camera is True
            assert cmd.reset_progress == 1.0

            # It should lock progress at 1.0 and reset_camera to False on subsequent frames of the same hold
            mock_time.return_value = 1001.6
            mapper.process_two_hands(right, left)
            cmd2 = mapper.process(right)
            assert cmd2.reset_camera is False
            assert cmd2.reset_progress == 1.0

            # Releasing one hand (e.g. left hand stops open palm) should clear the reset triggered lock
            mock_time.return_value = 1001.7
            left_point = _make_landmarks()
            mapper.process_two_hands(right, left_point)
            cmd3 = mapper.process(right)
            assert cmd3.gesture == "point"
            assert mapper._reset_triggered is False
            assert mapper._open_palm_start_time is None
            assert cmd3.reset_progress == 0.0


class TestEMASmoothing:
    """Test EMA smoothing of rotation coordinates during pinching."""

    def test_first_pinch_frame_initializes_no_movement(self):
        mapper = GestureMapper(alpha=0.5)
        lms = _make_pinch()
        lms[INDEX_MCP].x = 0.8
        lms[INDEX_MCP].y = 0.3
        cmd = mapper.process(lms)
        # First frame of pinch sets anchor, so delta is 0.0
        assert cmd.azimuth_delta == 0.0
        assert cmd.elevation_delta == 0.0

    def test_smoothing_dampens_jump(self):
        mapper = GestureMapper(alpha=0.15)
        # Frame 1: pinch at center position
        lms1 = _make_pinch()
        lms1[INDEX_MCP].x = 0.5
        mapper.process(lms1)

        # Frame 2: pinch jump to far right
        lms2 = _make_pinch()
        lms2[INDEX_MCP].x = 0.9
        cmd = mapper.process(lms2)

        # With alpha=0.15, smoothed_x should be:
        # smoothed = 0.15 * 0.9 + 0.85 * 0.5 = 0.56
        # dx = 0.56 - 0.5 = 0.06
        # azimuth_delta = -dx * ROTATION_SCALE = -0.06 * 360 = -21.6
        assert -22.0 < cmd.azimuth_delta < -21.0

    def test_reset_clears_smoothing(self):
        mapper = GestureMapper()
        lms = _make_pinch()
        mapper.process(lms)
        assert mapper._smooth_x is not None
        assert mapper._prev_drag_x is not None
        mapper.reset()
        assert mapper._smooth_x is None
        assert mapper._smooth_y is None
        assert mapper._prev_drag_x is None
        assert mapper._prev_drag_y is None

    def test_elevation_delta_sign_direction(self):
        mapper = GestureMapper(alpha=1.0)
        # Frame 1: pinch at center
        lms1 = _make_pinch()
        lms1[INDEX_MCP].y = 0.5
        mapper.process(lms1)

        # Frame 2: drag UP (dy < 0)
        lms2 = _make_pinch()
        lms2[INDEX_MCP].y = 0.4
        cmd2 = mapper.process(lms2)
        # dy = 0.4 - 0.5 = -0.1
        # elevation_delta = -dy * ROTATION_SCALE = 0.1 * 360 = 36.0
        assert cmd2.elevation_delta == pytest.approx(36.0)

        # Reset and do drag DOWN (dy > 0)
        mapper.reset()
        lms3 = _make_pinch()
        lms3[INDEX_MCP].y = 0.5
        mapper.process(lms3)

        lms4 = _make_pinch()
        lms4[INDEX_MCP].y = 0.6
        cmd4 = mapper.process(lms4)
        # dy = 0.6 - 0.5 = 0.1
        # elevation_delta = -dy * ROTATION_SCALE = -36.0
        assert cmd4.elevation_delta == pytest.approx(-36.0)



class TestCameraCommand:
    """Test CameraCommand dataclass defaults."""

    def test_defaults(self):
        cmd = CameraCommand()
        assert cmd.azimuth_delta == 0.0
        assert cmd.elevation_delta == 0.0
        assert cmd.dolly_factor == 1.0
        assert cmd.reset_camera is False
        assert cmd.gesture == "none"


class TestZoomDisabled:
    """Test pinch-to-zoom is disabled and does not change dolly factor."""

    def test_pinch_does_not_produce_dolly(self):
        mapper = GestureMapper()
        # First pinch frame
        lms1 = _make_pinch()
        cmd1 = mapper.process(lms1)
        assert cmd1.gesture == "pinch"
        # dolly_factor should be 1.0
        assert cmd1.dolly_factor == 1.0

        # Second pinch frame with different distance
        lms2 = _make_pinch()
        lms2[8].x += 0.02  # Move index slightly
        cmd2 = mapper.process(lms2)
        assert cmd2.gesture == "pinch"
        # dolly_factor should remain 1.0 as zoom is disabled
        assert cmd2.dolly_factor == 1.0


class TestModelManager:
    """Test model manager path resolution (no download)."""

    def test_import(self):
        from app.tracking.model_manager import ensure_model, MODEL_URL
        assert "mediapipe-models" in MODEL_URL


def _make_two_hand_pinch(
    right_pinch_x: float = 0.7,
    right_pinch_y: float = 0.5,
    left_pinch_x: float = 0.3,
    left_pinch_y: float = 0.5,
) -> tuple[list, list]:
    """Create two sets of landmarks where both hands are pinching.

    Returns (right_landmarks, left_landmarks).
    The pinch position is controlled by placing thumb and index tips
    very close together at the specified coordinates.
    """
    right = _make_landmarks()
    right[THUMB_TIP].x = right_pinch_x
    right[THUMB_TIP].y = right_pinch_y
    right[INDEX_TIP].x = right_pinch_x + PINCH_THRESHOLD * 0.3
    right[INDEX_TIP].y = right_pinch_y

    left = _make_landmarks()
    left[THUMB_TIP].x = left_pinch_x
    left[THUMB_TIP].y = left_pinch_y
    left[INDEX_TIP].x = left_pinch_x + PINCH_THRESHOLD * 0.3
    left[INDEX_TIP].y = left_pinch_y

    return right, left


class TestTwoHandZoom:
    """Test two-hand pinch-zoom gesture."""

    def test_both_hands_pinching_produces_zoom_gesture(self):
        """When both hands pinch, gesture should be 'two_hand_zoom'."""
        mapper = GestureMapper()
        right, left = _make_two_hand_pinch()
        cmd = mapper.process_two_hands(right, left)
        assert cmd.gesture == "two_hand_zoom"

    def test_only_right_pinching_returns_noop(self):
        """When only right hand pinches, process_two_hands returns no-op
        (rotation handled by separate process() call)."""
        mapper = GestureMapper()
        right = _make_pinch()
        left = _make_open_palm()
        cmd = mapper.process_two_hands(right, left)
        # Should return no-op — the right hand's rotation is handled by process()
        assert cmd.gesture == "none"
        assert cmd.dolly_factor == 1.0

    def test_only_left_pinching_returns_noop(self):
        """When only left hand pinches, process_two_hands returns no-op."""
        mapper = GestureMapper()
        right = _make_landmarks()
        left = _make_pinch()
        cmd = mapper.process_two_hands(right, left)
        assert cmd.gesture == "none"

    def test_right_hand_rotation_continues_when_left_visible_but_not_pinching(self):
        """Simulates real signal flow: left hand visible but not pinching.
        process_two_hands returns no-op, then process() should still produce
        rotation from the right hand pinch-drag."""
        mapper = GestureMapper(alpha=1.0)

        # Frame 1: only right hand — establish drag anchor
        lms1 = _make_pinch()
        lms1[INDEX_MCP].x = 0.5
        mapper.process(lms1)

        # Frame 2: left hand appears (open palm), right hand moves
        right2 = _make_pinch()
        right2[INDEX_MCP].x = 0.6
        left2 = _make_open_palm()
        # process_two_hands runs first (only left not pinching → no-op)
        mapper.process_two_hands(right2, left2)
        # then process() runs for the right hand
        cmd = mapper.process(right2)
        # Should still produce rotation from right hand movement
        assert abs(cmd.azimuth_delta) > 0.1, "Right hand rotation should continue"

    def test_zoom_active_suppresses_rotation(self):
        """When both hands pinch (zoom active), subsequent process() should
        return no-op to prevent rotation during zoom."""
        mapper = GestureMapper()
        # Activate zoom
        right, left = _make_two_hand_pinch()
        mapper.process_two_hands(right, left)
        assert mapper._zoom_active is True

        # process() should be suppressed
        cmd = mapper.process(right)
        assert cmd.gesture == "two_hand_zoom"
        assert cmd.azimuth_delta == 0.0
        assert cmd.elevation_delta == 0.0

    def test_zoom_deactivates_when_hand_stops_pinching(self):
        """After zoom deactivates, process() should resume normal rotation."""
        mapper = GestureMapper()
        # Activate zoom
        right, left = _make_two_hand_pinch()
        mapper.process_two_hands(right, left)
        assert mapper._zoom_active is True

        # Left hand stops pinching
        right2 = _make_pinch()
        left2 = _make_open_palm()
        mapper.process_two_hands(right2, left2)
        assert mapper._zoom_active is False

        # process() should work normally again
        cmd = mapper.process(right2)
        assert cmd.gesture == "pinch"

    def test_first_frame_no_dolly(self):
        """First frame of two-hand pinch should initialise without dolly."""
        mapper = GestureMapper()
        right, left = _make_two_hand_pinch()
        cmd = mapper.process_two_hands(right, left)
        assert cmd.gesture == "two_hand_zoom"
        assert cmd.dolly_factor == 1.0

    def test_hands_apart_zooms_in(self):
        """Moving hands apart while pinching should produce dolly > 1 (zoom in)."""
        mapper = GestureMapper(alpha=1.0)  # No smoothing for predictable test
        # Frame 1: hands at initial distance
        right1, left1 = _make_two_hand_pinch(right_pinch_x=0.6, left_pinch_x=0.4)
        mapper.process_two_hands(right1, left1)

        # Frame 2: hands further apart
        right2, left2 = _make_two_hand_pinch(right_pinch_x=0.8, left_pinch_x=0.2)
        cmd = mapper.process_two_hands(right2, left2)

        assert cmd.gesture == "two_hand_zoom"
        assert cmd.dolly_factor > 1.0, "Hands apart should zoom in (dolly > 1)"

    def test_hands_together_zooms_out(self):
        """Moving hands together while pinching should produce dolly < 1 (zoom out)."""
        mapper = GestureMapper(alpha=1.0)
        # Frame 1: hands far apart
        right1, left1 = _make_two_hand_pinch(right_pinch_x=0.8, left_pinch_x=0.2)
        mapper.process_two_hands(right1, left1)

        # Frame 2: hands closer together
        right2, left2 = _make_two_hand_pinch(right_pinch_x=0.6, left_pinch_x=0.4)
        cmd = mapper.process_two_hands(right2, left2)

        assert cmd.gesture == "two_hand_zoom"
        assert cmd.dolly_factor < 1.0, "Hands together should zoom out (dolly < 1)"

    def test_zoom_scale_applied(self):
        """The ZOOM_SCALE constant should amplify the dolly deviation from 1.0."""
        mapper = GestureMapper(alpha=1.0)
        # Frame 1: distance = 0.2 (hands at 0.6 and 0.4)
        right1, left1 = _make_two_hand_pinch(right_pinch_x=0.6, left_pinch_x=0.4)
        mapper.process_two_hands(right1, left1)

        # Frame 2: distance = 0.4 (hands at 0.7 and 0.3)
        right2, left2 = _make_two_hand_pinch(right_pinch_x=0.7, left_pinch_x=0.3)
        cmd = mapper.process_two_hands(right2, left2)

        # ratio = 0.4 / 0.2 = 2.0, so dolly = 1.0 + (2.0 - 1.0) * ZOOM_SCALE
        expected_dolly = 1.0 + (2.0 - 1.0) * ZOOM_SCALE
        assert cmd.dolly_factor == pytest.approx(expected_dolly, abs=0.05)

    def test_reset_clears_zoom_state(self):
        """reset() should clear two-hand zoom tracking state."""
        mapper = GestureMapper()
        right, left = _make_two_hand_pinch()
        mapper.process_two_hands(right, left)
        assert mapper._smooth_two_hand_dist is not None
        assert mapper._zoom_active is True

        mapper.reset()
        assert mapper._prev_two_hand_dist is None
        assert mapper._smooth_two_hand_dist is None
        assert mapper._zoom_active is False

    def test_zoom_clears_drag_state(self):
        """Two-hand zoom should clear single-hand drag anchors to prevent jumps."""
        mapper = GestureMapper()
        # First: do a single-hand pinch to set drag state
        lms = _make_pinch()
        mapper.process(lms)
        assert mapper._prev_drag_x is not None

        # Now: two-hand zoom should clear drag state
        right, left = _make_two_hand_pinch()
        mapper.process_two_hands(right, left)
        assert mapper._prev_drag_x is None
        assert mapper._prev_drag_y is None
