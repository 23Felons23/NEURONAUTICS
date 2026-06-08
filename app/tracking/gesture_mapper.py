"""Gesture recognition and VTK camera command mapping.

Translates raw MediaPipe hand landmarks into camera control commands
with EMA smoothing for jitter-free interaction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

# Landmark indices (MediaPipe 21-point hand model)
WRIST = 0
THUMB_TIP = 4
THUMB_IP = 3
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_PIP = 14
RING_TIP = 16
PINKY_PIP = 18
PINKY_TIP = 20

# Gesture thresholds
PINCH_THRESHOLD = 0.05
ROTATION_SCALE = 360.0  # degrees of rotation per normalized unit of hand movement
ZOOM_SCALE = 1.5        # dolly sensitivity
EMA_ALPHA = 0.6         # smoothing factor (0 = frozen, 1 = raw)


@dataclass
class CameraCommand:
    """VTK camera update produced by GestureMapper."""
    azimuth_delta: float = 0.0
    elevation_delta: float = 0.0
    dolly_factor: float = 1.0
    reset_camera: bool = False
    gesture: str = "none"
    reset_progress: float = 0.0


class GestureMapper:
    """Stateful mapper: raw landmarks → smoothed VTK camera commands.

    Call ``process(landmarks)`` each frame for single-hand gestures.
    Call ``process_two_hands(right, left)`` when both hands are visible
    to enable the two-hand pinch-zoom gesture.
    Call ``reset()`` when tracking is lost and resumed (clears EMA state).
    """

    def __init__(self, alpha: float = EMA_ALPHA) -> None:
        self._alpha = alpha
        self._smooth_x: float | None = None
        self._smooth_y: float | None = None
        self._prev_pinch: float | None = None
        self._prev_drag_x: float | None = None
        self._prev_drag_y: float | None = None
        self._open_palm_start_time: float | None = None
        self._reset_triggered: bool = False
        # Two-hand zoom state
        self._prev_two_hand_dist: float | None = None
        self._smooth_two_hand_dist: float | None = None
        self._zoom_active: bool = False
        self._two_hands_active: bool = False
        self._both_palms_active: bool = False

    def reset(self) -> None:
        """Clear all smoothing state."""
        self._smooth_x = None
        self._smooth_y = None
        self._prev_pinch = None
        self._prev_drag_x = None
        self._prev_drag_y = None
        self._open_palm_start_time = None
        self._reset_triggered = False
        self._prev_two_hand_dist = None
        self._smooth_two_hand_dist = None
        self._zoom_active = False
        self._two_hands_active = False
        self._both_palms_active = False

    def process(self, landmarks: list) -> CameraCommand:
        """Convert a frame's landmarks into a camera command.

        Parameters
        ----------
        landmarks:
            List of 21 landmark objects, each with ``.x``, ``.y``, ``.z``.

        Returns
        -------
        CameraCommand
        """
        two_hands = self._two_hands_active
        self._two_hands_active = False

        if not two_hands:
            self._zoom_active = False
            self._both_palms_active = False

        # When two-hand zoom is active, suppress single-hand processing
        # so rotation doesn't interfere with the zoom gesture.
        if self._zoom_active:
            return CameraCommand(gesture="two_hand_zoom")

        gesture = self._classify_gesture(landmarks)
        cmd = CameraCommand(gesture=gesture)

        if gesture == "fist":
            # Pause — return no-op command
            self._prev_drag_x = None
            self._prev_drag_y = None
            return cmd

        if gesture == "open_palm":
            if self._both_palms_active:
                if self._reset_triggered:
                    cmd.reset_progress = 1.0
                    return cmd

                if self._open_palm_start_time is None:
                    self._open_palm_start_time = time.time()
                elapsed = time.time() - self._open_palm_start_time
                cmd.reset_progress = min(elapsed / 1.5, 1.0)
                if elapsed >= 1.5:
                    cmd.reset_camera = True
                    self._reset_triggered = True
                self._prev_drag_x = None
                self._prev_drag_y = None
                return cmd
            else:
                gesture = "point"
                cmd.gesture = "point"
                self._open_palm_start_time = None
                self._reset_triggered = False
        else:
            self._open_palm_start_time = None
            self._reset_triggered = False

        # --- Rotation (from index finger base) ---
        raw_x = landmarks[INDEX_MCP].x
        raw_y = landmarks[INDEX_MCP].y

        if gesture == "pinch":
            if self._prev_drag_x is None or self._prev_drag_y is None or self._smooth_x is None or self._smooth_y is None:
                # Initialize drag anchor and smooth coordinates on pinch start
                self._smooth_x = raw_x
                self._smooth_y = raw_y
                self._prev_drag_x = raw_x
                self._prev_drag_y = raw_y
            else:
                # Drag is active: update smoothed position and calculate delta
                self._smooth_x = self._alpha * raw_x + (1 - self._alpha) * self._smooth_x
                self._smooth_y = self._alpha * raw_y + (1 - self._alpha) * self._smooth_y

                dx = self._smooth_x - self._prev_drag_x
                dy = self._smooth_y - self._prev_drag_y
                cmd.azimuth_delta = -dx * ROTATION_SCALE
                cmd.elevation_delta = -dy * ROTATION_SCALE

                self._prev_drag_x = self._smooth_x
                self._prev_drag_y = self._smooth_y
        else:
            # Not pinching: clear drag anchors and bypass smoothing lag
            self._smooth_x = raw_x
            self._smooth_y = raw_y
            self._prev_drag_x = None
            self._prev_drag_y = None

        # --- Zoom (from pinch distance) ---
        # Single-hand pinch-to-zoom is disabled; dolly_factor remains 1.0.
        self._prev_pinch = None

        return cmd

    def process_two_hands(
        self, right_landmarks: list, left_landmarks: list
    ) -> CameraCommand:
        """Compute a zoom command from two-hand pinch-apart / pinch-together.

        Both hands must be pinching (thumb-index close) for the gesture
        to be recognised.  The dolly factor is derived from the change
        in distance between the two hands' pinch midpoints.

        Parameters
        ----------
        right_landmarks:
            21 landmarks for the right hand.
        left_landmarks:
            21 landmarks for the left hand.

        Returns
        -------
        CameraCommand with ``gesture="two_hand_zoom"`` and a ``dolly_factor``
        when both hands are pinching, or a no-op command otherwise.
        """
        self._two_hands_active = True

        right_open = self._classify_gesture(right_landmarks) == "open_palm"
        left_open = self._classify_gesture(left_landmarks) == "open_palm"

        if right_open and left_open:
            self._both_palms_active = True
            self._zoom_active = False
            self._prev_two_hand_dist = None
            self._smooth_two_hand_dist = None
            return CameraCommand(gesture="open_palm")

        self._both_palms_active = False

        right_pinching = self._pinch_distance(right_landmarks) < PINCH_THRESHOLD
        left_pinching = self._pinch_distance(left_landmarks) < PINCH_THRESHOLD

        if not (right_pinching and left_pinching):
            # One or both hands are not pinching — deactivate zoom,
            # let the subsequent process() call handle the right hand.
            self._zoom_active = False
            self._prev_two_hand_dist = None
            self._smooth_two_hand_dist = None
            return CameraCommand(gesture="none")

        # Both hands are pinching — compute inter-hand distance
        # Use the midpoint between thumb tip and index tip on each hand
        r_mx = (right_landmarks[THUMB_TIP].x + right_landmarks[INDEX_TIP].x) / 2.0
        r_my = (right_landmarks[THUMB_TIP].y + right_landmarks[INDEX_TIP].y) / 2.0
        l_mx = (left_landmarks[THUMB_TIP].x + left_landmarks[INDEX_TIP].x) / 2.0
        l_my = (left_landmarks[THUMB_TIP].y + left_landmarks[INDEX_TIP].y) / 2.0

        raw_dist = ((r_mx - l_mx) ** 2 + (r_my - l_my) ** 2) ** 0.5

        cmd = CameraCommand(gesture="two_hand_zoom")
        self._zoom_active = True

        # Clear single-hand drag state so rotation doesn't jump when
        # transitioning back to one hand
        self._prev_drag_x = None
        self._prev_drag_y = None

        if self._smooth_two_hand_dist is None:
            # First frame of two-hand pinch — initialise, no movement yet
            self._smooth_two_hand_dist = raw_dist
            self._prev_two_hand_dist = raw_dist
            return cmd

        # EMA smooth the distance
        self._smooth_two_hand_dist = (
            self._alpha * raw_dist
            + (1 - self._alpha) * self._smooth_two_hand_dist
        )

        # Compute ratio: hands moving apart → zoom in (dolly > 1),
        # hands moving together → zoom out (dolly < 1)
        if self._prev_two_hand_dist > 0.001:
            ratio = self._smooth_two_hand_dist / self._prev_two_hand_dist
            # Scale the deviation from 1.0 by ZOOM_SCALE for sensitivity
            cmd.dolly_factor = 1.0 + (ratio - 1.0) * ZOOM_SCALE

        self._prev_two_hand_dist = self._smooth_two_hand_dist

        return cmd

    def _classify_gesture(self, landmarks: list) -> str:
        """Classify the current hand gesture."""
        pinch_dist = self._pinch_distance(landmarks)
        if pinch_dist < PINCH_THRESHOLD:
            return "pinch"

        # Check finger extension (tip Y vs PIP Y — in camera space, lower Y = higher on screen)
        tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        pips = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]

        extended = sum(
            1 for tip, pip in zip(tips, pips)
            if landmarks[tip].y < landmarks[pip].y
        )
        # Thumb: tip vs IP joint
        thumb_extended = landmarks[THUMB_TIP].y < landmarks[THUMB_IP].y

        if extended >= 4 and thumb_extended:
            return "open_palm"
        if extended == 0 and not thumb_extended:
            return "fist"

        return "point"

    @staticmethod
    def _pinch_distance(landmarks: list) -> float:
        """Euclidean distance between thumb tip and index tip."""
        dx = landmarks[THUMB_TIP].x - landmarks[INDEX_TIP].x
        dy = landmarks[THUMB_TIP].y - landmarks[INDEX_TIP].y
        return (dx * dx + dy * dy) ** 0.5
