"""Background worker for MediaPipe hand tracking.

Runs in a QThread. Captures webcam frames, performs hand landmark detection,
and emits Qt signals with results for the main thread to consume.
"""

from __future__ import annotations

import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage

from app.tracking.model_manager import ensure_model


class HandTrackingWorker(QObject):
    """Webcam → MediaPipe HandLandmarker → Qt signals.

    Signals
    -------
    frame_ready : QImage
        Annotated BGR→RGB video frame suitable for QLabel display.
        Emitted at ~15 fps (every other frame) to reduce GUI load.
    landmarks_detected : dict
        Raw landmark data dict emitted every frame (~30 fps).
        Keys: "landmarks" (list of 21 landmark objects),
              "index_tip" (x,y,z), "thumb_tip" (x,y,z),
              "pinch_distance" (float).
    hand_lost : (no args)
        Emitted when no hand is detected in the frame.
    error : str
        Emitted on camera or detection errors.
    """

    frame_ready = Signal(QImage)
    landmarks_detected = Signal(dict)
    two_hands_detected = Signal(dict)
    hand_lost = Signal()
    error = Signal(str)

    def __init__(self, camera_id: int = 0) -> None:
        super().__init__()
        self._camera_id = camera_id
        self._running = False
        self._frame_skip = 0  # Counter for frame_ready throttling

    @Slot()
    def run(self) -> None:
        """Main capture + inference loop. Call from QThread.started signal."""
        # --- Initialize MediaPipe ---
        try:
            model_path = ensure_model()
        except RuntimeError as exc:
            self.error.emit(str(exc))
            return

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )

        try:
            detector = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            self.error.emit(f"Failed to create HandLandmarker: {exc}")
            return

        # --- Open camera ---
        cap = None
        for cam_id in [self._camera_id, 0, 1, 2]:
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                break
            cap.release()

        if cap is None or not cap.isOpened():
            self.error.emit("No camera found. Check that a webcam is connected.")
            detector.close()
            return

        # Set resolution for performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self._running = True

        # Warm-up: skip first few frames (some cameras return black)
        for _ in range(5):
            cap.read()

        # --- Main loop ---
        last_send_time = 0.0

        while self._running and cap.isOpened():
            success, frame = cap.read()
            if not success:
                # Retry once before erroring
                success, frame = cap.read()
                if not success:
                    self.error.emit("Camera capture failed.")
                    break

            frame = cv2.flip(frame, 1)  # Mirror for natural interaction
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Performance: mark as non-writeable during inference
            rgb_frame.flags.writeable = False
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            try:
                result = detector.detect(mp_image)
            except Exception as exc:
                self.error.emit(f"Detection error: {exc}")
                break

            rgb_frame.flags.writeable = True

            if result.hand_landmarks and result.handedness:
                # Build a map of handedness → landmarks for all detected hands
                hand_map = {}  # "right" or "left" → landmarks
                for i, (landmarks, hdns) in enumerate(zip(result.hand_landmarks, result.handedness)):
                    category = hdns[0].category_name
                    # MediaPipe's handedness is anatomically inverted when the camera feed is mirrored.
                    # "Left" corresponds to the user's physical right hand.
                    if category == "Left":
                        hand_map["right"] = landmarks
                    else:
                        hand_map["left"] = landmarks

                now = time.time()
                throttle_ok = now - last_send_time > 0.03  # ~33 Hz cap

                if "right" in hand_map and throttle_ok:
                    # Emit two-hand data FIRST when both hands are visible,
                    # so the zoom-active flag is set before the rotation
                    # handler runs via landmarks_detected.
                    if "left" in hand_map:
                        self.two_hands_detected.emit({
                            "right_landmarks": hand_map["right"],
                            "left_landmarks": hand_map["left"],
                        })

                    # Always emit right-hand landmarks for rotation/gestures.
                    # The gesture mapper internally suppresses rotation when
                    # two-hand zoom is active.
                    right_landmarks = hand_map["right"]
                    data = {
                        "landmarks": right_landmarks,
                        "index_tip": (right_landmarks[8].x, right_landmarks[8].y, right_landmarks[8].z),
                        "thumb_tip": (right_landmarks[4].x, right_landmarks[4].y, right_landmarks[4].z),
                        "index_base": (right_landmarks[5].x, right_landmarks[5].y, right_landmarks[5].z),
                        "wrist": (right_landmarks[0].x, right_landmarks[0].y, right_landmarks[0].z),
                        "pinch_distance": (
                            (right_landmarks[4].x - right_landmarks[8].x) ** 2
                            + (right_landmarks[4].y - right_landmarks[8].y) ** 2
                        ) ** 0.5,
                    }
                    self.landmarks_detected.emit(data)
                    last_send_time = now
                elif "right" not in hand_map:
                    self.hand_lost.emit()

                # Draw landmarks on frame for all detected hands
                h, w = frame.shape[:2]
                for side, hand_landmarks in hand_map.items():
                    color = (0, 255, 128) if side == "right" else (120, 120, 120)
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(rgb_frame, (cx, cy), 3, color, -1)
            else:
                self.hand_lost.emit()

            # Emit video frame at ~15fps (every other frame)
            self._frame_skip += 1
            if self._frame_skip >= 2:
                self._frame_skip = 0
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                q_img = QImage(
                    rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
                ).copy()  # .copy() to decouple from numpy buffer
                self.frame_ready.emit(q_img)

            # Small yield to prevent CPU saturation
            time.sleep(0.001)

        # Cleanup
        cap.release()
        detector.close()

    def stop(self) -> None:
        """Request the worker loop to stop."""
        self._running = False
