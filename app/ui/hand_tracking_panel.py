"""Hand tracking controls panel styled as a floating HUD overlay."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget


class HandTrackingPanel(QFrame):
    """Toggleable hand tracking panel with video feed and status.

    Signals
    -------
    tracking_toggled : bool
        Emitted when the user toggles hand tracking on/off.
    """

    tracking_toggled = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("HandTrackingPanel")
        
        # Sleek glassmorphism style sheet
        self.setStyleSheet("""
            #HandTrackingPanel {
                background-color: rgba(30, 30, 30, 0.85);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 8px;
                padding: 6px;
            }
            QLabel {
                color: #eee;
            }
            QCheckBox {
                color: #eee;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Title bar with minimize/maximize control
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        title_label = QLabel("HAND TRACKING")
        title_label.setStyleSheet("font-weight: bold; font-size: 10px; color: #aaa; letter-spacing: 1px;")
        title_layout.addWidget(title_label)

        self._min_btn = QPushButton("−")
        self._min_btn.setFixedSize(20, 20)
        self._min_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #aaa;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                color: white;
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 4px;
            }
        """)
        self._min_btn.clicked.connect(self._on_minimize_toggled)
        title_layout.addWidget(self._min_btn)
        
        layout.addLayout(title_layout)

        # Toggle checkbox
        self._toggle = QCheckBox("Enable Hand Tracking")
        self._toggle.setToolTip(
            "Gestures:\n"
            "• Pinch & Drag (right hand) → Grab & rotate camera\n"
            "• Both Hands Pinch Apart/Together → Zoom in/out\n"
            "• Both Palms (hold) → Reset camera\n"
            "• Fist → Pause tracking"
        )
        self._toggle.toggled.connect(self._on_toggled)
        layout.addWidget(self._toggle)

        # Status label
        self._status = QLabel("Status: OFF")
        self._status.setStyleSheet("color: #888; font-size: 11px;")
        self._status.setVisible(False)
        layout.addWidget(self._status)

        # Video feed label (hidden initially)
        self._feed = QLabel()
        self._feed.setFixedSize(240, 180)
        self._feed.setStyleSheet(
            "background-color: #111; border: 1px solid #333; border-radius: 4px;"
        )
        self._feed.setAlignment(Qt.AlignCenter)
        self._feed.setText("Camera feed")
        self._feed.setVisible(False)
        layout.addWidget(self._feed)

        # Gesture hint (visible when tracking active)
        self._gesture_label = QLabel("Gesture: —")
        self._gesture_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._gesture_label.setVisible(False)
        layout.addWidget(self._gesture_label)

        # Reset progress bar (visible during camera reset hold)
        self._reset_progress = QProgressBar()
        self._reset_progress.setRange(0, 100)
        self._reset_progress.setValue(0)
        self._reset_progress.setTextVisible(False)
        self._reset_progress.setFixedHeight(6)
        self._reset_progress.setStyleSheet("""
            QProgressBar {
                background-color: rgba(255, 255, 255, 0.1);
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: #00bcd4;
                border-radius: 3px;
            }
        """)
        self._reset_progress.setVisible(False)
        layout.addWidget(self._reset_progress)

        self._minimized = False

    def _trigger_parent_resize(self) -> None:
        """Traverse upwards to find MainWindow and trigger positioning updates."""
        parent = self.parent()
        if parent is not None:
            from PySide6.QtWidgets import QMainWindow
            while parent is not None and not isinstance(parent, QMainWindow):
                parent = parent.parent()
            if parent is not None and hasattr(parent, "resizeEvent"):
                import PySide6.QtGui as QtGui
                parent.resizeEvent(QtGui.QResizeEvent(parent.size(), parent.size()))

    def _on_minimize_toggled(self) -> None:
        """Collapse or expand the webcam camera feed to save viewport space."""
        self._minimized = not self._minimized
        self._feed.setVisible(self._toggle.isChecked() and not self._minimized)
        self._min_btn.setText("+" if self._minimized else "−")
        self.adjustSize()
        self._trigger_parent_resize()

    def _on_toggled(self, checked: bool) -> None:
        """Handle toggle checkbox state change."""
        self._feed.setVisible(checked and not self._minimized)
        self._gesture_label.setVisible(checked)
        if checked:
            self._status.setVisible(True)
            self._status.setText("Status: STARTING...")
            self._status.setStyleSheet("color: #ffcc00; font-size: 11px;")
        else:
            if self._status.text() == "Status: ERROR":
                self._status.setVisible(True)
            else:
                self._status.setVisible(False)
                self._status.setText("Status: OFF")
                self._status.setStyleSheet("color: #888; font-size: 11px;")
            self._feed.clear()
            if self._status.text() != "Status: ERROR":
                self._feed.setText("Camera feed")
            self._gesture_label.setText("Gesture: —")
            self._reset_progress.setVisible(False)
            self._reset_progress.setValue(0)
        self.tracking_toggled.emit(checked)

        # Trigger resize adjustment when switching tracking on/off
        self.adjustSize()
        self._trigger_parent_resize()
        self.update()

    def update_frame(self, image: QImage) -> None:
        """Update the video feed with a new frame."""
        if self._feed.isVisible():
            pixmap = QPixmap.fromImage(image).scaled(
                self._feed.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self._feed.setPixmap(pixmap)

    def set_status(self, text: str, color: str = "#0f0") -> None:
        """Update the status label."""
        self._status.setText(f"Status: {text}")
        self._status.setStyleSheet(f"color: {color}; font-size: 11px;")
        self.update()

    def set_gesture(self, gesture: str) -> None:
        """Update the current gesture display."""
        display_names = {
            "point": "✋ HOVERING",
            "pinch": "🤏 GRABBING",
            "two_hand_zoom": "🔍 ZOOMING",
            "open_palm": "🖐 RESET",
            "fist": "✊ PAUSED",
            "none": "—",
        }
        self._gesture_label.setText(f"Gesture: {display_names.get(gesture, gesture)}")
        self.update()

    def set_reset_progress(self, progress: float) -> None:
        """Update the reset progress bar."""
        if progress > 0.0:
            was_visible = self._reset_progress.isVisible()
            self._reset_progress.setVisible(True)
            self._reset_progress.setValue(int(progress * 100))
            if not was_visible:
                self.adjustSize()
                self._trigger_parent_resize()
        else:
            was_visible = self._reset_progress.isVisible()
            self._reset_progress.setVisible(False)
            self._reset_progress.setValue(0)
            if was_visible:
                self.adjustSize()
                self._trigger_parent_resize()
        self.update()

    def set_error(self, message: str) -> None:
        """Show error state."""
        self._status.setText("Status: ERROR")
        self._status.setStyleSheet("color: #ff4444; font-size: 11px;")
        self._status.setVisible(True)
        self._feed.setText(message)
        self._toggle.setChecked(False)
        self.update()

