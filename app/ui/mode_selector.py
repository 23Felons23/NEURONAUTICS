"""Mode selector panel — a checkbox list wired to a ModeController."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QGroupBox, QVBoxLayout, QWidget

from app.scene.mode_controller import ModeController


class ModeSelector(QGroupBox):
    """A ``QGroupBox`` containing one checkbox per registered visualization mode.

    Toggling a checkbox calls ``ModeController.set_visible()`` on the
    corresponding mode, and enables/disables any nested sub-widgets.

    Parameters
    ----------
    controller:
        The ``ModeController`` that owns the actor groups.
    sub_widgets:
        A dict mapping mode names (as registered in ModeController) to styling QWidgets.
    parent:
        Optional Qt parent widget.
    """

    def __init__(
        self,
        controller: ModeController,
        sub_widgets: dict[str, QWidget] | None = None,
        parent=None,
    ) -> None:
        super().__init__("Visualization modes", parent)
        self._controller = controller
        
        # Space out the checkboxes and sub-widgets more
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(10, 14, 10, 14)
        
        sub_widgets = sub_widgets or {}

        for mode in controller.modes:
            label = mode.replace("_", " ").title()
            cb = QCheckBox(label)
            cb.setChecked(True)
            # Capture mode in default arg to avoid late-binding closure issue
            cb.toggled.connect(
                lambda checked, m=mode: controller.set_visible(m, checked)
            )
            layout.addWidget(cb)

            if mode in sub_widgets:
                sw = sub_widgets[mode]
                layout.addWidget(sw)
                cb.toggled.connect(sw.setEnabled)
                # Ensure initial enabled state matches checkbox status
                sw.setEnabled(cb.isChecked())

    def register_mode(self, mode_name: str, sub_widget: QWidget | None = None) -> None:
        """Add a new checkbox for *mode_name* (for dynamically loaded modes).

        Call this after construction when an optional renderer (e.g. voxel
        model) is loaded at runtime and registered with the ModeController.

        Parameters
        ----------
        mode_name:
            The mode name exactly as registered with ``ModeController``.
        sub_widget:
            Optional settings QWidget to nest under the checkbox.
        """
        label = mode_name.replace("_", " ").title()
        cb = QCheckBox(label)
        cb.setChecked(True)
        cb.toggled.connect(
            lambda checked, m=mode_name: self._controller.set_visible(m, checked)
        )
        self.layout().addWidget(cb)
        if sub_widget is not None:
            self.layout().addWidget(sub_widget)
            cb.toggled.connect(sub_widget.setEnabled)
            sub_widget.setEnabled(cb.isChecked())
