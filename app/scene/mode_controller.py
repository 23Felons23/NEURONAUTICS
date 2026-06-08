"""Mode controller: camera-stable actor-group visibility switching.

Switching between visualization modes (T1 slices, tractography, voxel model…)
must not jump the camera.  ``ModeController`` saves the camera state before
toggling actor visibility and restores it afterwards.
"""

from __future__ import annotations

from app.scene.camera import CameraState
from app.scene.scene_manager import SceneManager


class ModeController:
    """Manages named groups of actors and switches their visibility.

    Parameters
    ----------
    scene:
        The ``SceneManager`` that owns the actors and renderer.
    """

    def __init__(self, scene: SceneManager) -> None:
        self._scene = scene
        self._groups: dict[str, list[str]] = {}  # mode_name → [actor_key, …]
        self._active: set[str] = set()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, mode_name: str, actor_keys: list[str]) -> None:
        """Register a named group of actor keys as a visualization mode.

        Parameters
        ----------
        mode_name:
            Human-readable mode identifier (e.g. ``"t1_slices"``).
        actor_keys:
            Keys in the ``SceneManager`` actor registry that belong to this
            mode.
        """
        self._groups[mode_name] = actor_keys
        self._active.add(mode_name)  # start visible by default

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def set_visible(self, mode_name: str, visible: bool) -> None:
        """Show or hide a mode's actors while preserving the camera.

        Parameters
        ----------
        mode_name:
            The mode to show or hide.
        visible:
            ``True`` to show, ``False`` to hide.
        """
        camera = self._scene.renderer.GetActiveCamera()
        state = CameraState.from_camera(camera)

        for key in self._groups.get(mode_name, []):
            self._scene.set_visible(key, visible)

        state.restore(camera)

        if visible:
            self._active.add(mode_name)
        else:
            self._active.discard(mode_name)

        # Trigger an immediate redraw — without this, VTK only repaints on
        # the next camera interaction (mouse move / scroll).
        self._scene.render()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def modes(self) -> list[str]:
        """All registered mode names, in registration order."""
        return list(self._groups.keys())

    def is_active(self, mode_name: str) -> bool:
        """Return ``True`` if the mode is currently visible."""
        return mode_name in self._active
