"""Camera state preservation for viewpoint-stable mode switches.

Captures a ``vtkCamera``'s position, focal point and view-up vector so that
switching visualization modes does not jump the camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import vtkmodules.all as vtk  # type: ignore


@dataclass
class CameraState:
    """Immutable snapshot of a VTK camera's orientation.

    Parameters
    ----------
    position:
        Camera position in world coordinates.
    focal_point:
        The point the camera is looking at.
    view_up:
        The "up" direction vector of the camera.
    """

    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    view_up: tuple[float, float, float]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_camera(cls, camera: vtk.vtkCamera) -> "CameraState":
        """Snapshot the current state of *camera*."""
        return cls(
            position=tuple(camera.GetPosition()),    # type: ignore[arg-type]
            focal_point=tuple(camera.GetFocalPoint()),  # type: ignore[arg-type]
            view_up=tuple(camera.GetViewUp()),        # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore(self, camera: vtk.vtkCamera) -> None:
        """Apply this state to *camera*, restoring its orientation."""
        camera.SetPosition(*self.position)
        camera.SetFocalPoint(*self.focal_point)
        camera.SetViewUp(*self.view_up)
