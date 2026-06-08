"""2D overlay scale-bar text actor for the VTK viewport."""

from __future__ import annotations

import vtkmodules.all as vtk  # type: ignore


class ScaleIndicator:
    """A VTK 2D text actor that displays the voxel size as a scale bar.

    Parameters
    ----------
    voxel_sizes_mm:
        (dx, dy, dz) voxel dimensions in millimetres.
    position:
        (x, y) pixel position of the bottom-left corner of the text.
    font_size:
        Text font size in points.
    """

    def __init__(
        self,
        voxel_sizes_mm: tuple[float, float, float],
        position: tuple[int, int] = (10, 10),
        font_size: int = 14,
    ) -> None:
        dx, dy, dz = voxel_sizes_mm
        self._actor = vtk.vtkTextActor()
        self._actor.SetInput(f"Voxel: {dx:.2f} × {dy:.2f} × {dz:.2f} mm")
        prop = self._actor.GetTextProperty()
        prop.SetFontSize(font_size)
        prop.SetColor(0.9, 0.9, 0.9)
        self._actor.SetPosition(*position)

    @property
    def actor(self) -> vtk.vtkTextActor:
        """The underlying ``vtkTextActor`` — add this to the renderer."""
        return self._actor

    def update(self, voxel_sizes_mm: tuple[float, float, float]) -> None:
        """Update the displayed text with new voxel dimensions."""
        dx, dy, dz = voxel_sizes_mm
        self._actor.SetInput(f"Voxel: {dx:.2f} × {dy:.2f} × {dz:.2f} mm")
