"""VTK-based orthogonal slice viewer for a Volume.

Builds three ``vtkImageSlice`` actors — one per axis (axial, coronal, sagittal).
Spacing, origin, and direction cosines are extracted from the affine and stored
directly in the ``vtkImageData``, so no ``UserMatrix`` / plane-in-world-space
gymnastics are required.
"""

from __future__ import annotations

import numpy as np
import vtkmodules.all as vtk  # type: ignore

from app.core.volume import Volume


def _numpy_to_vtk_image(volume: Volume) -> vtk.vtkImageData:
    """Convert a Volume to a ``vtkImageData`` with correct spatial metadata.

    Spacing, origin, and (when VTK ≥ 9.0) direction cosines are derived from
    the affine so that VTK places every slice at the correct world-space
    position without any additional actor ``UserMatrix``.
    """
    from vtkmodules.util import numpy_support  # type: ignore

    affine = volume.affine  # (4, 4) float64

    # Column norms of the upper-left 3×3 give the voxel spacing
    spacing = np.linalg.norm(affine[:3, :3], axis=0).astype(np.float64)

    # World-space origin = affine applied to voxel (0, 0, 0)
    origin = affine[:3, 3].astype(np.float64)

    # Normalised direction cosines (columns)
    dirs = (affine[:3, :3] / spacing).astype(np.float64)

    nx, ny, nz = volume.shape
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(nx, ny, nz)
    vtk_image.SetSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))
    vtk_image.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))

    # SetDirectionMatrix available in VTK >= 9.0
    if hasattr(vtk_image, "SetDirectionMatrix"):
        m = vtk.vtkMatrix3x3()
        for i in range(3):
            for j in range(3):
                m.SetElement(i, j, float(dirs[i, j]))
        vtk_image.SetDirectionMatrix(m)

    # VTK expects x-fastest (Fortran / column-major) order
    flat = volume.data.flatten(order="F")
    vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_image.GetPointData().SetScalars(vtk_arr)
    return vtk_image


class SliceViewer:
    """Manages three orthogonal image-slice actors for a Volume.

    Parameters
    ----------
    volume:
        The brain volume to display.

    Attributes
    ----------
    actors:
        Dict mapping ``"axial"``, ``"coronal"``, ``"sagittal"`` to
        ``vtkImageSlice`` actors ready to be added to a renderer.
    slice_indices:
        Current slice index per axis.
    """

    # axis name -> vtkImageSliceMapper orientation constant
    #   0 = cut perpendicular to X (sagittal, YZ face visible)
    #   1 = cut perpendicular to Y (coronal,  XZ face visible)
    #   2 = cut perpendicular to Z (axial,    XY face visible)
    _AXIS_TO_ORIENTATION: dict[str, int] = {
        "sagittal": 0,
        "coronal":  1,
        "axial":    2,
    }

    def __init__(self, volume: Volume) -> None:
        self._volume = volume
        self._vtk_image = _numpy_to_vtk_image(volume)

        wmin, wmax = volume.percentile_window(1.0, 99.0)
        self._wmin = wmin
        self._wmax = wmax

        self.actors: dict[str, vtk.vtkImageSlice] = {}
        self.slice_indices: dict[str, int] = {}

        for name, orientation in self._AXIS_TO_ORIENTATION.items():
            n = volume.shape[orientation]   # orientation 0/1/2 == x/y/z dim
            mid = n // 2
            self.slice_indices[name] = mid
            self.actors[name] = self._build_slice_actor(orientation, mid)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_slice(self, axis_name: str, index: int) -> None:
        """Move a slice plane to the given voxel index."""
        orientation = self._AXIS_TO_ORIENTATION[axis_name]
        n = self._volume.shape[orientation]
        index = int(np.clip(index, 0, n - 1))
        self.slice_indices[axis_name] = index

        mapper: vtk.vtkImageSliceMapper = self.actors[axis_name].GetMapper()
        mapper.SetSliceNumber(index)
        mapper.Update()

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_slice_actor(self, orientation: int, index: int) -> vtk.vtkImageSlice:
        """Build a single ``vtkImageSlice`` actor."""
        mapper = vtk.vtkImageSliceMapper()
        mapper.SetInputData(self._vtk_image)
        mapper.SetOrientation(orientation)
        mapper.SetSliceNumber(index)
        mapper.Update()

        prop = vtk.vtkImageProperty()
        prop.SetColorWindow(self._wmax - self._wmin)
        prop.SetColorLevel((self._wmax + self._wmin) / 2.0)
        prop.SetInterpolationTypeToLinear()

        actor = vtk.vtkImageSlice()
        actor.SetMapper(mapper)
        actor.SetProperty(prop)
        # No UserMatrix — geometry is fully encoded in vtkImageData
        return actor
