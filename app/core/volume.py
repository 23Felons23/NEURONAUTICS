"""Core Volume domain object."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from app.core.spatial import voxel_size


class Volume:
    """A 3-D voxel volume with an associated affine transform.

    Attributes
    ----------
    data:
        (x, y, z) float32 array of voxel intensities.
    affine:
        (4, 4) float64 voxel-to-world (RAS+) affine matrix.
    path:
        Optional source file path.
    """

    def __init__(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        path: Optional[Path] = None,
    ) -> None:
        if data.ndim != 3:
            raise ValueError(
                f"Volume data must be 3-D, got shape {data.shape}"
            )
        if affine.shape != (4, 4):
            raise ValueError(
                f"Affine must be (4, 4), got shape {affine.shape}"
            )
        self.data: np.ndarray = np.asarray(data, dtype=np.float32)
        self.affine: np.ndarray = np.asarray(affine, dtype=np.float64)
        self.path: Optional[Path] = path

    # ------------------------------------------------------------------
    # Shape / metadata helpers
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        """Voxel dimensions (nx, ny, nz)."""
        return self.data.shape  # type: ignore[return-value]

    @property
    def voxel_sizes_mm(self) -> np.ndarray:
        """Physical voxel size in mm along each axis."""
        return voxel_size(self.affine)

    # ------------------------------------------------------------------
    # Slicing helpers — return 2-D arrays for downstream rendering
    # ------------------------------------------------------------------

    def slice_x(self, idx: int) -> np.ndarray:
        """Sagittal slice at voxel index *idx* along the x axis."""
        idx = int(np.clip(idx, 0, self.shape[0] - 1))
        return self.data[idx, :, :]

    def slice_y(self, idx: int) -> np.ndarray:
        """Coronal slice at voxel index *idx* along the y axis."""
        idx = int(np.clip(idx, 0, self.shape[1] - 1))
        return self.data[:, idx, :]

    def slice_z(self, idx: int) -> np.ndarray:
        """Axial slice at voxel index *idx* along the z axis."""
        idx = int(np.clip(idx, 0, self.shape[2] - 1))
        return self.data[:, :, idx]

    @property
    def center_ijk(self) -> np.ndarray:
        """Voxel indices at the centre of the volume."""
        return np.array([s // 2 for s in self.shape], dtype=np.int64)

    # ------------------------------------------------------------------
    # Intensity helpers
    # ------------------------------------------------------------------

    def percentile_window(
        self, low: float = 1.0, high: float = 99.0
    ) -> tuple[float, float]:
        """Return (min, max) intensity at the given percentile clipping values."""
        nz = self.data[self.data > 0]
        if nz.size == 0:
            return float(self.data.min()), float(self.data.max())
        return float(np.percentile(nz, low)), float(np.percentile(nz, high))

    def __repr__(self) -> str:
        vs = self.voxel_sizes_mm
        return (
            f"Volume(shape={self.shape}, "
            f"voxel_size=({vs[0]:.2f}, {vs[1]:.2f}, {vs[2]:.2f}) mm, "
            f"path={self.path})"
        )
