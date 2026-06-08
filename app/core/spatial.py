"""Spatial utility functions for affine-based coordinate transforms."""

from __future__ import annotations

import numpy as np


def apply_affine(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4×4 affine matrix to an array of 3-D points.

    Parameters
    ----------
    affine:
        (4, 4) homogeneous transformation matrix.
    points:
        (N, 3) array of points in the source space.

    Returns
    -------
    np.ndarray
        (N, 3) transformed points.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points[np.newaxis, :]
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([points, ones])          # (N, 4)
    transformed = (affine @ homogeneous.T).T          # (N, 4)
    return transformed[:, :3]


def invert_affine(affine: np.ndarray) -> np.ndarray:
    """Return the inverse of a 4×4 affine matrix."""
    return np.linalg.inv(np.asarray(affine, dtype=np.float64))


def voxel_size(affine: np.ndarray) -> np.ndarray:
    """Return the (dx, dy, dz) voxel dimensions in mm from an affine matrix.

    Parameters
    ----------
    affine:
        (4, 4) voxel-to-world affine.

    Returns
    -------
    np.ndarray
        1-D array of shape (3,) with voxel sizes in millimetres.
    """
    affine = np.asarray(affine, dtype=np.float64)
    return np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))


def voxel_to_world(affine: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    """Convert voxel indices to world (mm) coordinates."""
    return apply_affine(affine, np.asarray(ijk, dtype=np.float64))


def world_to_voxel(affine: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Convert world (mm) coordinates to nearest voxel indices."""
    inv = invert_affine(affine)
    voxels = apply_affine(inv, np.asarray(xyz, dtype=np.float64))
    return np.round(voxels).astype(np.int64)
