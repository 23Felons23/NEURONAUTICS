"""Unit tests for app.core.spatial."""

import numpy as np
import pytest

from app.core.spatial import (
    apply_affine,
    invert_affine,
    voxel_size,
    voxel_to_world,
    world_to_voxel,
)


IDENTITY = np.eye(4, dtype=np.float64)
SCALED = np.diag([2.0, 3.0, 4.0, 1.0])
SHIFTED = np.eye(4, dtype=np.float64)
SHIFTED[:3, 3] = [10.0, 20.0, 30.0]


class TestApplyAffine:
    def test_identity_returns_same(self):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = apply_affine(IDENTITY, pts)
        np.testing.assert_allclose(result, pts)

    def test_scaling(self):
        pts = np.array([[1.0, 1.0, 1.0]])
        result = apply_affine(SCALED, pts)
        np.testing.assert_allclose(result, [[2.0, 3.0, 4.0]])

    def test_translation(self):
        pts = np.array([[0.0, 0.0, 0.0]])
        result = apply_affine(SHIFTED, pts)
        np.testing.assert_allclose(result, [[10.0, 20.0, 30.0]])

    def test_single_point_1d_input(self):
        result = apply_affine(IDENTITY, np.array([5.0, 6.0, 7.0]))
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result, [[5.0, 6.0, 7.0]])


class TestInvertAffine:
    def test_identity_is_its_own_inverse(self):
        result = invert_affine(IDENTITY)
        np.testing.assert_allclose(result, IDENTITY, atol=1e-12)

    def test_scale_inverse(self):
        result = invert_affine(SCALED)
        np.testing.assert_allclose(
            result, np.diag([0.5, 1 / 3, 0.25, 1.0]), atol=1e-12
        )


class TestVoxelSize:
    def test_isotropic_1mm(self):
        result = voxel_size(IDENTITY)
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0])

    def test_anisotropic(self):
        result = voxel_size(SCALED)
        np.testing.assert_allclose(result, [2.0, 3.0, 4.0])


class TestRoundTrip:
    def test_world_to_voxel_roundtrip(self):
        affine = SHIFTED @ SCALED
        ijk = np.array([[5, 10, 15]], dtype=np.float64)
        xyz = voxel_to_world(affine, ijk)
        ijk_back = world_to_voxel(affine, xyz)
        np.testing.assert_allclose(ijk_back, ijk, atol=1)
