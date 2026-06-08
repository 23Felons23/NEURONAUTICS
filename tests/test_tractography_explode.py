"""Tests for tractography explosion (parcellation-based displacement).

Covers:
- Factor 0 → no displacement
- Factor 1 → points move away from global centroid
- No parc_vol → silent no-op
- Per-point labels assigned correctly
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.tractography import TractographyBundle
from app.core.volume import Volume
from app.viz.tractography_renderer import TractographyRenderer


def _make_bundle_and_parc() -> tuple[TractographyBundle, Volume]:
    """Create a minimal bundle and parcellation volume for testing.

    Two streamlines:
    - Streamline A: 3 points near (10, 10, 10) → will map to label 1
    - Streamline B: 3 points near (30, 30, 30) → will map to label 2

    Parcellation: 40×40×40 identity-affine volume
    - Label 1 in voxels [5:15, 5:15, 5:15]
    - Label 2 in voxels [25:35, 25:35, 25:35]
    """
    sl_a = np.array(
        [[10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [12.0, 10.0, 10.0]],
        dtype=np.float32,
    )
    sl_b = np.array(
        [[30.0, 30.0, 30.0], [31.0, 30.0, 30.0], [32.0, 30.0, 30.0]],
        dtype=np.float32,
    )
    bundle = TractographyBundle([sl_a, sl_b])

    parc_data = np.zeros((40, 40, 40), dtype=np.float32)
    parc_data[5:15, 5:15, 5:15] = 1
    parc_data[25:35, 25:35, 25:35] = 2
    parc_vol = Volume(data=parc_data, affine=np.eye(4))

    return bundle, parc_vol


def _read_vtk_points(renderer: TractographyRenderer) -> np.ndarray:
    """Read current point coordinates from the renderer's vtkPoints."""
    from vtkmodules.util import numpy_support
    assert renderer._vtk_points is not None
    vtk_data = renderer._vtk_points.GetData()
    return numpy_support.vtk_to_numpy(vtk_data).astype(np.float32)


class TestTractographyExplode:
    """Tests for TractographyRenderer.explode()."""

    def test_explode_zero_leaves_points_unchanged(self) -> None:
        """Factor=0 should leave all points at their original positions."""
        bundle, parc_vol = _make_bundle_and_parc()
        renderer = TractographyRenderer.from_bundle(bundle, parc_vol=parc_vol)

        assert renderer._original_points is not None
        original = renderer._original_points.copy()
        global_centroid = original.mean(axis=0)

        renderer.explode(0.0, 150.0, global_centroid)

        current = _read_vtk_points(renderer)
        np.testing.assert_allclose(current, original, atol=1e-4)

    def test_explode_nonzero_displaces_points(self) -> None:
        """Factor>0 should move points radially away from global centroid."""
        bundle, parc_vol = _make_bundle_and_parc()
        renderer = TractographyRenderer.from_bundle(bundle, parc_vol=parc_vol)

        assert renderer._original_points is not None
        original = renderer._original_points.copy()
        global_centroid = original.mean(axis=0)

        renderer.explode(1.0, 150.0, global_centroid)

        current = _read_vtk_points(renderer)

        # Overall points should have moved
        assert not np.allclose(current, original, atol=1.0)

        # Each label group should be further from global_centroid than before
        dist_before_1 = np.linalg.norm(original[:3].mean(axis=0) - global_centroid)
        dist_after_1 = np.linalg.norm(current[:3].mean(axis=0) - global_centroid)
        assert dist_after_1 > dist_before_1, (
            f"Label-1 points should move away from centroid: "
            f"before={dist_before_1:.2f}, after={dist_after_1:.2f}"
        )

        dist_before_2 = np.linalg.norm(original[3:].mean(axis=0) - global_centroid)
        dist_after_2 = np.linalg.norm(current[3:].mean(axis=0) - global_centroid)
        assert dist_after_2 > dist_before_2, (
            f"Label-2 points should move away from centroid: "
            f"before={dist_before_2:.2f}, after={dist_after_2:.2f}"
        )

    def test_explode_without_parcellation_is_noop(self) -> None:
        """Without parc_vol, explode() should silently do nothing (no crash)."""
        sl = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        bundle = TractographyBundle([sl])
        renderer = TractographyRenderer.from_bundle(bundle)  # no parc_vol

        # Must not raise
        renderer.explode(1.0, 150.0, np.zeros(3, dtype=np.float32))

        # Labels should not be set
        assert renderer._labels is None

    def test_labels_assigned_correctly(self) -> None:
        """Each streamline point should be mapped to the correct parcellation label."""
        bundle, parc_vol = _make_bundle_and_parc()
        renderer = TractographyRenderer.from_bundle(bundle, parc_vol=parc_vol)

        assert renderer._labels is not None, "Labels should be assigned when parc_vol provided"

        # First 3 points (streamline A near 10,10,10) → label 1
        np.testing.assert_array_equal(
            renderer._labels[:3],
            [1, 1, 1],
            err_msg="Streamline A points should map to label 1",
        )
        # Last 3 points (streamline B near 30,30,30) → label 2
        np.testing.assert_array_equal(
            renderer._labels[3:],
            [2, 2, 2],
            err_msg="Streamline B points should map to label 2",
        )
