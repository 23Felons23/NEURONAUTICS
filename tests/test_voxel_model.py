"""Unit tests for VoxelModelRenderer.

All tests are headless — no vtkRenderWindow.Initialize() / .Start() calls.
Pattern mirrors tests/test_tractography.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import vtkmodules.all as vtk  # type: ignore

from app.core.volume import Volume
from app.viz.voxel_model import VoxelModelRenderer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_volume(
    shape: tuple[int, int, int] = (20, 20, 20),
    voxel_size_mm: float = 1.0,
    intensity_range: tuple[float, float] = (0.0, 100.0),
    seed: int = 42,
) -> Volume:
    """Return a small synthetic Volume filled with random intensities."""
    rng = np.random.default_rng(seed)
    lo, hi = intensity_range
    data = rng.uniform(lo, hi, shape).astype(np.float32)
    affine = np.eye(4, dtype=np.float64) * voxel_size_mm
    affine[3, 3] = 1.0
    return Volume(data=data, affine=affine)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVoxelModelRenderer:
    def test_from_volume_returns_renderer(self) -> None:
        vol = _make_volume()
        renderer = VoxelModelRenderer.from_volume(vol, stride=4)
        assert isinstance(renderer, VoxelModelRenderer)

    def test_actors_key_exists(self) -> None:
        vol = _make_volume()
        renderer = VoxelModelRenderer.from_volume(vol, stride=4)
        assert "voxel_model" in renderer.actors

    def test_actor_is_vtk_actor(self) -> None:
        vol = _make_volume()
        renderer = VoxelModelRenderer.from_volume(vol, stride=4)
        assert isinstance(renderer.actors["voxel_model"], vtk.vtkActor)

    def test_stride1_produces_more_or_equal_points_than_stride4(self) -> None:
        """Finer stride → more sampled voxels → point cloud has more points."""
        vol = _make_volume(shape=(16, 16, 16))
        r1 = VoxelModelRenderer.from_volume(vol, stride=1)
        r4 = VoxelModelRenderer.from_volume(vol, stride=4)
        m1 = r1.actors["voxel_model"].GetMapper()
        m4 = r4.actors["voxel_model"].GetMapper()
        m1.Update()
        m4.Update()
        pts1 = m1.GetInput().GetNumberOfPoints()
        pts4 = m4.GetInput().GetNumberOfPoints()
        assert pts1 >= pts4, (
            f"stride=1 should produce >= points as stride=4: {pts1} vs {pts4}"
        )

    def test_empty_volume_no_crash(self) -> None:
        """Volume of zeros (all below threshold) must not raise or return None."""
        data = np.zeros((10, 10, 10), dtype=np.float32)
        vol = Volume(data=data, affine=np.eye(4, dtype=np.float64))
        renderer = VoxelModelRenderer.from_volume(vol, stride=2)
        assert "voxel_model" in renderer.actors
        assert isinstance(renderer.actors["voxel_model"], vtk.vtkActor)

    def test_anisotropic_affine(self) -> None:
        """Non-identity affine (different voxel sizes) must not crash."""
        data = np.random.default_rng(0).uniform(0, 100, (12, 12, 12)).astype(np.float32)
        affine = np.diag([2.0, 1.5, 0.9, 1.0])
        vol = Volume(data=data, affine=affine)
        renderer = VoxelModelRenderer.from_volume(vol, stride=3)
        assert isinstance(renderer.actors["voxel_model"], vtk.vtkActor)

    def test_actor_visibility_on_by_default(self) -> None:
        """Actor must be visible by default so the mode starts enabled."""
        vol = _make_volume()
        renderer = VoxelModelRenderer.from_volume(vol, stride=4)
        assert renderer.actors["voxel_model"].GetVisibility() == 1

    def test_has_global_centroid(self) -> None:
        vol = _make_volume()
        renderer = VoxelModelRenderer.from_volume(vol, stride=4)
        assert hasattr(renderer, 'global_centroid')
        assert isinstance(renderer.global_centroid, np.ndarray)
        assert renderer.global_centroid.shape == (3,)

    def test_explode_no_crash_without_parcellation(self) -> None:
        """explode() should be a no-op when no parcellation is provided."""
        vol = _make_volume()
        renderer = VoxelModelRenderer.from_volume(vol, stride=4)
        renderer.explode(0.5, 150.0, renderer.global_centroid)  # should not crash

    def test_point_size(self) -> None:
        vol = _make_volume()
        renderer = VoxelModelRenderer.from_volume(vol, stride=4)
        assert renderer.actors["voxel_model"].GetProperty().GetPointSize() == 3

    def test_grayscale_lut(self) -> None:
        """LUT should map min→black, max→white (grayscale)."""
        vol = _make_volume()
        renderer = VoxelModelRenderer.from_volume(vol, stride=4)
        mapper = renderer.actors["voxel_model"].GetMapper()
        lut = mapper.GetLookupTable()
        s_min, s_max = mapper.GetScalarRange()
        # Check color at min is near black
        color_min = [0.0, 0.0, 0.0]
        lut.GetColor(s_min, color_min)
        assert all(c < 0.1 for c in color_min), f"Min color should be black, got {color_min}"
        # Check color at max is near white
        color_max = [0.0, 0.0, 0.0]
        lut.GetColor(s_max, color_max)
        assert all(c > 0.9 for c in color_max), f"Max color should be white, got {color_max}"

    def test_single_actor_key(self) -> None:
        """Should always be a single 'voxel_model' actor, not per-label actors."""
        vol = _make_volume()
        renderer = VoxelModelRenderer.from_volume(vol, stride=4)
        assert list(renderer.actors.keys()) == ["voxel_model"]

    def test_explode_with_parcellation(self) -> None:
        vol = _make_volume(shape=(10, 10, 10))
        # Create a parcellation volume where one half has label 1, other half has label 2
        parc_data = np.zeros((10, 10, 10), dtype=np.int32)
        parc_data[:5, :, :] = 1
        parc_data[5:, :, :] = 2
        parc_vol = Volume(data=parc_data, affine=np.eye(4, dtype=np.float64))
        
        renderer = VoxelModelRenderer.from_volume(vol, stride=2, parc_vol=parc_vol)
        assert renderer._labels is not None
        assert np.any(renderer._labels == 1)
        assert np.any(renderer._labels == 2)
        
        orig_pts = renderer._original_points.copy()
        
        # Explode
        renderer.explode(0.5, 10.0, renderer.global_centroid)
        
        # Get the modified points
        from vtkmodules.util import numpy_support
        mod_pts = numpy_support.vtk_to_numpy(renderer._vtk_points.GetData())
        
        # Make sure points have actually moved
        assert not np.array_equal(orig_pts, mod_pts)

    def test_explode_with_custom_centroids(self) -> None:
        vol = _make_volume(shape=(10, 10, 10))
        parc_data = np.zeros((10, 10, 10), dtype=np.int32)
        parc_data[:5, :, :] = 1
        parc_data[5:, :, :] = 2
        parc_vol = Volume(data=parc_data, affine=np.eye(4, dtype=np.float64))
        
        renderer = VoxelModelRenderer.from_volume(vol, stride=2, parc_vol=parc_vol)
        
        custom_centroids = {
            1: np.array([100.0, 100.0, 100.0]),
            2: np.array([-100.0, -100.0, -100.0])
        }
        
        renderer.explode(0.5, 10.0, renderer.global_centroid, custom_centroids)
        
        from vtkmodules.util import numpy_support
        mod_pts = numpy_support.vtk_to_numpy(renderer._vtk_points.GetData())
        assert len(mod_pts) > 0


class TestVoxelLabelMasking:
    """Tests for hide_label / restore_all_labels (Plan 8 addition)."""

    def test_hide_label_no_crash_without_parcellation(self) -> None:
        """hide_label() is a no-op when no parcellation was provided."""
        vol = _make_volume()
        renderer = VoxelModelRenderer.from_volume(vol, stride=4)
        renderer.hide_label(5)  # must not raise

    def test_restore_all_labels_no_crash_without_parcellation(self) -> None:
        vol = _make_volume()
        renderer = VoxelModelRenderer.from_volume(vol, stride=4)
        renderer.restore_all_labels()  # must not raise

    def test_hidden_labels_set_initially_empty(self) -> None:
        vol = _make_volume()
        renderer = VoxelModelRenderer.from_volume(vol, stride=4)
        assert hasattr(renderer, "_hidden_labels")
        assert len(renderer._hidden_labels) == 0

    def test_hide_label_adds_to_hidden_set(self) -> None:
        """When parcellation is absent, hide_label() returns early — but
        when labels exist, the label should be stored in _hidden_labels."""
        vol = _make_volume()
        renderer = VoxelModelRenderer.from_volume(vol, stride=4)
        # Simulate having labels so hide_label actually stores the value
        import numpy as np
        renderer._labels = np.zeros(10, dtype=int)
        renderer._original_points = np.zeros((10, 3), dtype=np.float32)
        from vtkmodules.util import numpy_support
        renderer._vtk_points = vtk.vtkPoints()
        renderer._vtk_points.SetData(
            numpy_support.numpy_to_vtk(renderer._original_points, deep=True)
        )
        renderer._label_centroids = {0: np.zeros(3)}
        renderer.hide_label(0)
        assert 0 in renderer._hidden_labels

    def test_restore_all_labels_clears_hidden_set(self) -> None:
        vol = _make_volume()
        renderer = VoxelModelRenderer.from_volume(vol, stride=4)
        import numpy as np
        renderer._labels = np.zeros(10, dtype=int)
        renderer._original_points = np.zeros((10, 3), dtype=np.float32)
        from vtkmodules.util import numpy_support
        renderer._vtk_points = vtk.vtkPoints()
        renderer._vtk_points.SetData(
            numpy_support.numpy_to_vtk(renderer._original_points, deep=True)
        )
        renderer._label_centroids = {0: np.zeros(3)}
        renderer.hide_label(0)
        assert 0 in renderer._hidden_labels
        renderer.restore_all_labels()
        assert len(renderer._hidden_labels) == 0
