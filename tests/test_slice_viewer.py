"""Tests for app.viz.slice_viewer.

VTK does not need a display to run these checks — vtkImageSliceMapper and
vtkImageData work in off-screen / headless mode without a render window.
"""

from __future__ import annotations

import numpy as np
import pytest
import vtkmodules.all as vtk  # type: ignore

from app.core.volume import Volume
from app.viz.slice_viewer import SliceViewer, _numpy_to_vtk_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_volume(shape=(20, 24, 16), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """Synthetic volume with a diagonal affine (no shear/rotation)."""
    rng = np.random.default_rng(42)
    data = rng.random(shape).astype(np.float32)
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    affine[:3, 3] = origin
    return Volume(data=data, affine=affine)


# ---------------------------------------------------------------------------
# _numpy_to_vtk_image
# ---------------------------------------------------------------------------

class TestNumpyToVtkImage:
    def test_dimensions_match_volume_shape(self):
        vol = _make_volume((10, 12, 8))
        img = _numpy_to_vtk_image(vol)
        assert img.GetDimensions() == (10, 12, 8)

    def test_spacing_from_affine(self):
        vol = _make_volume(spacing=(2.0, 3.0, 4.0))
        img = _numpy_to_vtk_image(vol)
        np.testing.assert_allclose(img.GetSpacing(), (2.0, 3.0, 4.0), atol=1e-6)

    def test_origin_from_affine(self):
        vol = _make_volume(origin=(10.0, -5.0, 3.0))
        img = _numpy_to_vtk_image(vol)
        np.testing.assert_allclose(img.GetOrigin(), (10.0, -5.0, 3.0), atol=1e-6)

    def test_scalar_count_matches_voxel_count(self):
        shape = (10, 12, 8)
        vol = _make_volume(shape)
        img = _numpy_to_vtk_image(vol)
        assert img.GetNumberOfPoints() == shape[0] * shape[1] * shape[2]

    def test_scalar_type_is_float(self):
        vol = _make_volume()
        img = _numpy_to_vtk_image(vol)
        assert img.GetScalarType() == vtk.VTK_FLOAT


# ---------------------------------------------------------------------------
# SliceViewer construction
# ---------------------------------------------------------------------------

class TestSliceViewerConstruction:
    AXES = ("axial", "coronal", "sagittal")

    def test_all_three_actors_created(self):
        sv = SliceViewer(_make_volume())
        assert set(sv.actors.keys()) == set(self.AXES)

    def test_actors_are_vtk_image_slice(self):
        sv = SliceViewer(_make_volume())
        for name, actor in sv.actors.items():
            assert isinstance(actor, vtk.vtkImageSlice), f"{name} actor wrong type"

    def test_mappers_are_vtk_image_slice_mapper(self):
        sv = SliceViewer(_make_volume())
        for name, actor in sv.actors.items():
            assert isinstance(
                actor.GetMapper(), vtk.vtkImageSliceMapper
            ), f"{name} mapper wrong type"

    def test_initial_indices_are_midpoints(self):
        shape = (20, 24, 16)
        sv = SliceViewer(_make_volume(shape))
        # orientation 0→x(sagittal), 1→y(coronal), 2→z(axial)
        assert sv.slice_indices["sagittal"] == shape[0] // 2
        assert sv.slice_indices["coronal"]  == shape[1] // 2
        assert sv.slice_indices["axial"]    == shape[2] // 2

    def test_no_user_matrix_on_actors(self):
        """Regression: the double-transform bug set a UserMatrix on every actor."""
        sv = SliceViewer(_make_volume())
        for name, actor in sv.actors.items():
            m = actor.GetUserMatrix()
            assert m is None, f"{name} actor has an unexpected UserMatrix"

    def test_mapper_orientation_matches_axis(self):
        sv = SliceViewer(_make_volume())
        expected = {"sagittal": 0, "coronal": 1, "axial": 2}
        for name, orientation in expected.items():
            mapper = sv.actors[name].GetMapper()
            assert mapper.GetOrientation() == orientation, f"{name} orientation wrong"


# ---------------------------------------------------------------------------
# SliceViewer.update_slice
# ---------------------------------------------------------------------------

class TestUpdateSlice:
    def setup_method(self):
        self.shape = (20, 24, 16)
        self.sv = SliceViewer(_make_volume(self.shape))

    def test_updates_stored_index(self):
        self.sv.update_slice("axial", 10)
        assert self.sv.slice_indices["axial"] == 10

    def test_updates_mapper_slice_number(self):
        self.sv.update_slice("coronal", 8)
        mapper = self.sv.actors["coronal"].GetMapper()
        assert mapper.GetSliceNumber() == 8

    def test_clamps_negative_index_to_zero(self):
        self.sv.update_slice("sagittal", -99)
        assert self.sv.slice_indices["sagittal"] == 0

    def test_clamps_overflow_index_to_max(self):
        max_idx = self.shape[0] - 1   # sagittal → x dim
        self.sv.update_slice("sagittal", 9999)
        assert self.sv.slice_indices["sagittal"] == max_idx

    def test_all_axes_independently_updatable(self):
        self.sv.update_slice("axial",    5)
        self.sv.update_slice("coronal",  7)
        self.sv.update_slice("sagittal", 3)
        assert self.sv.slice_indices["axial"]    == 5
        assert self.sv.slice_indices["coronal"]  == 7
        assert self.sv.slice_indices["sagittal"] == 3
