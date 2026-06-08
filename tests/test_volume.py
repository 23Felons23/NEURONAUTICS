"""Unit tests for app.core.volume."""

import numpy as np
import pytest

from app.core.volume import Volume


def _make_volume(shape=(64, 64, 40)):
    data = np.random.rand(*shape).astype(np.float32)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] *= 1.5  # 1.5 mm isotropic
    return Volume(data=data, affine=affine)


class TestVolumeConstruction:
    def test_basic(self):
        v = _make_volume()
        assert v.shape == (64, 64, 40)

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            Volume(data=np.zeros((4, 4)), affine=np.eye(4))

    def test_wrong_affine_raises(self):
        with pytest.raises(ValueError, match=r"\(4, 4\)"):
            Volume(data=np.zeros((4, 4, 4)), affine=np.eye(3))

    def test_voxel_sizes_isotropic(self):
        v = _make_volume()
        np.testing.assert_allclose(v.voxel_sizes_mm, [1.5, 1.5, 1.5], atol=1e-6)


class TestSlicers:
    def setup_method(self):
        self.v = _make_volume((10, 12, 14))

    def test_slice_z_shape(self):
        sl = self.v.slice_z(7)
        assert sl.shape == (10, 12)

    def test_slice_y_shape(self):
        sl = self.v.slice_y(5)
        assert sl.shape == (10, 14)

    def test_slice_x_shape(self):
        sl = self.v.slice_x(3)
        assert sl.shape == (12, 14)

    def test_slice_clips_to_bounds(self):
        sl_low = self.v.slice_z(-999)
        sl_high = self.v.slice_z(9999)
        assert sl_low.shape == sl_high.shape == (10, 12)


class TestCenterAndWindow:
    def test_center_ijk(self):
        v = _make_volume((10, 20, 30))
        np.testing.assert_array_equal(v.center_ijk, [5, 10, 15])

    def test_percentile_window_bounds(self):
        v = _make_volume()
        lo, hi = v.percentile_window(1, 99)
        assert lo < hi
        assert float(v.data.min()) <= lo <= hi <= float(v.data.max())
