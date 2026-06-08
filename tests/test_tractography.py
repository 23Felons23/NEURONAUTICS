"""Unit tests for TractographyBundle and TractographyRenderer."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.tractography import TractographyBundle
from app.viz.tractography_renderer import TractographyRenderer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bundle(n_streamlines: int = 3, n_points: int = 10) -> TractographyBundle:
    """Return a synthetic ``TractographyBundle``."""
    rng = np.random.default_rng(42)
    streamlines = [
        rng.uniform(0, 100, (n_points, 3)).astype(np.float32)
        for _ in range(n_streamlines)
    ]
    return TractographyBundle(streamlines=streamlines)


# ---------------------------------------------------------------------------
# TractographyBundle
# ---------------------------------------------------------------------------

class TestTractographyBundle:
    def test_repr_contains_class_name(self):
        bundle = _make_bundle()
        assert "TractographyBundle" in repr(bundle)

    def test_n_streamlines(self):
        bundle = _make_bundle(n_streamlines=5)
        assert bundle.n_streamlines == 5

    def test_n_points_total(self):
        bundle = _make_bundle(n_streamlines=3, n_points=10)
        assert bundle.n_points_total == 30

    def test_color_by_direction_shape(self):
        bundle = _make_bundle(n_streamlines=4, n_points=8)
        colours = bundle.color_by_direction()
        assert len(colours) == 4
        for sl_rgb in colours:
            assert sl_rgb.shape == (8, 3)

    def test_color_by_direction_range(self):
        """Each RGB value must be in [0, 255]."""
        bundle = _make_bundle(n_streamlines=3, n_points=6)
        colours = bundle.color_by_direction()
        for sl_rgb in colours:
            assert sl_rgb.min() >= 0
            assert sl_rgb.max() <= 255

    def test_empty_bundle(self):
        bundle = TractographyBundle(streamlines=[])
        assert bundle.n_streamlines == 0
        assert bundle.n_points_total == 0
        assert bundle.color_by_direction() == []

    def test_single_point_streamline(self):
        """A streamline with a single point should return black (direction = 0)."""
        sl = [np.zeros((1, 3), dtype=np.float32)]
        bundle = TractographyBundle(streamlines=sl)
        colours = bundle.color_by_direction()
        assert colours[0].shape == (1, 3)
        # direction vector is [0,0,0] → colour is [0,0,0]
        assert np.all(colours[0] == 0)

    def test_affine_default_identity(self):
        bundle = _make_bundle()
        assert bundle.affine is None or np.allclose(bundle.affine, np.eye(4))


# ---------------------------------------------------------------------------
# TractographyRenderer
# ---------------------------------------------------------------------------

class TestTractographyRenderer:
    def test_from_bundle_returns_renderer(self):
        bundle = _make_bundle()
        renderer = TractographyRenderer.from_bundle(bundle)
        assert isinstance(renderer, TractographyRenderer)

    def test_actors_key_exists(self):
        bundle = _make_bundle()
        renderer = TractographyRenderer.from_bundle(bundle)
        assert "streamlines" in renderer.actors

    def test_actor_is_vtk_actor(self):
        import vtkmodules.all as vtk  # type: ignore

        bundle = _make_bundle()
        renderer = TractographyRenderer.from_bundle(bundle)
        assert isinstance(renderer.actors["streamlines"], vtk.vtkActor)

    def test_empty_bundle_produces_actor(self):
        """Even an empty bundle must return a valid (empty) actor."""
        bundle = TractographyBundle(streamlines=[])
        renderer = TractographyRenderer.from_bundle(bundle)
        assert "streamlines" in renderer.actors

    def test_large_bundle_no_crash(self):
        """500 streamlines × 200 points should not raise."""
        bundle = _make_bundle(n_streamlines=500, n_points=200)
        renderer = TractographyRenderer.from_bundle(bundle)
        assert renderer is not None
