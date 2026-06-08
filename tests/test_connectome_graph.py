"""Unit tests for ConnectomeGraphRenderer — all headless (no render window)."""

from __future__ import annotations

import numpy as np
import pytest
import vtkmodules.all as vtk  # type: ignore

from app.core.connectome import ConnectomeMatrix
from app.viz.connectome_graph import ConnectomeGraphRenderer


def _make_matrix(n: int = 8) -> ConnectomeMatrix:
    rng = np.random.default_rng(1)
    w = rng.uniform(0, 10, (n, n)).astype(np.float32)
    w = (w + w.T) / 2
    np.fill_diagonal(w, 0.0)
    return ConnectomeMatrix(weights=w, labels=[f"R{i}" for i in range(n)])


class TestConnectomeGraphRenderer:
    def test_from_matrix_returns_renderer(self):
        cm = _make_matrix()
        r = ConnectomeGraphRenderer.from_matrix(cm)
        assert isinstance(r, ConnectomeGraphRenderer)

    def test_actors_keys_exist(self):
        cm = _make_matrix()
        r = ConnectomeGraphRenderer.from_matrix(cm)
        assert "connectome_nodes" in r.actors
        assert "connectome_edges" in r.actors

    def test_node_actor_is_vtk_actor(self):
        cm = _make_matrix()
        r = ConnectomeGraphRenderer.from_matrix(cm)
        assert isinstance(r.actors["connectome_nodes"], vtk.vtkActor)

    def test_edge_actor_is_vtk_actor(self):
        cm = _make_matrix()
        r = ConnectomeGraphRenderer.from_matrix(cm)
        assert isinstance(r.actors["connectome_edges"], vtk.vtkActor)

    def test_custom_centroids_accepted(self):
        cm = _make_matrix(4)
        centroids = np.eye(4, 3, dtype=np.float32) * 20.0
        r = ConnectomeGraphRenderer.from_matrix(cm, centroids=centroids)
        assert isinstance(r.actors["connectome_nodes"], vtk.vtkActor)

    def test_empty_matrix_no_crash(self):
        """A 1-node matrix with no off-diagonal edges must not raise."""
        w = np.zeros((1, 1), dtype=np.float32)
        cm = ConnectomeMatrix(weights=w, labels=["only"])
        r = ConnectomeGraphRenderer.from_matrix(cm)
        assert "connectome_nodes" in r.actors

    def test_zero_weight_matrix_no_crash(self):
        """All-zero weight matrix → no edges drawn, must not raise."""
        w = np.zeros((5, 5), dtype=np.float32)
        cm = ConnectomeMatrix(weights=w)
        r = ConnectomeGraphRenderer.from_matrix(cm, top_k_edges=10)
        assert isinstance(r.actors["connectome_edges"], vtk.vtkActor)

    def test_explode_with_custom_centroids(self):
        cm = _make_matrix(4)
        centroids = np.eye(4, 3, dtype=np.float32) * 10.0
        r = ConnectomeGraphRenderer.from_matrix(cm, centroids=centroids)
        
        # Test explode with default internal centroids fallback
        global_centroid = np.zeros(3, dtype=np.float32)
        r.explode(0.5, 10.0, global_centroid)
        # Verify nodes are displaced
        assert not np.allclose(r._centroids, centroids)
        
        # Test explode with custom centroids passed to explode
        custom_centroids = {
            1: np.array([10.0, 0.0, 0.0], dtype=np.float32),
            2: np.array([0.0, 10.0, 0.0], dtype=np.float32),
            3: np.array([0.0, 0.0, 10.0], dtype=np.float32),
            4: np.array([10.0, 10.0, 10.0], dtype=np.float32),
        }
        r.explode(1.0, 50.0, global_centroid, custom_centroids)
        
        # Verify displacement matches expected direction using custom centroids
        for i in range(4):
            lbl = i + 1
            direction = custom_centroids[lbl] - global_centroid
            norm = np.linalg.norm(direction)
            expected_pos = centroids[i] + (direction / norm) * 50.0
            assert np.allclose(r._centroids[i], expected_pos)
