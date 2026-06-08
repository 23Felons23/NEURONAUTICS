"""Unit tests for ConnectomeMatrix domain object."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.connectome import ConnectomeMatrix


def _make_matrix(n: int = 4) -> ConnectomeMatrix:
    rng = np.random.default_rng(0)
    w = rng.uniform(0, 10, (n, n)).astype(np.float32)
    w = (w + w.T) / 2   # symmetrise
    np.fill_diagonal(w, 0.0)
    return ConnectomeMatrix(weights=w, labels=[f"R{i}" for i in range(n)])


class TestConnectomeMatrix:
    def test_node_count(self):
        cm = _make_matrix(5)
        assert cm.node_count == 5

    def test_default_labels(self):
        w = np.zeros((3, 3), dtype=np.float32)
        cm = ConnectomeMatrix(weights=w)
        assert cm.labels == ["0", "1", "2"]

    def test_threshold_zeros_weak_edges(self):
        w = np.array([[0, 1, 5], [1, 0, 3], [5, 3, 0]], dtype=np.float32)
        cm = ConnectomeMatrix(weights=w, labels=["a", "b", "c"])
        thresholded = cm.threshold(4.0)
        assert thresholded.weights[0, 1] == pytest.approx(0.0)   # 1 < 4 → zeroed
        assert thresholded.weights[0, 2] == pytest.approx(5.0)   # 5 ≥ 4 → kept

    def test_threshold_does_not_mutate_original(self):
        w = np.array([[0, 2], [2, 0]], dtype=np.float32)
        cm = ConnectomeMatrix(weights=w)
        cm.threshold(10.0)          # would zero everything
        assert cm.weights[0, 1] == pytest.approx(2.0)  # original unchanged

    def test_degree_row_sums(self):
        w = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=np.float32)
        cm = ConnectomeMatrix(weights=w)
        deg = cm.degree()
        assert deg == pytest.approx([3.0, 4.0, 5.0])

    def test_strongest_connections_sorted_descending(self):
        w = np.array([[0, 1, 5], [1, 0, 3], [5, 3, 0]], dtype=np.float32)
        cm = ConnectomeMatrix(weights=w)
        top = cm.strongest_connections(2)
        assert len(top) == 2
        assert top[0][2] == pytest.approx(5.0)   # strongest first
        assert top[1][2] == pytest.approx(3.0)

    def test_strongest_connections_upper_triangle_only(self):
        """Each edge must appear only once — no (i,j) and (j,i) duplicates."""
        cm = _make_matrix(6)
        top = cm.strongest_connections(20)
        seen: set[tuple[int, int]] = set()
        for i, j, _ in top:
            assert (i, j) not in seen and (j, i) not in seen
            seen.add((i, j))
