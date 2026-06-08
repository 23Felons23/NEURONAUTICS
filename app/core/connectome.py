"""Connectome domain object.

A ``ConnectomeMatrix`` stores a symmetric adjacency matrix representing
structural brain connectivity between cortical parcels.
"""

from __future__ import annotations

import numpy as np


class ConnectomeMatrix:
    """Symmetric adjacency matrix for structural brain connectivity.

    Parameters
    ----------
    weights:
        (N, N) float32 array of connection strengths.
    labels:
        N parcel/region label strings.  If *None*, integer strings are used.
    """

    def __init__(
        self,
        weights: np.ndarray,
        labels: list[str] | None = None,
    ) -> None:
        self.weights: np.ndarray = np.asarray(weights, dtype=np.float32)
        n = self.weights.shape[0]
        self.labels: list[str] = labels if labels is not None else [str(i) for i in range(n)]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        """Number of parcels / nodes in the connectome."""
        return self.weights.shape[0]

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def threshold(self, t: float) -> "ConnectomeMatrix":
        """Return a **new** matrix with all weights below *t* set to zero.

        The original matrix is unchanged.

        Parameters
        ----------
        t:
            Minimum connection strength to retain.
        """
        new_w = self.weights.copy()
        new_w[new_w < t] = 0.0
        return ConnectomeMatrix(weights=new_w, labels=list(self.labels))

    def degree(self) -> np.ndarray:
        """Node degree — row sums of the weight matrix.

        Returns
        -------
        np.ndarray
            (N,) float64 array of per-node connection strength totals.
        """
        return self.weights.sum(axis=1).astype(np.float64)

    def strongest_connections(self, k: int) -> list[tuple[int, int, float]]:
        """Return the top-*k* edges by weight (upper triangle only).

        Parameters
        ----------
        k:
            Maximum number of edges to return.

        Returns
        -------
        list of (i, j, weight) tuples sorted descending by weight.
        Only edges with weight > 0 are included.
        """
        rows, cols = np.triu_indices(self.node_count, k=1)
        vals = self.weights[rows, cols]
        # Sort descending, take up to k
        order = np.argsort(vals)[::-1][:k]
        return [
            (int(rows[o]), int(cols[o]), float(vals[o]))
            for o in order
            if vals[o] > 0
        ]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        preview = self.labels[:3]
        suffix = "..." if self.node_count > 3 else ""
        return (
            f"ConnectomeMatrix("
            f"nodes={self.node_count}, "
            f"labels={preview}{suffix})"
        )
