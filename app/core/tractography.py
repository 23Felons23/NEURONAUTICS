"""Tractography domain object.

A ``TractographyBundle`` stores a list of streamlines and provides
colour-by-direction functionality used by the VTK renderer.
"""

from __future__ import annotations

import numpy as np


class TractographyBundle:
    """A collection of white-matter streamlines.

    Parameters
    ----------
    streamlines:
        Sequence of ``(N_i, 3)`` float32 arrays of world-space (mm)
        coordinates.  Each element is one streamline; ``N_i`` may differ
        between streamlines.

    Attributes
    ----------
    streamlines:
        Stored as a plain ``list`` for VTK compatibility.
    """

    def __init__(self, streamlines: list[np.ndarray], affine: np.ndarray | None = None) -> None:
        self.streamlines: list[np.ndarray] = [
            np.asarray(sl, dtype=np.float32) for sl in streamlines
        ]
        self.affine: np.ndarray | None = affine

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_streamlines(self) -> int:
        """Return the number of streamlines in the bundle."""
        return len(self.streamlines)

    @property
    def n_points_total(self) -> int:
        """Return the total number of points across all streamlines."""
        return sum(len(sl) for sl in self.streamlines)

    # ------------------------------------------------------------------
    # Colour helpers
    # ------------------------------------------------------------------

    def color_by_direction(self) -> list[np.ndarray]:
        """Compute per-point RGB colours from local tangent direction.

        For each point in a streamline the colour is derived from the
        absolute value of the normalised tangent vector:

            R ← |dx|,  G ← |dy|,  B ← |dz|

        This is the standard DTI-style colour convention (red = L-R,
        green = A-P, blue = I-S).

        Returns
        -------
        list[np.ndarray]
            One ``(N_i, 3)`` uint8 array per streamline, matching the
            point count of the corresponding entry in ``self.streamlines``.
        """
        colours: list[np.ndarray] = []

        for sl in self.streamlines:
            n = len(sl)
            if n == 0:
                colours.append(np.zeros((0, 3), dtype=np.uint8))
                continue

            # Tangent vectors: forward differences padded at the end
            tangents = np.empty_like(sl)
            if n > 1:
                tangents[:-1] = sl[1:] - sl[:-1]
                tangents[-1] = tangents[-2]          # repeat last tangent
            else:
                tangents[0] = [0.0, 0.0, 0.0]       # single point → black

            # Normalise (avoid division by zero)
            norms = np.linalg.norm(tangents, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            tangents /= norms

            rgb_f = np.abs(tangents) * 255.0
            colours.append(rgb_f.clip(0, 255).astype(np.uint8))

        return colours

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.streamlines)

    def __repr__(self) -> str:
        n_pts = sum(len(sl) for sl in self.streamlines)
        return (
            f"TractographyBundle("
            f"n_streamlines={len(self)}, "
            f"n_points={n_pts})"
        )
