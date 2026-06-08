"""VTK renderer for white-matter streamlines.

Converts a ``TractographyBundle`` to a single ``vtkPolyData`` actor
with per-point direction-coloured RGB values.  Rendering is done with
``vtkPolyDataMapper``; no lighting is applied so colours appear exactly
as computed.

Supports parcellation-based exploded view via :meth:`explode` — each
streamline point is assigned a parcellation label and displaced radially
along the direction from the global centroid to its label's centroid,
matching the same displacement used by ``VoxelModelRenderer`` and
``ConnectomeGraphRenderer``.
"""

from __future__ import annotations

import numpy as np
import vtkmodules.all as vtk  # type: ignore
from vtkmodules.util import numpy_support  # type: ignore

from app.core.tractography import TractographyBundle

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.core.volume import Volume


class TractographyRenderer:
    """VTK actor for a tractogram.

    Attributes
    ----------
    actors:
        Dict mapping ``"streamlines"`` to the underlying ``vtkActor``.
        Using a dict keeps the interface consistent with ``SliceViewer``.
    """

    def __init__(self, actor: vtk.vtkActor) -> None:
        self.actors: dict[str, vtk.vtkActor] = {"streamlines": actor}
        # Parcellation explosion state — populated by from_bundle when parc_vol provided
        self._vtk_points: vtk.vtkPoints | None = None
        self._original_points: np.ndarray | None = None  # (N, 3) float32
        self._labels: np.ndarray | None = None           # (N,) int
        self._label_centroids: dict[int, np.ndarray] = {}
        # Store last explode params to support re-explosion on demand
        self._last_explode_factor: float = 0.0
        self._last_max_dist: float = 150.0
        self._last_global_centroid: np.ndarray = np.zeros(3, dtype=np.float32)
        self._last_centroids: dict[int, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_bundle(
        cls,
        bundle: TractographyBundle,
        parc_vol: "Volume | None" = None,
    ) -> "TractographyRenderer":
        """Build a ``TractographyRenderer`` from a ``TractographyBundle``.

        Parameters
        ----------
        bundle:
            The streamline bundle to render.
        parc_vol:
            Optional parcellation volume.  When provided, each streamline
            point is mapped to its nearest parcellation label so that
            :meth:`explode` can displace streamlines radially in sync with
            mesh, voxel and connectome exploded views.

        Returns
        -------
        TractographyRenderer
        """
        colours_per_sl = bundle.color_by_direction()

        # Collect all points and build cell (poly-line) connectivity
        all_points: list[np.ndarray] = []
        cell_ids: list[int] = []  # interleaved: [n_pts, id0, id1, …, n_pts, …]
        all_colours: list[np.ndarray] = []

        point_offset = 0
        for sl, rgb in zip(bundle.streamlines, colours_per_sl):
            n = len(sl)
            if n == 0:
                continue
            all_points.append(sl)
            all_colours.append(rgb)
            cell_ids.append(n)
            cell_ids.extend(range(point_offset, point_offset + n))
            point_offset += n

        # Concatenated (N, 3) float32 array — kept for explosion
        pts_array: np.ndarray | None = None
        if all_points:
            pts_array = np.concatenate(all_points, axis=0).astype(np.float32)

        # Build vtkPoints
        vtk_points = vtk.vtkPoints()
        if pts_array is not None:
            vtk_pts_arr = numpy_support.numpy_to_vtk(pts_array.astype(np.float64), deep=True)
            vtk_points.SetData(vtk_pts_arr)

        # Build vtkCellArray (poly-lines)
        vtk_cells = vtk.vtkCellArray()
        if cell_ids:
            cell_array = np.array(cell_ids, dtype=np.int64)
            vtk_id_arr = numpy_support.numpy_to_vtkIdTypeArray(cell_array, deep=True)
            vtk_cells.ImportLegacyFormat(vtk_id_arr)

        # Build RGB colour array
        vtk_colours = vtk.vtkUnsignedCharArray()
        vtk_colours.SetNumberOfComponents(3)
        vtk_colours.SetName("Direction colours")
        if all_colours:
            rgb_flat = np.concatenate(all_colours, axis=0)
            # Set colours point-by-point (simple and portable across VTK versions)
            for r, g, b in rgb_flat:
                vtk_colours.InsertNextTuple3(int(r), int(g), int(b))

        # Assemble vtkPolyData
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        poly_data.SetLines(vtk_cells)
        poly_data.GetPointData().SetScalars(vtk_colours)

        # Mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        mapper.SetScalarModeToUsePointData()
        mapper.SetColorModeToDirectScalars()
        mapper.ScalarVisibilityOn()

        # Actor — no lighting so RGB is exact
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(1.0)
        actor.GetProperty().LightingOff()

        renderer = cls(actor)

        # Store VTK points reference and original coordinates for in-place updates
        renderer._vtk_points = vtk_points
        if pts_array is not None:
            renderer._original_points = pts_array.copy()

            # Map streamline points to parcellation labels if volume provided
            if parc_vol is not None:
                from app.io.parcellation_mapper import map_points_to_parcellation
                labels = map_points_to_parcellation(pts_array, parc_vol)
                renderer._labels = labels

                # Pre-compute per-label centroids from original point positions
                unique_labels = np.unique(labels)
                for lbl in unique_labels:
                    mask = labels == lbl
                    renderer._label_centroids[int(lbl)] = pts_array[mask].mean(axis=0)

        return renderer

    # ------------------------------------------------------------------
    # Exploded view support
    # ------------------------------------------------------------------

    def explode(
        self,
        factor: float,
        max_dist: float,
        global_centroid: np.ndarray,
        centroids: dict[int, np.ndarray] | None = None,
    ) -> None:
        """Displace streamline points radially by parcellation label.

        Modifies the ``vtkPoints`` coordinates in-place (no new actors
        created).  Uses the same centroid-based radial displacement as
        ``VoxelModelRenderer`` and ``ConnectomeGraphRenderer`` so all
        visualisation layers stay in sync during exploded view.

        Parameters
        ----------
        factor:
            Explosion factor in [0, 1].  0 = assembled, 1 = fully exploded.
        max_dist:
            Maximum displacement distance in world units (mm).
        global_centroid:
            Centre of the whole brain in world space.
        centroids:
            Optional per-label centroid overrides (keyed by integer label).
            When provided, these are used to compute displacement directions
            instead of the internally computed label centroids.  Pass the
            mesh-derived centroids so all layers share the same reference.
        """
        # Store latest params for potential re-explosion
        self._last_explode_factor = factor
        self._last_max_dist = max_dist
        self._last_global_centroid = global_centroid
        self._last_centroids = centroids

        if self._labels is None or self._original_points is None or len(self._original_points) == 0:
            return  # no parcellation mapping — silent no-op

        if self._vtk_points is None:
            return

        displaced = self._original_points.copy()

        for lbl in self._label_centroids:
            # Resolve centroid: prefer external override, fall back to internal
            centroid = None
            if centroids is not None:
                centroid = centroids.get(lbl)
            if centroid is None:
                centroid = self._label_centroids[lbl]

            direction = centroid - global_centroid
            norm = np.linalg.norm(direction)
            if norm > 0.001:
                direction = direction / norm
            else:
                continue  # label centroid coincides with global centroid — skip

            mask = self._labels == lbl
            offset = direction * (factor * max_dist)
            displaced[mask] += offset

        # Update VTK points in-place — no new actor needed
        vtk_data = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(displaced, dtype=np.float64), deep=True
        )
        self._vtk_points.SetData(vtk_data)
        self._vtk_points.Modified()
