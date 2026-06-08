"""Voxel model renderer — discrete cube grid for a T1 volume.

Renders one cube per sampled voxel above threshold using ``vtkGlyph3D``.
Sampling stride and intensity threshold are configurable to manage performance
for large volumes (e.g. 256³ T1 data).
"""

from __future__ import annotations

import numpy as np
import vtkmodules.all as vtk  # type: ignore

from app.core.volume import Volume


class VoxelModelRenderer:
    """Renders a stride-decimated voxel point cloud for a brain Volume.

    Renders small point dots with grayscale MRI intensity mapping. Supports
    parcellation-based exploded view using a single-actor architecture
    with in-place point coordinate modification.

    Parameters
    ----------
    volume:
        The T1 brain volume.
    stride:
        Sample every ``stride`` voxels along each axis.
        stride=1 renders every voxel (very slow for large volumes).
        stride=4 (default) gives ~(N/4)³ points — fast enough for interaction.
    threshold_percentile:
        Voxels below this intensity percentile (of non-zero values) are
        excluded.  Default=5.0 removes background noise.
    parc_vol:
        Optional parcellation Volume for exploded view.

    Attributes
    ----------
    actors:
        Dict mapping ``"voxel_model"`` to the ready-to-use ``vtkActor``.
    global_centroid:
        Mean of all points (RAS mm space).
    """

    def __init__(
        self,
        volume: Volume,
        stride: int = 4,
        threshold_percentile: float = 5.0,
        parc_vol: Volume | None = None,
        point_size: float = 3.0,
        black_value: float | None = None,
        white_value: float | None = None,
        render_spheres: bool = False,
    ) -> None:
        self._volume = volume
        self._stride = max(1, int(stride))

        # Compute intensity threshold from non-zero voxels
        nz = volume.data[volume.data > 0]
        self._threshold = float(np.percentile(nz, threshold_percentile)) if nz.size > 0 else 0.0

        points, scalars = self._sample_voxels()
        self._original_points = points.copy()
        
        if scalars.size > 0:
            self.min_intensity = float(scalars.min())
            self.max_intensity = float(scalars.max())
        else:
            self.min_intensity = 0.0
            self.max_intensity = 1.0

        # Map labels if parcellation volume is provided
        if parc_vol is not None and len(points) > 0:
            from app.io.parcellation_mapper import map_points_to_parcellation
            self._labels = map_points_to_parcellation(points, parc_vol)
            
            unique_labels = np.unique(self._labels)
            self._label_centroids = {}
            for lbl in unique_labels:
                mask = self._labels == lbl
                self._label_centroids[int(lbl)] = points[mask].mean(axis=0)
        else:
            self._labels = None
            self._label_centroids = {}

        # Label masking for deletion sync
        self._hidden_labels: set[int] = set()

        self.global_centroid = points.mean(axis=0) if len(points) > 0 else np.zeros(3, dtype=np.float32)

        # Store latest explode parameters to allow synchronization during label hiding/restoring
        self._last_explode_factor = 0.0
        self._last_max_dist = 150.0
        self._last_global_centroid = self.global_centroid
        self._last_centroids = None

        self.actors: dict[str, vtk.vtkActor] = {
            "voxel_model": self._build_point_actor(
                points,
                scalars,
                point_size=point_size,
                black_value=black_value,
                white_value=white_value,
                render_spheres=render_spheres,
            ),
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_volume(
        cls,
        volume: Volume,
        stride: int = 4,
        threshold_percentile: float = 5.0,
        parc_vol: Volume | None = None,
        point_size: float = 3.0,
        black_value: float | None = None,
        white_value: float | None = None,
        render_spheres: bool = False,
    ) -> "VoxelModelRenderer":
        """Create a ``VoxelModelRenderer`` from a ``Volume``.

        Parameters
        ----------
        volume:
            Brain volume to render.
        stride:
            Decimation stride — higher = fewer points = faster.
        threshold_percentile:
            Minimum intensity percentile; lower values include more voxels.
        parc_vol:
            Optional parcellation volume for exploded view.
        point_size:
            Initial visual point size.
        black_value:
            Initial intensity threshold mapped to black.
        white_value:
            Initial intensity threshold mapped to white.
        render_spheres:
            Whether to render points as 3D spheres.
        """
        return cls(
            volume,
            stride=stride,
            threshold_percentile=threshold_percentile,
            parc_vol=parc_vol,
            point_size=point_size,
            black_value=black_value,
            white_value=white_value,
            render_spheres=render_spheres,
        )

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
        """Displace voxel points radially by parcellation label for exploded view.

        Modifies the vtkPoints coordinates in-place (no new actors created).
        Uses the same centroid-based radial displacement as MeshRenderer but
        operates on point coordinates directly for performance.
        """
        # Store latest explode parameters to allow synchronization during label hiding/restoring
        self._last_explode_factor = factor
        self._last_max_dist = max_dist
        self._last_global_centroid = global_centroid
        self._last_centroids = centroids

        if self._labels is None or len(self._original_points) == 0:
            return

        displaced = self._original_points.copy()

        # Resolve centroids per label, falling back to internal centroids if needed
        label_centroids = centroids if centroids is not None else self._label_centroids

        for lbl in self._label_centroids:
            centroid = label_centroids.get(lbl)
            if centroid is None:
                centroid = self._label_centroids[lbl]

            direction = centroid - global_centroid
            norm = np.linalg.norm(direction)
            if norm > 0.001:
                direction = direction / norm
            else:
                continue

            mask = self._labels == lbl
            offset = direction * (factor * max_dist)
            displaced[mask] += offset

        # Apply hidden-label sentinel after displacement using NaN
        if self._hidden_labels:
            for lbl in self._hidden_labels:
                mask = self._labels == lbl
                displaced[mask] = np.nan

        from vtkmodules.util import numpy_support
        vtk_data = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(displaced, dtype=np.float32), deep=True
        )
        self._vtk_points.SetData(vtk_data)
        self._vtk_points.Modified()

    def hide_label(self, label: int) -> None:
        """Hide all voxel points belonging to *label*.

        Uses a NaN sentinel to make hidden points invisible without rebuilding the
        vtkCellArray or affecting VTK's camera clipping planes.
        Compatible with the explode() single-actor architecture.

        Parameters
        ----------
        label:
            Parcellation label integer matching a ``surface_mesh_{label}`` key.
        """
        if self._labels is None:
            return
        self._hidden_labels.add(int(label))
        # Re-run explode with stored parameters to preserve explosion state
        self.explode(
            self._last_explode_factor,
            self._last_max_dist,
            self._last_global_centroid,
            self._last_centroids,
        )

    def restore_all_labels(self) -> None:
        """Make all previously hidden labels visible again."""
        if self._labels is None:
            return
        self._hidden_labels.clear()
        # Re-run explode with stored parameters to preserve explosion state
        self.explode(
            self._last_explode_factor,
            self._last_max_dist,
            self._last_global_centroid,
            self._last_centroids,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_voxels(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(world_points, intensity_scalars)`` for sampled voxels.

        Returns
        -------
        world_points:
            (N, 3) float32 array of voxel centres in world (RAS mm) space.
        intensity_scalars:
            (N,) float32 array of MRI intensities at those positions.
        """
        data = self._volume.data      # (nx, ny, nz) float32
        affine = self._volume.affine  # (4, 4) float64
        s = self._stride

        nx, ny, nz = data.shape
        xi = np.arange(0, nx, s)
        yi = np.arange(0, ny, s)
        zi = np.arange(0, nz, s)
        ii, jj, kk = np.meshgrid(xi, yi, zi, indexing="ij")
        ii = ii.ravel()
        jj = jj.ravel()
        kk = kk.ravel()

        # Sample intensities and apply threshold mask
        intensities = data[ii, jj, kk]
        mask = intensities > self._threshold
        ii = ii[mask]
        jj = jj[mask]
        kk = kk[mask]
        intensities = intensities[mask]

        if ii.size == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)

        # Voxel indices → world coordinates via affine
        ijk_hom = np.column_stack([ii, jj, kk, np.ones(len(ii))]).T  # (4, N)
        world = (affine @ ijk_hom)[:3].T.astype(np.float32)            # (N, 3)

        return world, intensities.astype(np.float32)

    def _build_point_actor(
        self,
        world_points: np.ndarray,
        scalars: np.ndarray,
        point_size: float = 3.0,
        black_value: float | None = None,
        white_value: float | None = None,
        render_spheres: bool = False,
    ) -> vtk.vtkActor:
        """Construct a direct point cloud actor from world-space points + scalars."""
        from vtkmodules.util import numpy_support  # type: ignore

        # Build vtkPoints
        vtk_pts = vtk.vtkPoints()
        if world_points.shape[0] > 0:
            vtk_pts.SetData(numpy_support.numpy_to_vtk(world_points, deep=True))

        self._vtk_points = vtk_pts

        # Build vtkPolyData (point cloud — one point per voxel centre)
        pd = vtk.vtkPolyData()
        pd.SetPoints(vtk_pts)

        if scalars.size > 0:
            vtk_scalars = numpy_support.numpy_to_vtk(scalars, deep=True)
            vtk_scalars.SetName("intensity")
            pd.GetPointData().SetScalars(vtk_scalars)

            # Vertex cells required for point rendering
            verts = vtk.vtkCellArray()
            for i in range(world_points.shape[0]):
                verts.InsertNextCell(1)
                verts.InsertCellPoint(i)
            pd.SetVerts(verts)

        # Grayscale transfer function
        if scalars.size > 0:
            s_min = black_value if black_value is not None else float(scalars.min())
            s_max = white_value if white_value is not None else float(scalars.max())
        else:
            s_min, s_max = 0.0, 1.0

        lut = vtk.vtkColorTransferFunction()
        # Quadratic contrast curve (gamma = 2.0) to make white areas pop relative to dark
        for pct in [0.0, 0.25, 0.5, 0.75, 1.0]:
            val = s_min + pct * (s_max - s_min)
            intensity = pct ** 2.0
            lut.AddRGBPoint(val, intensity, intensity, intensity)

        self._lut = lut

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(s_min, s_max)
        mapper.ScalarVisibilityOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Point rendering properties
        prop = actor.GetProperty()
        prop.SetPointSize(point_size)
        if render_spheres:
            prop.RenderPointsAsSpheresOn()
        else:
            prop.RenderPointsAsSpheresOff()
        prop.SetOpacity(0.85)
        
        return actor

    def set_point_size(self, size: float) -> None:
        """Dynamically update visual point size on the GPU."""
        actor = self.actors.get("voxel_model")
        if actor is not None:
            actor.GetProperty().SetPointSize(size)

    def set_render_spheres(self, enabled: bool) -> None:
        """Dynamically update whether points are rendered as 3D spheres on the GPU."""
        actor = self.actors.get("voxel_model")
        if actor is not None:
            if enabled:
                actor.GetProperty().RenderPointsAsSpheresOn()
            else:
                actor.GetProperty().RenderPointsAsSpheresOff()

    def set_intensity_range(self, black_value: float, white_value: float) -> None:
        """Dynamically update intensity contrast mapping (windowing) on the GPU."""
        actor = self.actors.get("voxel_model")
        if actor is None:
            return
        mapper = actor.GetMapper()
        mapper.SetScalarRange(black_value, white_value)

        if hasattr(self, "_lut") and self._lut is not None:
            self._lut.RemoveAllPoints()
            for pct in [0.0, 0.25, 0.5, 0.75, 1.0]:
                val = black_value + pct * (white_value - black_value)
                intensity = pct ** 2.0
                self._lut.AddRGBPoint(val, intensity, intensity, intensity)

