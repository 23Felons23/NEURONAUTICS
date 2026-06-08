"""Connectome graph renderer — 3D node–edge visualization.

Renders parcellation regions as degree-coloured spheres and the
strongest structural connections as semi-transparent cylinder edges.
"""

from __future__ import annotations

import numpy as np
import vtkmodules.all as vtk  # type: ignore

from app.core.connectome import ConnectomeMatrix
from app.io.nifti_loader import load_nifti


def _fibonacci_sphere(n: int, radius: float = 80.0) -> np.ndarray:
    """Return N points uniformly distributed on a sphere of *radius* mm.

    Uses the golden-angle (Fibonacci) method for maximum separation.
    """
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    golden = (1 + np.sqrt(5)) / 2
    indices = np.arange(n, dtype=np.float64)
    theta = 2 * np.pi * indices / golden
    phi = np.arccos(1 - 2 * (indices + 0.5) / n)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z]).astype(np.float32)


def centroids_from_parcellation(
    parc_path: "str",
    n_nodes: int,
) -> np.ndarray:
    """Compute parcel centroid world-coordinates from a parcellation volume.

    Supports ``.nii``, ``.nii.gz`` (via nibabel) and ``.mif`` (via
    ``nifti_loader``, which tries ``mrconvert`` first then a built-in parser).
    Each voxel value is an integer parcel index; 0 is background.

    The returned array has exactly *n_nodes* rows, ordered by parcel label
    1..n_nodes (MRtrix ``tck2connectome`` uses 1-based parcel labels).
    Parcels missing from the image fall back to the brain centroid.

    Parameters
    ----------
    parc_path:
        Path to the parcellation image.
    n_nodes:
        Expected number of nodes (= connectome matrix size).

    Returns
    -------
    np.ndarray — (n_nodes, 3) float32 world-space centroid array.
    """
    import os

    if not os.path.exists(parc_path):
        raise FileNotFoundError(f"Parcellation file not found: {parc_path}")

    vol = load_nifti(parc_path)
    data = np.asarray(vol.data, dtype=np.int32)
    affine = np.asarray(vol.affine, dtype=np.float64)

    # Unique parcel labels, excluding background (0)
    labels = np.unique(data)
    labels = labels[labels > 0]

    # Centre-of-mass per label in voxel space → world space
    centroids_world: dict[int, np.ndarray] = {}
    for lbl in labels:
        coords = np.argwhere(data == lbl).astype(np.float64)
        com_ijk = coords.mean(axis=0)
        world_xyz = (affine @ np.append(com_ijk, 1.0))[:3]
        centroids_world[int(lbl)] = world_xyz.astype(np.float32)

    all_pos = np.array(list(centroids_world.values()), dtype=np.float32)
    fallback = all_pos.mean(axis=0) if len(all_pos) > 0 else np.zeros(3, dtype=np.float32)

    result = np.tile(fallback, (n_nodes, 1))
    for node_idx in range(n_nodes):
        lbl = node_idx + 1   # MRtrix 1-based parcel labels
        if lbl in centroids_world:
            result[node_idx] = centroids_world[lbl]

    return result.astype(np.float32)


class ConnectomeGraphRenderer:

    """Renders a structural connectome as a 3D node–edge graph.

    Nodes are spheres colour-mapped by connection degree (blue → red).
    Edges are grey cylinders drawn for the top-K strongest connections.

    Parameters
    ----------
    matrix:
        The connectome adjacency matrix.
    centroids:
        (N, 3) float32 array of parcel centroid positions in world mm.
        If *None*, Fibonacci sphere positions are used as a layout fallback.
    top_k_edges:
        Maximum number of strongest edges to draw (default 200).

    Attributes
    ----------
    actors:
        Dict with keys ``"connectome_nodes"`` and ``"connectome_edges"``.
    """

    def __init__(
        self,
        matrix: ConnectomeMatrix,
        centroids: np.ndarray | None = None,
        top_k_edges: int = 200,
    ) -> None:
        n = matrix.node_count
        if centroids is None:
            centroids = _fibonacci_sphere(n)
        self._original_centroids = np.asarray(centroids, dtype=np.float32).copy()
        self._centroids = np.asarray(centroids, dtype=np.float32)
        self.has_centroids = centroids is not None
        self._matrix = matrix
        self._top_k = top_k_edges

        degrees = matrix.degree().astype(np.float32)
        self._edges = matrix.strongest_connections(top_k_edges)

        self._node_points = None
        self._edge_mapper = None

        self.actors: dict[str, vtk.vtkActor] = {
            "connectome_nodes": self._build_node_actor(degrees),
            "connectome_edges": self._build_edge_actor(self._edges),
        }

    def explode(
        self,
        factor: float,
        max_dist: float,
        global_centroid: np.ndarray,
        centroids: dict[int, np.ndarray] | None = None,
    ) -> None:
        """Radially displace the connectome nodes and edges for an exploded view."""
        if len(self._original_centroids) == 0:
            return

        displaced_centroids = self._original_centroids.copy()

        for node_idx in range(len(self._original_centroids)):
            lbl = node_idx + 1  # 1-based parcel labels
            
            centroid = None
            if centroids is not None:
                centroid = centroids.get(lbl)
            if centroid is None:
                centroid = self._original_centroids[node_idx]

            direction = centroid - global_centroid
            norm = np.linalg.norm(direction)
            if norm > 0.001:
                direction_normalized = direction / norm
            else:
                direction_normalized = np.zeros(3, dtype=np.float32)

            displaced_centroids[node_idx] = self._original_centroids[node_idx] + direction_normalized * (factor * max_dist)

        self._centroids = displaced_centroids
        
        # 1. Update node positions
        if self._node_points is not None:
            from vtkmodules.util import numpy_support
            vtk_data = numpy_support.numpy_to_vtk(np.ascontiguousarray(self._centroids, dtype=np.float32), deep=True)
            self._node_points.SetData(vtk_data)
            self._node_points.Modified()
            
        # 2. Rebuild edge polydata and update the mapper
        if self._edge_mapper is not None:
            new_append = self._create_edges_append_filter(self._edges)
            self._edge_mapper.SetInputConnection(new_append.GetOutputPort())

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_matrix(
        cls,
        matrix: ConnectomeMatrix,
        centroids: np.ndarray | None = None,
        top_k_edges: int = 200,
    ) -> "ConnectomeGraphRenderer":
        """Create a ``ConnectomeGraphRenderer`` from a ``ConnectomeMatrix``."""
        return cls(matrix, centroids=centroids, top_k_edges=top_k_edges)

    # ------------------------------------------------------------------
    # Node actor — degree-coloured vtkGlyph3D spheres
    # ------------------------------------------------------------------

    def _build_node_actor(self, degrees: np.ndarray) -> vtk.vtkActor:
        """Degree-coloured sphere glyphs at parcel centroid positions."""
        from vtkmodules.util import numpy_support  # type: ignore

        vtk_pts = vtk.vtkPoints()
        if self._centroids.shape[0] > 0:
            vtk_pts.SetData(numpy_support.numpy_to_vtk(self._centroids, deep=True))
        
        self._node_points = vtk_pts

        pd = vtk.vtkPolyData()
        pd.SetPoints(vtk_pts)

        if degrees.size > 0 and self._centroids.shape[0] > 0:
            vtk_scalars = numpy_support.numpy_to_vtk(degrees, deep=True)
            vtk_scalars.SetName("degree")
            pd.GetPointData().SetScalars(vtk_scalars)

            verts = vtk.vtkCellArray()
            for i in range(self._centroids.shape[0]):
                verts.InsertNextCell(1)
                verts.InsertCellPoint(i)
            pd.SetVerts(verts)

        sphere_src = vtk.vtkSphereSource()
        sphere_src.SetRadius(2.5)
        sphere_src.SetPhiResolution(8)
        sphere_src.SetThetaResolution(8)
        sphere_src.Update()

        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(pd)
        glyph.SetSourceConnection(sphere_src.GetOutputPort())
        glyph.SetScaleModeToDataScalingOff()
        glyph.SetColorModeToColorByScalar()
        glyph.Update()

        # Blue (low degree) → yellow (mid) → red (hub)
        d_min = float(degrees.min()) if degrees.size > 0 else 0.0
        d_max = float(degrees.max()) if degrees.size > 0 else 1.0

        lut = vtk.vtkColorTransferFunction()
        lut.AddRGBPoint(d_min,                   0.10, 0.20, 0.80)
        lut.AddRGBPoint((d_min + d_max) / 2.0,  0.90, 0.90, 0.20)
        lut.AddRGBPoint(d_max,                   0.90, 0.10, 0.10)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(d_min, d_max)
        mapper.ScalarVisibilityOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        return actor

    # ------------------------------------------------------------------
    # Edge actor — grey cylinders for top-K connections
    # ------------------------------------------------------------------

    def _create_edges_append_filter(self, edges: list[tuple[int, int, float]]) -> vtk.vtkAppendPolyData:
        """Create the append filter that merges all edge cylinder polydata."""
        append = vtk.vtkAppendPolyData()
        centroids = self._centroids
        has_any = False

        if len(edges) == 0 or centroids.shape[0] == 0:
            append.AddInputData(vtk.vtkPolyData())
        else:
            max_w = max((w for _, _, w in edges), default=1.0)
            if max_w == 0:
                max_w = 1.0

            for i, j, weight in edges:
                if i >= len(centroids) or j >= len(centroids):
                    continue
                p1 = centroids[i].astype(np.float64)
                p2 = centroids[j].astype(np.float64)
                diff = p2 - p1
                length = float(np.linalg.norm(diff))
                if length < 1e-6:
                    continue

                mid = (p1 + p2) / 2.0
                radius = 0.3 + 1.7 * (weight / max_w)

                cyl = vtk.vtkCylinderSource()
                cyl.SetRadius(radius)
                cyl.SetHeight(length)
                cyl.SetResolution(6)
                cyl.Update()

                # Rotate default Y-axis cylinder to align with edge direction
                direction = diff / length
                axis_y = np.array([0.0, 1.0, 0.0])
                cross = np.cross(axis_y, direction)
                cross_norm = float(np.linalg.norm(cross))
                dot = float(np.dot(axis_y, direction))
                angle_deg = float(np.degrees(np.arctan2(cross_norm, dot)))

                transform = vtk.vtkTransform()
                transform.Translate(*mid)
                if cross_norm > 1e-6:
                    transform.RotateWXYZ(angle_deg, *(cross / cross_norm))
                elif dot < 0:
                    transform.RotateX(180.0)

                tf_filter = vtk.vtkTransformPolyDataFilter()
                tf_filter.SetInputConnection(cyl.GetOutputPort())
                tf_filter.SetTransform(transform)
                tf_filter.Update()

                append.AddInputData(tf_filter.GetOutput())
                has_any = True

        if not has_any:
            append.AddInputData(vtk.vtkPolyData())

        append.Update()
        return append

    def _build_edge_actor(
        self,
        edges: list[tuple[int, int, float]],
    ) -> vtk.vtkActor:
        """Semi-transparent grey cylinder for each top-K edge."""
        append = self._create_edges_append_filter(edges)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(append.GetOutputPort())
        mapper.ScalarVisibilityOff()
        self._edge_mapper = mapper

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.7, 0.7, 0.7)
        actor.GetProperty().SetOpacity(0.6)
        return actor
