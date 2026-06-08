import numpy as np
from scipy.spatial import cKDTree
from app.core.mesh import SurfaceMesh
from app.core.volume import Volume
from app.core.spatial import apply_affine

def map_points_to_parcellation(points: np.ndarray, parcellation_vol: Volume) -> np.ndarray:
    """
    Map world-space coordinates to the nearest non-zero voxel label in a parcellation volume.

    Args:
        points: (N, 3) array of world-space coordinates.
        parcellation_vol: The Volume containing categorical integer labels.

    Returns:
        (N,) array of integer labels.
    """
    data = parcellation_vol.data
    nonzero_indices = np.argwhere(data > 0)

    if len(nonzero_indices) == 0:
        return np.zeros(len(points), dtype=int)

    nonzero_labels = data[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]]
    world_coords = apply_affine(parcellation_vol.affine, nonzero_indices)

    tree = cKDTree(world_coords)
    _, nearest_idx = tree.query(points, k=1)

    return np.round(nonzero_labels[nearest_idx]).astype(int)


def map_parcellation_to_mesh(mesh: SurfaceMesh, parcellation_vol: Volume) -> SurfaceMesh:
    """
    Map voxel values from a parcellation volume to the vertices of a surface mesh.

    Instead of direct sampling, this uses a cKDTree to map every mesh vertex to the
    strictly nearest NON-ZERO voxel in the parcellation volume. The distance is
    calculated in world space to account for anisotropic voxel sizes.

    Args:
        mesh: The input SurfaceMesh with coordinates in world space.
        parcellation_vol: The Volume containing categorical integer labels.

    Returns:
        A new SurfaceMesh instance with the `labels` attribute populated.
    """
    labels = map_points_to_parcellation(mesh.vertices, parcellation_vol)

    return SurfaceMesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        labels=labels
    )
