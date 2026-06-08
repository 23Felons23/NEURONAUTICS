import numpy as np
import pytest
from app.core.mesh import SurfaceMesh
from app.core.volume import Volume
from app.io.parcellation_mapper import map_parcellation_to_mesh

def test_map_parcellation_to_mesh_assigns_labels():
    # Create a small dummy volume (3x3x3)
    data = np.zeros((3, 3, 3), dtype=np.float32)
    # Assign specific labels to specific voxels
    data[1, 1, 1] = 5.0
    data[2, 2, 2] = 10.0
    
    # Simple identity affine
    affine = np.eye(4)
    vol = Volume(data=data, affine=affine, path="dummy.nii")
    
    # Create a mesh with vertices that land exactly on those voxels
    # Since affine is identity, world coords = voxel coords
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ])
    faces = np.array([[0, 1, 2]])
    mesh = SurfaceMesh(vertices=vertices, faces=faces)
    
    # Map parcellation
    labeled_mesh = map_parcellation_to_mesh(mesh, vol)
    
    # Assertions
    assert labeled_mesh.labels is not None
    assert len(labeled_mesh.labels) == 3
    # With cKDTree snapping to nearest NON-ZERO voxel:
    # [0,0,0] -> nearest is [1,1,1] (label 5)
    # [1,1,1] -> nearest is [1,1,1] (label 5)
    # [2,2,2] -> nearest is [2,2,2] (label 10)
    assert labeled_mesh.labels[0] == 5
    assert labeled_mesh.labels[1] == 5
    assert labeled_mesh.labels[2] == 10

def test_map_parcellation_to_mesh_clips_bounds():
    # Volume is 2x2x2
    data = np.zeros((2, 2, 2), dtype=np.float32)
    data[1, 1, 1] = 7.0
    vol = Volume(data=data, affine=np.eye(4), path="dummy.nii")
    
    # Vertices outside bounds (e.g. 5,5,5 and -1,-1,-1)
    vertices = np.array([
        [5.0, 5.0, 5.0],
        [-1.0, -1.0, -1.0],
    ])
    faces = np.array([[0, 1, 0]])
    mesh = SurfaceMesh(vertices=vertices, faces=faces)
    
    labeled_mesh = map_parcellation_to_mesh(mesh, vol)
    
    assert labeled_mesh.labels is not None
    # Should safely snap to the only non-zero voxel (7) for both
    assert np.all(labeled_mesh.labels == 7)
