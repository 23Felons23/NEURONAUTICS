import numpy as np
import pytest
from app.core.mesh import SurfaceMesh
from app.viz.mesh_renderer import MeshRenderer

def create_dummy_mesh(with_labels=False):
    # Two distinct triangles
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],  # Triangle 1
        [2, 0, 0], [3, 0, 0], [2, 1, 0],  # Triangle 2
    ], dtype=np.float32)
    
    faces = np.array([
        [3, 0, 1, 2],  # pyvista format for triangle: 3 vertices, v0, v1, v2
        [3, 3, 4, 5],
    ], dtype=np.int32)
    
    if with_labels:
        # Triangle 1 gets label 1, Triangle 2 gets label 2
        labels = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)
    else:
        labels = None
        
    return SurfaceMesh(vertices=vertices, faces=faces, labels=labels)

def test_mesh_renderer_single_actor_no_labels():
    mesh = create_dummy_mesh(with_labels=False)
    renderer = MeshRenderer.from_mesh(mesh)
    
    assert len(renderer.actors) == 1
    assert "surface_mesh" in renderer.actors

def test_mesh_renderer_splits_by_labels():
    mesh = create_dummy_mesh(with_labels=True)
    renderer = MeshRenderer.from_mesh(mesh)
    
    # We should have two actors because there are two distinct labels (1 and 2)
    assert len(renderer.actors) == 2
    assert "surface_mesh_1" in renderer.actors
    assert "surface_mesh_2" in renderer.actors

def test_mesh_renderer_explode():
    mesh = create_dummy_mesh(with_labels=True)
    renderer = MeshRenderer.from_mesh(mesh)
    
    actor_1 = renderer.actors["surface_mesh_1"]
    actor_2 = renderer.actors["surface_mesh_2"]
    
    # Initial positions should be 0,0,0
    pos1_initial = actor_1.GetPosition()
    pos2_initial = actor_2.GetPosition()
    assert pos1_initial == (0.0, 0.0, 0.0)
    assert pos2_initial == (0.0, 0.0, 0.0)
    
    # Explode
    renderer.explode(factor=0.5, max_dist=10.0)
    
    pos1_exploded = actor_1.GetPosition()
    pos2_exploded = actor_2.GetPosition()
    
    # They should have moved
    assert pos1_exploded != (0.0, 0.0, 0.0)
    assert pos2_exploded != (0.0, 0.0, 0.0)
    
    # And they should have moved in different directions
    assert pos1_exploded != pos2_exploded


def test_mesh_renderer_custom_color():
    mesh = create_dummy_mesh(with_labels=False)
    custom_color = (0.2, 0.4, 0.6)
    renderer = MeshRenderer.from_mesh(mesh, color=custom_color)
    
    actor = renderer.actors["surface_mesh"]
    prop = actor.GetProperty()
    color = prop.GetColor()
    assert pytest.approx(color[0]) == 0.2
    assert pytest.approx(color[1]) == 0.4
    assert pytest.approx(color[2]) == 0.6


def test_mesh_renderer_custom_lut():
    mesh = create_dummy_mesh(with_labels=True)
    custom_lut = {
        "1": (0.8, 0.1, 0.2)
    }
    renderer = MeshRenderer.from_mesh(mesh, lut=custom_lut)
    
    actor_1 = renderer.actors["surface_mesh_1"]
    prop_1 = actor_1.GetProperty()
    color_1 = prop_1.GetColor()
    assert pytest.approx(color_1[0]) == 0.8
    assert pytest.approx(color_1[1]) == 0.1
    assert pytest.approx(color_1[2]) == 0.2
    
    actor_2 = renderer.actors["surface_mesh_2"]
    prop_2 = actor_2.GetProperty()
    color_2 = prop_2.GetColor()
    assert color_2 != (0.8, 0.1, 0.2)
