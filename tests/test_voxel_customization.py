import json
import numpy as np
import pytest
import vtkmodules.all as vtk  # type: ignore

from app.io.manifest import load_manifest
from app.core.volume import Volume
from app.viz.voxel_model import VoxelModelRenderer


def _make_volume(
    shape: tuple[int, int, int] = (10, 10, 10),
    voxel_size_mm: float = 1.0,
    intensity_range: tuple[float, float] = (0.0, 100.0),
    seed: int = 42,
) -> Volume:
    """Return a small synthetic Volume filled with random intensities."""
    rng = np.random.default_rng(seed)
    lo, hi = intensity_range
    data = rng.uniform(lo, hi, shape).astype(np.float32)
    affine = np.eye(4, dtype=np.float64) * voxel_size_mm
    affine[3, 3] = 1.0
    return Volume(data=data, affine=affine)


def test_manifest_voxel_settings_parsing(tmp_path):
    """Verify that DatasetBundle correctly parses voxel resolution and custom styling fields."""
    manifest_data = {
        "t1": "mri/t1.nii.gz",
        "voxel_stride": 3,
        "voxel_point_size": 5.5,
        "voxel_black_value": 12.0,
        "voxel_white_value": 85.0,
        "voxel_spheres": True
    }
    manifest_file = tmp_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    bundle = load_manifest(manifest_file)
    assert bundle.voxel_stride == 3
    assert bundle.voxel_point_size == 5.5
    assert bundle.voxel_black_value == 12.0
    assert bundle.voxel_white_value == 85.0
    assert bundle.voxel_spheres is True


def test_renderer_initialization_defaults():
    """Verify that VoxelModelRenderer initializes with standard default parameters if not specified."""
    vol = _make_volume()
    renderer = VoxelModelRenderer.from_volume(vol)
    actor = renderer.actors["voxel_model"]
    prop = actor.GetProperty()

    # Defaults: point size = 3.0, spheres = False
    assert prop.GetPointSize() == 3.0
    assert prop.GetRenderPointsAsSpheres() == 0


def test_renderer_initialization_with_manifest_values():
    """Verify that VoxelModelRenderer initializes with custom settings."""
    vol = _make_volume()
    renderer = VoxelModelRenderer.from_volume(
        vol,
        point_size=6.0,
        black_value=10.0,
        white_value=90.0,
        render_spheres=True
    )
    actor = renderer.actors["voxel_model"]
    prop = actor.GetProperty()
    mapper = actor.GetMapper()

    assert prop.GetPointSize() == 6.0
    assert prop.GetRenderPointsAsSpheres() == 1
    
    # Check scalar range mapping
    s_min, s_max = mapper.GetScalarRange()
    assert s_min == 10.0
    assert s_max == 90.0


def test_renderer_dynamic_setters():
    """Verify that dynamic style modifiers update the actor and mapper on the GPU instantly."""
    vol = _make_volume()
    renderer = VoxelModelRenderer.from_volume(vol)
    actor = renderer.actors["voxel_model"]
    prop = actor.GetProperty()
    mapper = actor.GetMapper()

    # 1. Update point size
    renderer.set_point_size(8.5)
    assert prop.GetPointSize() == 8.5

    # 2. Update sphere rendering
    renderer.set_render_spheres(True)
    assert prop.GetRenderPointsAsSpheres() == 1
    renderer.set_render_spheres(False)
    assert prop.GetRenderPointsAsSpheres() == 0

    # 3. Update intensity contrast range
    renderer.set_intensity_range(25.0, 75.0)
    s_min, s_max = mapper.GetScalarRange()
    assert s_min == 25.0
    assert s_max == 75.0

    # Ensure LUT table points were rebuilt
    lut = mapper.GetLookupTable()
    # Query rebuilt LUT points
    color_val = [0.0, 0.0, 0.0]
    lut.GetColor(25.0, color_val)
    # At black value (25.0) intensity should be 0.0 (black)
    assert all(c < 1e-5 for c in color_val)
    
    lut.GetColor(75.0, color_val)
    # At white value (75.0) intensity should be 1.0 (white)
    assert all(c > 0.999 for c in color_val)


def test_setters_no_coordinate_rebuild():
    """Verify that changing point size, spheres, and contrast does not mutate original coordinates array."""
    vol = _make_volume()
    renderer = VoxelModelRenderer.from_volume(vol)
    
    orig_coords = renderer._original_points.copy()
    
    renderer.set_point_size(9.0)
    renderer.set_render_spheres(True)
    renderer.set_intensity_range(30.0, 80.0)
    
    # Coordinates array should be completely untouched
    assert np.array_equal(renderer._original_points, orig_coords)
