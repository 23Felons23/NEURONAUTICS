import numpy as np
import nibabel as nib
from vtkmodules.vtkRenderingCore import vtkActor
from vtkmodules.vtkCommonDataModel import vtkPolyData

from app.viz.tumor_overlay import TumorOverlayRenderer


def test_tumor_overlay_renderer(tmp_path):
    # Create a dummy binary mask NIfTI
    data = np.zeros((20, 20, 20), dtype=np.uint8)
    # A small cube in the center
    data[5:15, 5:15, 5:15] = 1
    
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(img, mask_path)
    
    renderer = TumorOverlayRenderer.from_nifti(str(mask_path))
    
    # Check that actor is a vtkActor
    assert isinstance(renderer.actor, vtkActor)
    
    # Verify the mapper has the right data type
    poly_data = renderer.actor.GetMapper().GetInput()
    assert isinstance(poly_data, vtkPolyData)
    
    # The cube contains 1000 voxels, all should be points (max 1500)
    assert poly_data.GetNumberOfPoints() == 1000
    # In a branching tree, V vertices will have roughly V-1 edges
    assert poly_data.GetNumberOfLines() > 900


def test_tumor_overlay_renderer_custom_params(tmp_path):
    # Create a dummy binary mask NIfTI
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[2:8, 2:8, 2:8] = 1
    
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    
    mask_path = tmp_path / "mask_custom.nii.gz"
    nib.save(img, mask_path)
    
    custom_color = (0.5, 0.6, 0.7)
    custom_opacity = 0.85
    custom_line_width = 4.2
    custom_num_nodes = 50
    
    renderer = TumorOverlayRenderer.from_nifti(
        str(mask_path),
        color=custom_color,
        opacity=custom_opacity,
        line_width=custom_line_width,
        num_nodes=custom_num_nodes
    )
    
    import pytest
    prop = renderer.actor.GetProperty()
    assert prop.GetColor() == custom_color
    assert prop.GetOpacity() == pytest.approx(custom_opacity)
    assert prop.GetLineWidth() == pytest.approx(custom_line_width)
    
    poly_data = renderer.actor.GetMapper().GetInput()
    assert poly_data.GetNumberOfPoints() == 50

