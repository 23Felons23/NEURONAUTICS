"""Unit tests for UI/UX components."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QWidget, QCheckBox

from app.scene.mode_controller import ModeController
from app.scene.scene_manager import SceneManager
from app.ui.hand_tracking_panel import HandTrackingPanel
from app.ui.mode_selector import ModeSelector


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_mode_selector_toggles_subwidget_enablement(qapp):
    scene = SceneManager()
    controller = ModeController(scene)
    controller.register("voxel_model", ["voxel_model"])

    sub_widget = QWidget()
    sub_widgets = {
        "voxel_model": sub_widget
    }

    selector = ModeSelector(controller, sub_widgets)
    
    # Locate Voxel Model checkbox
    # Title is based on: label = mode.replace("_", " ").title() -> "Voxel Model"
    checkboxes = selector.findChildren(QCheckBox)
    voxel_cb = None
    for cb in checkboxes:
        if cb.text() == "Voxel Model":
            voxel_cb = cb
            break

    assert voxel_cb is not None
    assert voxel_cb.isChecked()
    assert sub_widget.isEnabled()

    # Toggle checkbox off
    voxel_cb.setChecked(False)
    assert not sub_widget.isEnabled()

    # Toggle back on
    voxel_cb.setChecked(True)
    assert sub_widget.isEnabled()


def test_mode_selector_register_mode_with_subwidget(qapp):
    scene = SceneManager()
    controller = ModeController(scene)
    
    selector = ModeSelector(controller, {})
    sub_widget = QWidget()
    
    selector.register_mode("new_mode", sub_widget)
    
    checkboxes = selector.findChildren(QCheckBox)
    new_cb = None
    for cb in checkboxes:
        if cb.text() == "New Mode":
            new_cb = cb
            break

    assert new_cb is not None
    assert new_cb.isChecked()
    assert sub_widget.isEnabled()

    new_cb.setChecked(False)
    assert not sub_widget.isEnabled()


def test_hand_tracking_panel_minimization(qapp):
    panel = HandTrackingPanel()
    assert not panel._minimized
    
    # Initially the feed is hidden (since enable checkbox is unchecked)
    assert panel._feed.isHidden()

    # Toggle tracking on
    panel._toggle.setChecked(True)
    assert not panel._feed.isHidden()

    # Trigger minimize
    panel._on_minimize_toggled()
    assert panel._minimized
    assert panel._feed.isHidden()
    assert panel._min_btn.text() == "+"

    # Maximize back
    panel._on_minimize_toggled()
    assert not panel._minimized
    assert not panel._feed.isHidden()
    assert panel._min_btn.text() == "−"


def test_explosion_slider_lod_and_phantom(qapp):
    from unittest.mock import MagicMock
    import numpy as np
    import vtkmodules.all as vtk
    from app.core.volume import Volume
    from app.ui.main_window import MainWindow

    # 1. Create a dummy volume
    data = np.zeros((5, 5, 5), dtype=np.float32)
    affine = np.eye(4)
    volume = Volume(data, affine)

    # 2. Create mock renderers with required properties
    mock_tract = MagicMock()
    
    # Configure mock voxel renderer with actual VTK actor & mapper to prevent scalar range unpack error
    mock_voxel = MagicMock()
    mock_voxel.min_intensity = 0.0
    mock_voxel.max_intensity = 100.0
    voxel_actor = vtk.vtkActor()
    voxel_mapper = vtk.vtkPolyDataMapper()
    voxel_actor.SetMapper(voxel_mapper)
    mock_voxel.actors = {"voxel_model": voxel_actor}
    
    mock_connectome = MagicMock()
    
    mock_mesh = MagicMock()
    mock_mesh.centroids = {"surface_mesh_1": np.array([10.0, 10.0, 10.0])}
    mock_mesh.global_centroid = np.array([0.0, 0.0, 0.0])
    
    mesh_actor = vtk.vtkActor()
    mesh_mapper = vtk.vtkPolyDataMapper()
    mesh_actor.SetMapper(mesh_mapper)
    mock_mesh.actors = {"surface_mesh_1": mesh_actor}

    # Create the main window instance
    window = MainWindow(
        volume=volume,
        tract_renderer=mock_tract,
        voxel_renderer=mock_voxel,
        connectome_renderer=mock_connectome,
        mesh_renderer=mock_mesh,
    )

    # Initial states
    assert not window._is_dragging_explode
    assert not window._phantom_actors

    # 3. Test dragging (LOD preview active)
    window._on_explode_slider_pressed()
    assert window._is_dragging_explode
    assert "surface_mesh_1" in window._phantom_actors
    
    phantom = window._phantom_actors["surface_mesh_1"]
    assert phantom is not None
    # Check that color is solid red (1.0, 0.0, 0.0)
    color = phantom.GetProperty().GetColor()
    assert color == (1.0, 0.0, 0.0)
    # Check that opacity is 0.35
    assert abs(phantom.GetProperty().GetOpacity() - 0.35) < 1e-4

    # Reset calls to verify deferral
    mock_tract.explode.reset_mock()
    mock_voxel.explode.reset_mock()
    mock_connectome.explode.reset_mock()

    # Trigger slider move by setting the slider value
    window._explode_slider.setValue(50)

    # Verify that heavy CPU updates are deferred
    mock_tract.explode.assert_not_called()
    mock_voxel.explode.assert_not_called()
    mock_connectome.explode.assert_not_called()

    # Verify that phantom actor has translated, but the original actor is still stationary at (0, 0, 0)
    p_pos = phantom.GetPosition()
    orig_pos = mesh_actor.GetPosition()
    assert orig_pos == (0.0, 0.0, 0.0)
    assert p_pos != (0.0, 0.0, 0.0)

    # 4. Test slider release (high-fidelity sync)
    window._on_explode_slider_released()
    assert not window._is_dragging_explode
    assert not window._phantom_actors  # Phantom actors cleaned up

    # Verify that final sync occurs
    mock_tract.explode.assert_called_once()
    mock_voxel.explode.assert_called_once()
    mock_connectome.explode.assert_called_once()

    # Verify that original mesh actor has now snapped to its exploded position
    orig_pos_after = mesh_actor.GetPosition()
    assert orig_pos_after != (0.0, 0.0, 0.0)

    window.close()


def test_camera_orientation_alignment_with_tilted_affine(qapp):
    import numpy as np
    from app.core.volume import Volume
    from app.ui.main_window import MainWindow

    # Create a volume with a tilted/permuted affine matrix.
    # Column 0 (voxel X): [0.01, -0.99, -0.09]
    # Column 1 (voxel Y): [0.01, 0.09, -0.99]
    # Column 2 (voxel Z): [0.99, 0.01, 0.01]
    affine = np.array([
        [ 0.01,  0.01,  0.99, 0.0],
        [-0.99,  0.09,  0.01, 0.0],
        [-0.09, -0.99,  0.01, 0.0],
        [ 0.0,   0.0,   0.0,  1.0]
    ])
    data = np.zeros((5, 5, 5), dtype=np.float32)
    volume = Volume(data, affine)

    window = MainWindow(volume=volume)
    camera = window._scene.renderer.GetActiveCamera()

    # Expected positions:
    # dirs is same as upper 3x3 of affine (spacing is 1.0)
    # col_r = 2, col_a = 0, col_s = 1
    # r_vec = [0.99, 0.01, 0.01] * sign(0.99) = [0.99, 0.01, 0.01]
    # s_vec = [0.01, 0.09, -0.99] * sign(-0.99) = [-0.01, -0.09, 0.99]
    expected_pos = (-0.99, -0.01, -0.01)
    expected_up = (-0.01, -0.09, 0.99)

    pos = np.array(camera.GetPosition())
    fp = np.array(camera.GetFocalPoint())
    look_dir = fp - pos
    look_dir = look_dir / np.linalg.norm(look_dir)

    expected_look = -np.array(expected_pos) / np.linalg.norm(expected_pos)
    np.testing.assert_allclose(look_dir, expected_look, atol=1e-3)

    up = np.array(camera.GetViewUp())
    expected_up_norm = np.array(expected_up) / np.linalg.norm(expected_up)
    np.testing.assert_allclose(up, expected_up_norm, atol=1e-3)

    window.close()


def test_parcellation_warning_label_visibility(qapp):
    from unittest.mock import MagicMock
    import numpy as np
    from app.core.volume import Volume
    from app.ui.main_window import MainWindow
    from PySide6.QtWidgets import QLabel

    # Setup dummy volume
    data = np.zeros((5, 5, 5), dtype=np.float32)
    volume = Volume(data, np.eye(4))

    # Test 1: Connectome renderer is missing parcellation centroids
    mock_conn_fallback = MagicMock()
    mock_conn_fallback.has_centroids = False
    mock_conn_fallback.actors = {}

    window_fallback = MainWindow(volume=volume, connectome_renderer=mock_conn_fallback)
    labels = window_fallback.findChildren(QLabel)
    warning_label_found = False
    for lbl in labels:
        if "Warning: Parcellation data is missing" in lbl.text():
            warning_label_found = True
            break
    assert warning_label_found, "Warning label should be displayed when parcellation centroids are missing"
    window_fallback.close()

    # Test 2: Connectome renderer HAS parcellation centroids
    mock_conn_ok = MagicMock()
    mock_conn_ok.has_centroids = True
    mock_conn_ok.actors = {}

    window_ok = MainWindow(volume=volume, connectome_renderer=mock_conn_ok)
    labels_ok = window_ok.findChildren(QLabel)
    warning_label_found_ok = False
    for lbl in labels_ok:
        if "Warning: Parcellation data is missing" in lbl.text():
            warning_label_found_ok = True
            break
    assert not warning_label_found_ok, "Warning label should NOT be displayed when parcellation centroids are present"
    window_ok.close()



