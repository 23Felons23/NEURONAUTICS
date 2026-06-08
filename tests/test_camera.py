"""Unit tests for CameraState — save / restore round-trip."""

from __future__ import annotations

import pytest
import vtkmodules.all as vtk  # type: ignore

from app.scene.camera import CameraState


class TestCameraState:
    def test_from_camera_captures_position(self):
        cam = vtk.vtkCamera()
        cam.SetPosition(1.0, 2.0, 3.0)
        state = CameraState.from_camera(cam)
        assert state.position == pytest.approx((1.0, 2.0, 3.0))

    def test_from_camera_captures_focal_point(self):
        cam = vtk.vtkCamera()
        cam.SetFocalPoint(4.0, 5.0, 6.0)
        state = CameraState.from_camera(cam)
        assert state.focal_point == pytest.approx((4.0, 5.0, 6.0))

    def test_from_camera_captures_view_up(self):
        cam = vtk.vtkCamera()
        cam.SetViewUp(0.0, 1.0, 0.0)
        state = CameraState.from_camera(cam)
        assert state.view_up == pytest.approx((0.0, 1.0, 0.0))

    def test_restore_roundtrip(self):
        """Save then restore must recover the original orientation exactly."""
        cam = vtk.vtkCamera()
        cam.SetPosition(10.0, 20.0, 30.0)
        cam.SetFocalPoint(0.0, 0.0, 0.0)
        cam.SetViewUp(0.0, 0.0, 1.0)

        state = CameraState.from_camera(cam)

        # Mutate the camera
        cam.SetPosition(0.0, 0.0, 0.0)
        cam.SetFocalPoint(99.0, 99.0, 99.0)

        # Restore
        state.restore(cam)

        assert cam.GetPosition() == pytest.approx((10.0, 20.0, 30.0))
        assert cam.GetFocalPoint() == pytest.approx((0.0, 0.0, 0.0))
        assert cam.GetViewUp() == pytest.approx((0.0, 0.0, 1.0))

    def test_dataclass_direct_construction(self):
        state = CameraState(
            position=(1.0, 2.0, 3.0),
            focal_point=(4.0, 5.0, 6.0),
            view_up=(0.0, 0.0, 1.0),
        )
        assert state.position == (1.0, 2.0, 3.0)
        assert state.focal_point == (4.0, 5.0, 6.0)
        assert state.view_up == (0.0, 0.0, 1.0)
