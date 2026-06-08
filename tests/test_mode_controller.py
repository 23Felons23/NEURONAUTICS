"""Unit tests for ModeController — actor visibility + camera preservation."""

from __future__ import annotations

import pytest
import vtkmodules.all as vtk  # type: ignore

from app.scene.mode_controller import ModeController
from app.scene.scene_manager import SceneManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_actor() -> vtk.vtkActor:
    a = vtk.vtkActor()
    a.VisibilityOn()
    return a


def _scene_with_actor(key: str) -> tuple[SceneManager, vtk.vtkActor]:
    scene = SceneManager()
    actor = _make_actor()
    scene.add_actor(key, actor)
    return scene, actor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModeController:
    def test_register_adds_mode(self):
        scene, _ = _scene_with_actor("t1")
        ctrl = ModeController(scene)
        ctrl.register("T1 Slices", ["t1"])
        assert "T1 Slices" in ctrl.modes

    def test_set_visible_false_hides_actor(self):
        scene, actor = _scene_with_actor("t1")
        ctrl = ModeController(scene)
        ctrl.register("T1 Slices", ["t1"])
        ctrl.set_visible("T1 Slices", False)
        assert not actor.GetVisibility()

    def test_set_visible_true_shows_actor(self):
        scene, actor = _scene_with_actor("t1")
        actor.VisibilityOff()
        ctrl = ModeController(scene)
        ctrl.register("T1 Slices", ["t1"])
        ctrl.set_visible("T1 Slices", True)
        assert actor.GetVisibility()

    def test_camera_preserved_after_hide(self):
        """Switching a mode off must not move the camera."""
        scene, _ = _scene_with_actor("t1")
        cam = scene.renderer.GetActiveCamera()
        cam.SetPosition(10.0, 20.0, 30.0)

        ctrl = ModeController(scene)
        ctrl.register("T1 Slices", ["t1"])
        ctrl.set_visible("T1 Slices", False)

        assert cam.GetPosition() == pytest.approx((10.0, 20.0, 30.0))

    def test_camera_preserved_after_show(self):
        scene, actor = _scene_with_actor("t1")
        actor.VisibilityOff()
        cam = scene.renderer.GetActiveCamera()
        cam.SetPosition(5.0, 6.0, 7.0)

        ctrl = ModeController(scene)
        ctrl.register("T1 Slices", ["t1"])
        ctrl.set_visible("T1 Slices", True)

        assert cam.GetPosition() == pytest.approx((5.0, 6.0, 7.0))

    def test_unknown_mode_is_noop(self):
        """Calling set_visible on a non-registered mode must not raise."""
        scene, _ = _scene_with_actor("t1")
        ctrl = ModeController(scene)
        ctrl.set_visible("NonExistent", False)  # should not raise

    def test_is_active_initial_state(self):
        scene, _ = _scene_with_actor("t1")
        ctrl = ModeController(scene)
        ctrl.register("T1 Slices", ["t1"])
        assert ctrl.is_active("T1 Slices")

    def test_is_active_after_hide(self):
        scene, _ = _scene_with_actor("t1")
        ctrl = ModeController(scene)
        ctrl.register("T1 Slices", ["t1"])
        ctrl.set_visible("T1 Slices", False)
        assert not ctrl.is_active("T1 Slices")
