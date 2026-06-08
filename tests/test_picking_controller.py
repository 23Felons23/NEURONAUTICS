"""Tests for app.scene.picking_controller.PickingController."""

from __future__ import annotations

import pytest
import vtkmodules.all as vtk

from app.scene.picking_controller import PickingController


def _make_iren() -> vtk.vtkRenderWindowInteractor:
    """Create a minimal (off-screen) interactor for testing."""
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(True)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(rw)
    return iren


def _make_renderer() -> vtk.vtkRenderer:
    return vtk.vtkRenderer()


class TestPickingControllerInit:
    def test_instantiates_without_error(self) -> None:
        renderer = _make_renderer()
        iren = _make_iren()
        pc = PickingController(renderer, iren)
        assert pc is not None

    def test_highlighted_key_initially_none(self) -> None:
        renderer = _make_renderer()
        iren = _make_iren()
        pc = PickingController(renderer, iren)
        assert pc.highlighted_key is None


class TestActorRegistration:
    def test_register_actor_stored_in_lookup(self) -> None:
        renderer = _make_renderer()
        iren = _make_iren()
        pc = PickingController(renderer, iren)
        actor = vtk.vtkActor()
        pc.register_actor("surface_mesh_42", actor)
        assert pc._actor_to_key[id(actor)] == "surface_mesh_42"

    def test_unregister_actor_removes_from_lookup(self) -> None:
        renderer = _make_renderer()
        iren = _make_iren()
        pc = PickingController(renderer, iren)
        actor = vtk.vtkActor()
        pc.register_actor("surface_mesh_42", actor)
        pc.unregister_actor(actor)
        assert id(actor) not in pc._actor_to_key

    def test_unregister_nonexistent_actor_no_crash(self) -> None:
        renderer = _make_renderer()
        iren = _make_iren()
        pc = PickingController(renderer, iren)
        actor = vtk.vtkActor()
        pc.unregister_actor(actor)  # should not raise


class TestClearHighlight:
    def test_clear_highlight_when_nothing_selected_no_crash(self) -> None:
        renderer = _make_renderer()
        iren = _make_iren()
        pc = PickingController(renderer, iren)
        pc.clear_highlight()  # must not raise
        assert pc.highlighted_key is None

    def test_clear_highlight_resets_highlighted_key(self) -> None:
        renderer = _make_renderer()
        iren = _make_iren()
        pc = PickingController(renderer, iren)
        # Manually set internal state to simulate a highlight
        actor = vtk.vtkActor()
        pc._highlighted_key = "surface_mesh_7"
        pc._highlighted_actor = actor
        pc._orig_color = (0.9, 0.8, 0.7)
        pc._orig_ambient = 0.1
        pc.clear_highlight()
        assert pc.highlighted_key is None
        assert pc._highlighted_actor is None
        assert pc._orig_color is None
        assert pc._orig_ambient is None
