"""Scene manager: owns a vtkRenderer and a registry of named actors."""

from __future__ import annotations

import vtkmodules.all as vtk  # type: ignore


class SceneManager:
    """Manages a VTK renderer and a named collection of actors.

    Actors can be toggled on/off by name without removing them from the
    renderer, which preserves camera state.

    Parameters
    ----------
    renderer:
        An existing vtkRenderer to manage.  If *None*, a new one is created.
    background:
        RGB background colour as floats in [0, 1].
    """

    def __init__(
        self,
        renderer: vtk.vtkRenderer | None = None,
        background: tuple[float, float, float] = (0.08, 0.08, 0.12),
    ) -> None:
        if renderer is None:
            renderer = vtk.vtkRenderer()
        self._renderer = renderer
        self._renderer.SetBackground(*background)
        self._actors: dict[str, vtk.vtkProp] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def renderer(self) -> vtk.vtkRenderer:
        """The underlying vtkRenderer."""
        return self._renderer

    def add_actor(self, name: str, actor: vtk.vtkProp) -> None:
        """Add a named actor to the scene.

        If an actor with the same name already exists it is first removed.
        """
        if name in self._actors:
            self.remove_actor(name)
        self._actors[name] = actor
        self._renderer.AddActor(actor)

    def remove_actor(self, name: str) -> None:
        """Remove a named actor from the scene entirely."""
        if name in self._actors:
            self._renderer.RemoveActor(self._actors.pop(name))

    def set_visible(self, name: str, visible: bool) -> None:
        """Show or hide a named actor."""
        if name in self._actors:
            self._actors[name].SetVisibility(visible)

    def has_actor(self, name: str) -> bool:
        """Return True if an actor with the given name exists."""
        return name in self._actors

    def get_actor(self, name: str) -> vtk.vtkProp | None:
        """Return the actor registered under *name*, or ``None`` if not found."""
        return self._actors.get(name)

    def reset_camera(self) -> None:
        """Reset the camera to fit all visible actors."""
        self._renderer.ResetCamera()

    def render(self) -> None:
        """Trigger a render pass (useful for off-screen tests)."""
        if self._renderer.GetRenderWindow() is not None:
            self._renderer.GetRenderWindow().Render()
