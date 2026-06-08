"""VTK picking controller — maps Ctrl+Click to scene-key signals."""

from __future__ import annotations

import vtkmodules.all as vtk  # type: ignore
from PySide6.QtCore import QObject, Signal


class PickingController(QObject):
    """Manages VTK prop picking and emits Qt signals when a part is picked.

    Listens to ``LeftButtonPressEvent`` on the interactor. A pick is only
    triggered when the user holds **Ctrl** while clicking, so that normal
    left-click camera rotation is unaffected.

    Parameters
    ----------
    renderer:
        The ``vtkRenderer`` that contains the scene actors.
    interactor:
        The ``vtkRenderWindowInteractor`` to observe.
    """

    #: Emitted when a registered actor is Ctrl+clicked.
    #: Argument is the scene key (e.g. ``"surface_mesh_42"``).
    part_picked = Signal(str)

    def __init__(
        self,
        renderer: vtk.vtkRenderer,
        interactor: vtk.vtkRenderWindowInteractor,
    ) -> None:
        super().__init__()
        self._renderer = renderer
        self._interactor = interactor
        self._picker = vtk.vtkPropPicker()

        # Reverse lookup: id(vtkActor) → scene_key
        self._actor_to_key: dict[int, str] = {}

        # Currently highlighted key + saved display values
        self._highlighted_key: str | None = None
        self._highlighted_actor: vtk.vtkActor | None = None
        self._orig_color: tuple[float, float, float] | None = None
        self._orig_ambient: float | None = None

        # Register the observer
        interactor.AddObserver("LeftButtonPressEvent", self._on_left_button_press, 1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_actor(self, key: str, actor: vtk.vtkActor) -> None:
        """Register an actor so it can be reverse-looked-up after a pick.

        Parameters
        ----------
        key:
            The scene key used in ``SceneManager`` (e.g. ``"surface_mesh_42"``).
        actor:
            The corresponding ``vtkActor``.
        """
        self._actor_to_key[id(actor)] = key

    def unregister_actor(self, actor: vtk.vtkActor) -> None:
        """Remove an actor from the reverse-lookup table."""
        self._actor_to_key.pop(id(actor), None)

    def clear_highlight(self) -> None:
        """Remove any active highlight from the previously picked actor."""
        if self._highlighted_actor is not None and self._orig_color is not None:
            prop = self._highlighted_actor.GetProperty()
            prop.SetColor(*self._orig_color)
            if self._orig_ambient is not None:
                prop.SetAmbient(self._orig_ambient)
        self._highlighted_key = None
        self._highlighted_actor = None
        self._orig_color = None
        self._orig_ambient = None

    @property
    def highlighted_key(self) -> str | None:
        """The scene key of the currently highlighted actor, or ``None``."""
        return self._highlighted_key

    # ------------------------------------------------------------------
    # VTK observer callback
    # ------------------------------------------------------------------

    def _on_left_button_press(
        self, interactor: vtk.vtkRenderWindowInteractor, _event: str
    ) -> None:
        """Called on every left-button press. Only picks if Ctrl is held."""
        if not interactor.GetControlKey():
            # Normal click — let the interactor style handle it (camera rotation)
            return

        x, y = interactor.GetEventPosition()
        hit = self._picker.Pick(x, y, 0, self._renderer)

        if not hit:
            self.clear_highlight()
            return

        prop = self._picker.GetViewProp()
        if prop is None:
            self.clear_highlight()
            return

        key = self._actor_to_key.get(id(prop))
        if key is None:
            # Picked a non-registered actor (e.g. scale bar) — ignore
            self.clear_highlight()
            return

        # Highlight the picked actor
        self.clear_highlight()
        actor = prop  # prop is already a vtkActor for registered actors
        try:
            vtk_prop = actor.GetProperty()
            self._orig_color = tuple(vtk_prop.GetColor())  # (r, g, b)
            self._orig_ambient = vtk_prop.GetAmbient()
            vtk_prop.SetColor(1.0, 1.0, 0.0)   # yellow highlight
            vtk_prop.SetAmbient(0.6)
        except AttributeError:
            # Safety: vtkProp2D or other non-actor types have no GetProperty
            pass

        self._highlighted_key = key
        self._highlighted_actor = actor

        self.part_picked.emit(key)
