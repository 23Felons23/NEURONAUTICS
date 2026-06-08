"""Main application window (PySide6 + VTK)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollBar,
    QSlider,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from app.viz.tractography_renderer import TractographyRenderer
    from app.viz.voxel_model import VoxelModelRenderer
    from app.viz.connectome_graph import ConnectomeGraphRenderer
    from app.viz.tumor_overlay import TumorOverlayRenderer
    from app.viz.mesh_renderer import MeshRenderer

try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # type: ignore
except ImportError:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # type: ignore

import vtkmodules.all as vtk  # type: ignore

from app.core.volume import Volume
from app.scene.camera import CameraState  # noqa: F401 (re-exported for convenience)
from app.scene.mode_controller import ModeController
from app.scene.picking_controller import PickingController
from app.scene.scene_manager import SceneManager
from app.ui.mode_selector import ModeSelector
from app.viz.slice_viewer import SliceViewer


class MainWindow(QMainWindow):
    """Main application window.

    Creates a VTK viewport in the centre and a controls dock on the left.
    """

    def __init__(
        self,
        volume: Volume,
        tract_renderer: "TractographyRenderer | None" = None,
        voxel_renderer: "VoxelModelRenderer | None" = None,
        connectome_renderer: "ConnectomeGraphRenderer | None" = None,
        tumor_renderer: "TumorOverlayRenderer | None" = None,
        mesh_renderer: "MeshRenderer | None" = None,
        hand_tracking: bool = False,
    ) -> None:
        super().__init__()
        self.setWindowTitle("NeuroNautics — T1 MRI Viewer")
        self.resize(1280, 800)
        self._volume = volume
        self._tract_renderer = tract_renderer
        self._voxel_renderer = voxel_renderer
        self._is_dragging_explode = False
        self._phantom_actors = {}
        self._mesh_opacity_slider = None
        self._explode_slider = None

        # ---- VTK setup -----------------------------------------------
        self._vtk_widget = QVTKRenderWindowInteractor(self)
        self.setCentralWidget(self._vtk_widget)

        self._scene = SceneManager()
        self._vtk_widget.GetRenderWindow().AddRenderer(self._scene.renderer)

        # Trackball camera interaction
        style = vtk.vtkInteractorStyleTrackballCamera()
        self._vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)

        # ---- Slice viewer --------------------------------------------
        self._viewer = SliceViewer(volume)
        for name, actor in self._viewer.actors.items():
            self._scene.add_actor(f"slice_{name}", actor)

        # ---- Tractography overlay (optional) -------------------------
        if tract_renderer is not None:
            for name, actor in tract_renderer.actors.items():
                self._scene.add_actor(name, actor)

        # Configure default camera view: look at left side of the brain (view from negative anatomical X axis, view-up along anatomical Z axis)
        import numpy as np
        affine = volume.affine
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        dirs = affine[:3, :3] / spacing  # Normalized direction cosines (columns)

        # Dynamically identify columns corresponding to world X (Left-Right), Y (Anterior-Posterior), Z (Superior-Inferior)
        col_r = int(np.argmax(np.abs(dirs[0, :])))
        col_a = int(np.argmax(np.abs(dirs[1, :])))
        col_s = int(np.argmax(np.abs(dirs[2, :])))

        # Sign-align vectors to positive RAS direction cosines
        r_vec = dirs[:, col_r] * np.sign(dirs[0, col_r])
        a_vec = dirs[:, col_a] * np.sign(dirs[1, col_a])
        s_vec = dirs[:, col_s] * np.sign(dirs[2, col_s])

        camera = self._scene.renderer.GetActiveCamera()
        camera.SetFocalPoint(0.0, 0.0, 0.0)
        camera.SetPosition(-r_vec[0], -r_vec[1], -r_vec[2])
        camera.SetViewUp(s_vec[0], s_vec[1], s_vec[2])

        self._scene.reset_camera()
        from app.scene.camera import CameraState
        self._default_camera_state = CameraState.from_camera(camera)

        # ---- Mode controller -----------------------------------------
        self._mode_controller = ModeController(self._scene)
        slice_keys = [f"slice_{k}" for k in self._viewer.actors]
        self._mode_controller.register("T1 Slices", slice_keys)
        if tract_renderer is not None:
            tract_keys = list(tract_renderer.actors.keys())
            self._mode_controller.register("Tractography", tract_keys)

        # ---- Voxel model overlay (optional) -------------------------
        if voxel_renderer is not None:
            for name, actor in voxel_renderer.actors.items():
                self._scene.add_actor(name, actor)
            voxel_keys = list(voxel_renderer.actors.keys())
            self._mode_controller.register("Voxel Model", voxel_keys)

        # ---- Connectome graph overlay (optional) ---------------------
        if connectome_renderer is not None:
            self._connectome_renderer = connectome_renderer
            for name, actor in connectome_renderer.actors.items():
                self._scene.add_actor(name, actor)
            conn_keys = list(connectome_renderer.actors.keys())
            self._mode_controller.register("Connectome Graph", conn_keys)

        # ---- Tumor overlay (optional) --------------------------------
        if tumor_renderer is not None:
            self._tumor_renderer = tumor_renderer
            for name, actor in tumor_renderer.actors.items():
                self._scene.add_actor(name, actor)
            tumor_keys = list(tumor_renderer.actors.keys())
            self._mode_controller.register("Tumor Overlay", tumor_keys)

        # ---- Mesh overlay (optional) ---------------------------------
        if mesh_renderer is not None:
            self._mesh_renderer = mesh_renderer
            for name, actor in mesh_renderer.actors.items():
                self._scene.add_actor(name, actor)
            mesh_keys = list(mesh_renderer.actors.keys())
            self._mode_controller.register("Surface Mesh", mesh_keys)

        # ---- Picking controller (only when mesh renderer loaded) ------
        self._picking_controller: PickingController | None = None
        self._selected_key: str | None = None
        self._deleted_keys: list[str] = []

        # ---- Hand tracking (optional) ------------------------------------
        self._hand_tracking_enabled = hand_tracking
        self._hand_thread: QThread | None = None
        self._hand_worker = None  # type: HandTrackingWorker | None
        self._gesture_mapper = None  # type: GestureMapper | None

        if mesh_renderer is not None:
            iren = self._vtk_widget.GetRenderWindow().GetInteractor()
            self._picking_controller = PickingController(
                self._scene.renderer, iren
            )
            # Register all mesh actors for reverse lookup
            for key, actor in mesh_renderer.actors.items():
                self._picking_controller.register_actor(key, actor)
            self._picking_controller.part_picked.connect(self._on_part_picked)



        # ---- Controls dock -------------------------------------------
        self._dock = self._build_controls_dock(
            volume,
            tract_renderer,
            voxel_renderer,
            connectome_renderer,
            tumor_renderer,
            mesh_renderer,
            hand_tracking=self._hand_tracking_enabled,
        )
        self.addDockWidget(Qt.LeftDockWidgetArea, self._dock)

        # ---- Hand tracking panel overlay -----------------------------
        if self._hand_tracking_enabled:
            from app.ui.hand_tracking_panel import HandTrackingPanel
            self._hand_panel = HandTrackingPanel(self._vtk_widget)
            self._hand_panel.tracking_toggled.connect(self._on_hand_tracking_toggled)
            self._hand_panel.show()

        # ---- Start interaction ----------------------------------------
        self._vtk_widget.Initialize()
        self._vtk_widget.Start()

    # ------------------------------------------------------------------
    # Controls dock
    # ------------------------------------------------------------------

    def _build_controls_dock(
        self,
        volume: Volume,
        tract_renderer: "TractographyRenderer | None" = None,
        voxel_renderer: "VoxelModelRenderer | None" = None,
        connectome_renderer: "ConnectomeGraphRenderer | None" = None,
        tumor_renderer: "TumorOverlayRenderer | None" = None,
        mesh_renderer: "MeshRenderer | None" = None,
        hand_tracking: bool = False,
    ) -> QDockWidget:
        dock = QDockWidget("Controls", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignTop)

        group = QGroupBox("Slice planes")
        group_layout = QVBoxLayout(group)

        self._slice_labels = {}
        axes = [
            ("Axial (Z)",    "axial",    volume.shape[2]),
            ("Coronal (Y)",  "coronal",  volume.shape[1]),
            ("Sagittal (X)", "sagittal", volume.shape[0]),
        ]

        for label_text, axis_name, n_slices in axes:
            init_idx = n_slices // 2
            label = QLabel(f"{label_text}: Slice {init_idx + 1} of {n_slices}")
            scrollbar = QScrollBar(Qt.Horizontal)
            scrollbar.setRange(0, n_slices - 1)
            scrollbar.setValue(init_idx)
            scrollbar.valueChanged.connect(
                lambda idx, name=axis_name: self._on_slice_changed(name, idx)
            )
            self._slice_labels[axis_name] = label
            group_layout.addWidget(label)
            group_layout.addWidget(scrollbar)

        self._slice_widget = QWidget()
        slice_layout = QVBoxLayout(self._slice_widget)
        slice_layout.setContentsMargins(12, 4, 0, 4)
        slice_layout.addWidget(group)

        # ---- Voxel model controls (only when renderer loaded) --------
        self._voxel_widget = None
        if voxel_renderer is not None:
            self._voxel_widget = QWidget()
            voxel_layout = QVBoxLayout(self._voxel_widget)
            voxel_layout.setContentsMargins(12, 4, 0, 4)

            # 1. Opacity Slider
            opacity_row = QHBoxLayout()
            opacity_label = QLabel("Opacity")
            self._voxel_opacity_value = QLabel("85%")
            self._voxel_opacity_value.setStyleSheet("color: #aaa; min-width: 36px;")

            opacity_slider = QSlider(Qt.Horizontal)
            opacity_slider.setRange(0, 100)
            opacity_slider.setValue(85)          # matches VoxelModelRenderer default 0.85
            opacity_slider.setToolTip("Voxel point cloud opacity (0 – 100%)")
            opacity_slider.valueChanged.connect(self._on_voxel_opacity_changed)

            opacity_row.addWidget(opacity_label)
            opacity_row.addWidget(opacity_slider)
            opacity_row.addWidget(self._voxel_opacity_value)
            voxel_layout.addLayout(opacity_row)

            # 2. Voxel Size Slider
            size_row = QHBoxLayout()
            size_label = QLabel("Size")
            actor = voxel_renderer.actors["voxel_model"]
            init_point_size = actor.GetProperty().GetPointSize()
            self._voxel_size_value = QLabel(f"{int(init_point_size)}px")
            self._voxel_size_value.setStyleSheet("color: #aaa; min-width: 36px;")

            size_slider = QSlider(Qt.Horizontal)
            size_slider.setRange(1, 10)
            size_slider.setValue(int(init_point_size))
            size_slider.setToolTip("Voxel point size (1 – 10)")
            size_slider.valueChanged.connect(self._on_voxel_size_changed)

            size_row.addWidget(size_label)
            size_row.addWidget(size_slider)
            size_row.addWidget(self._voxel_size_value)
            voxel_layout.addLayout(size_row)

            # 3. Sphere Rendering Checkbox
            self._spheres_checkbox = QCheckBox("Render as 3D Spheres")
            self._spheres_checkbox.setChecked(actor.GetProperty().GetRenderPointsAsSpheres() == 1)
            self._spheres_checkbox.toggled.connect(self._on_voxel_spheres_toggled)
            voxel_layout.addWidget(self._spheres_checkbox)

            # 4. Contrast Controls (Black/White Level Slider mapping [min, max] intensities)
            self._voxel_min_intensity = voxel_renderer.min_intensity
            self._voxel_max_intensity = voxel_renderer.max_intensity

            s_min, s_max = actor.GetMapper().GetScalarRange()

            def to_slider(val):
                denominator = self._voxel_max_intensity - self._voxel_min_intensity
                if denominator < 1e-6:
                    return 0
                return int((val - self._voxel_min_intensity) / denominator * 1000)

            # Black Cutoff Slider
            black_row = QHBoxLayout()
            black_label = QLabel("Black Cutoff")
            self._voxel_black_value_lbl = QLabel(f"{s_min:.1f}")
            self._voxel_black_value_lbl.setStyleSheet("color: #aaa; min-width: 48px;")

            self._voxel_black_slider = QSlider(Qt.Horizontal)
            self._voxel_black_slider.setRange(0, 1000)
            self._voxel_black_slider.setValue(to_slider(s_min))
            self._voxel_black_slider.setToolTip("Raise to clamp darker voxels to pure black")
            self._voxel_black_slider.valueChanged.connect(self._on_voxel_contrast_changed)

            black_row.addWidget(black_label)
            black_row.addWidget(self._voxel_black_slider)
            black_row.addWidget(self._voxel_black_value_lbl)
            voxel_layout.addLayout(black_row)

            # White Cutoff Slider
            white_row = QHBoxLayout()
            white_label = QLabel("White Cutoff")
            self._voxel_white_value_lbl = QLabel(f"{s_max:.1f}")
            self._voxel_white_value_lbl.setStyleSheet("color: #aaa; min-width: 48px;")

            self._voxel_white_slider = QSlider(Qt.Horizontal)
            self._voxel_white_slider.setRange(0, 1000)
            self._voxel_white_slider.setValue(to_slider(s_max))
            self._voxel_white_slider.setToolTip("Lower to saturate voxels to pure white")
            self._voxel_white_slider.valueChanged.connect(self._on_voxel_contrast_changed)

            white_row.addWidget(white_label)
            white_row.addWidget(self._voxel_white_slider)
            white_row.addWidget(self._voxel_white_value_lbl)
            voxel_layout.addLayout(white_row)

        # ---- Mesh controls (only when renderer loaded) --------
        self._mesh_widget = None
        if mesh_renderer is not None:
            self._mesh_widget = QWidget()
            mesh_layout = QVBoxLayout(self._mesh_widget)
            mesh_layout.setContentsMargins(12, 4, 0, 4)

            opacity_row = QHBoxLayout()
            opacity_label = QLabel("Opacity")
            self._mesh_opacity_value = QLabel("30%")
            self._mesh_opacity_value.setStyleSheet("color: #aaa; min-width: 36px;")

            self._mesh_opacity_slider = QSlider(Qt.Horizontal)
            self._mesh_opacity_slider.setRange(0, 100)
            self._mesh_opacity_slider.setToolTip("Mesh opacity (0 – 100%)")
            self._mesh_opacity_slider.valueChanged.connect(self._on_mesh_opacity_changed)
            self._mesh_opacity_slider.setValue(30)

            opacity_row.addWidget(opacity_label)
            opacity_row.addWidget(self._mesh_opacity_slider)
            opacity_row.addWidget(self._mesh_opacity_value)
            mesh_layout.addLayout(opacity_row)

        # ---- Mode selector —  wired to ModeController ---------------
        sub_widgets = {}
        if hasattr(self, "_slice_widget") and self._slice_widget is not None:
            sub_widgets["T1 Slices"] = self._slice_widget
        if self._voxel_widget is not None:
            sub_widgets["Voxel Model"] = self._voxel_widget
        if self._mesh_widget is not None:
            sub_widgets["Surface Mesh"] = self._mesh_widget

        self._mode_selector = ModeSelector(self._mode_controller, sub_widgets)
        layout.addWidget(self._mode_selector)

        # ---- Mesh picking controls (only when mesh renderer loaded) ---
        if mesh_renderer is not None:
            pick_group = QGroupBox("Mesh Picking")
            pick_layout = QVBoxLayout(pick_group)

            hint_label = QLabel("Ctrl + click a mesh part to select it")
            hint_label.setStyleSheet("color: #aaa; font-size: 11px;")
            pick_layout.addWidget(hint_label)

            self._selected_label = QLabel("Selected: none")
            self._selected_label.setStyleSheet("color: #ccc; font-size: 11px;")
            pick_layout.addWidget(self._selected_label)

            self._delete_btn = QPushButton("Delete Selected Part")
            self._delete_btn.setEnabled(False)
            self._delete_btn.setToolTip(
                "Hide the selected mesh region and its voxel points"
            )
            self._delete_btn.clicked.connect(self._on_delete_selected)
            pick_layout.addWidget(self._delete_btn)

            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setStyleSheet("color: #444;")
            pick_layout.addWidget(separator)

            self._restore_btn = QPushButton("Restore All")
            self._restore_btn.setEnabled(False)
            self._restore_btn.setToolTip("Make all deleted parts visible again")
            self._restore_btn.clicked.connect(self._on_restore_all)
            pick_layout.addWidget(self._restore_btn)

            layout.addWidget(pick_group)

        # ---- Exploded view controls (when any explodable renderer is loaded) --------
        if mesh_renderer is not None or connectome_renderer is not None or voxel_renderer is not None:
            explode_group = QGroupBox("Exploded View")
            explode_layout = QVBoxLayout(explode_group)

            # Check if parcellation data is missing
            parc_missing = False
            if connectome_renderer is not None and not getattr(connectome_renderer, "has_centroids", True):
                parc_missing = True
            elif voxel_renderer is not None and getattr(voxel_renderer, "_labels", None) is None:
                parc_missing = True
            elif mesh_renderer is not None and not getattr(mesh_renderer, "centroids", None):
                parc_missing = True

            if parc_missing:
                warning_label = QLabel(
                    "⚠️ Warning: Parcellation data is missing.\n"
                    "Connectome will render as a sphere, and the\n"
                    "exploded view will not be anatomically accurate."
                )
                warning_label.setStyleSheet("color: #ffaa00; font-size: 11px; font-weight: bold;")
                warning_label.setWordWrap(True)
                explode_layout.addWidget(warning_label)

            explode_row = QHBoxLayout()
            explode_label = QLabel("Explode")
            self._explode_value = QLabel("0%")
            self._explode_value.setStyleSheet("color: #aaa; min-width: 36px;")

            self._explode_slider = QSlider(Qt.Horizontal)
            self._explode_slider.setRange(0, 100)
            self._explode_slider.setToolTip("Exploded view distance (0 – 100%)")
            self._explode_slider.valueChanged.connect(self._on_mesh_explode_changed)
            self._explode_slider.sliderPressed.connect(self._on_explode_slider_pressed)
            self._explode_slider.sliderReleased.connect(self._on_explode_slider_released)
            self._explode_slider.setValue(0)

            explode_row.addWidget(explode_label)
            explode_row.addWidget(self._explode_slider)
            explode_row.addWidget(self._explode_value)
            explode_layout.addLayout(explode_row)
            
            layout.addWidget(explode_group)

        dataset_group = QGroupBox("Dataset Properties")
        dataset_layout = QVBoxLayout(dataset_group)

        voxel_res_lbl = QLabel(
            f"Voxel Resolution: {volume.voxel_sizes_mm[0]:.2f} × "
            f"{volume.voxel_sizes_mm[1]:.2f} × "
            f"{volume.voxel_sizes_mm[2]:.2f} mm"
        )
        voxel_res_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        voxel_res_lbl.setToolTip("Physical size of a single voxel in millimeters.")

        fov_x = volume.shape[0] * volume.voxel_sizes_mm[0]
        fov_y = volume.shape[1] * volume.voxel_sizes_mm[1]
        fov_z = volume.shape[2] * volume.voxel_sizes_mm[2]
        fov_lbl = QLabel(
            f"Field of View: {fov_x:.1f} × {fov_y:.1f} × {fov_z:.1f} mm"
        )
        fov_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        fov_lbl.setToolTip("Physical dimensions of the scanned anatomical volume.")

        dataset_layout.addWidget(voxel_res_lbl)
        dataset_layout.addWidget(fov_lbl)
        layout.addWidget(dataset_group)

        dock.setWidget(container)
        return dock


    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_slice_changed(self, axis_name: str, index: int) -> None:
        self._viewer.update_slice(axis_name, index)
        if hasattr(self, "_slice_labels") and axis_name in self._slice_labels:
            lbl = self._slice_labels[axis_name]
            display_names = {
                "axial": ("Axial (Z)", self._volume.shape[2]),
                "coronal": ("Coronal (Y)", self._volume.shape[1]),
                "sagittal": ("Sagittal (X)", self._volume.shape[0]),
            }
            prefix, total = display_names[axis_name]
            lbl.setText(f"{prefix}: Slice {index + 1} of {total}")
        self._vtk_widget.GetRenderWindow().Render()

    def _on_tract_visibility_changed(self, visible: bool) -> None:
        """Delegated to ModeController for camera-stable toggling."""
        self._mode_controller.set_visible("Tractography", visible)
        self._vtk_widget.GetRenderWindow().Render()

    def _on_voxel_opacity_changed(self, value: int) -> None:
        """Set the voxel cube actor opacity from the slider (0-100 → 0.0-0.99).
        We clamp at 0.99 to prevent VTK from switching to the opaque rendering pass.
        """
        opacity = min(value / 100.0, 0.99)
        if hasattr(self, "_voxel_opacity_value"):
            self._voxel_opacity_value.setText(f"{value}%")
        actor = self._scene.get_actor("voxel_model")
        if actor is not None:
            actor.GetProperty().SetOpacity(opacity)
            self._vtk_widget.GetRenderWindow().Render()

    def _on_voxel_size_changed(self, value: int) -> None:
        """Set the voxel points visual size."""
        if hasattr(self, "_voxel_size_value"):
            self._voxel_size_value.setText(f"{value}px")
        if self._voxel_renderer is not None:
            self._voxel_renderer.set_point_size(value)
            self._vtk_widget.GetRenderWindow().Render()

    def _on_voxel_spheres_toggled(self, checked: bool) -> None:
        """Set whether voxel points render as 3D spheres."""
        if self._voxel_renderer is not None:
            self._voxel_renderer.set_render_spheres(checked)
            self._vtk_widget.GetRenderWindow().Render()

    def _on_voxel_contrast_changed(self) -> None:
        """Set voxel intensity contrast range from the black/white sliders."""
        if self._voxel_renderer is not None:
            # Map slider values [0, 1000] back to intensities
            b_slider = self._voxel_black_slider.value()
            w_slider = self._voxel_white_slider.value()

            min_i = self._voxel_min_intensity
            max_i = self._voxel_max_intensity
            span = max_i - min_i

            black_val = min_i + (b_slider / 1000.0) * span
            white_val = min_i + (w_slider / 1000.0) * span

            # Enforce bounds (e.g. white level >= black level) to prevent inverse mapping errors
            if white_val < black_val:
                white_val = black_val

            self._voxel_black_value_lbl.setText(f"{black_val:.1f}")
            self._voxel_white_value_lbl.setText(f"{white_val:.1f}")

            self._voxel_renderer.set_intensity_range(black_val, white_val)
            self._vtk_widget.GetRenderWindow().Render()


    def _on_mesh_opacity_changed(self, value: int) -> None:
        """Set the surface mesh actor opacity from the slider (0-100 → 0.0-0.99).
        We clamp at 0.99 to prevent VTK from switching to the opaque rendering pass.
        """
        opacity = min(value / 100.0, 0.99)
        if hasattr(self, "_mesh_opacity_value"):
            self._mesh_opacity_value.setText(f"{value}%")
            
        # Directly modify actors in the scene to guarantee visual updates
        for name, actor in self._scene._actors.items():
            if name.startswith("surface_mesh"):
                actor.GetProperty().SetOpacity(opacity)
        
        self._vtk_widget.GetRenderWindow().Render()

    def _on_mesh_explode_changed(self, value: int) -> None:
        """Explodes the parcellation regions away from the center (mesh, voxel, connectome, tractography)."""
        if hasattr(self, "_explode_value"):
            self._explode_value.setText(f"{value}%")
            
        factor = value / 100.0
        max_dist = 150.0
                
        # Resolve global centroid from available renderers or volume center
        global_centroid = None
        if hasattr(self, "_mesh_renderer") and self._mesh_renderer is not None:
            global_centroid = self._mesh_renderer.global_centroid
        elif hasattr(self, "_voxel_renderer") and self._voxel_renderer is not None:
            global_centroid = self._voxel_renderer.global_centroid
        elif hasattr(self, "_volume") and self._volume is not None:
            nx, ny, nz = self._volume.shape
            import numpy as np
            center_ijk = np.array([nx / 2.0, ny / 2.0, nz / 2.0, 1.0])
            global_centroid = (self._volume.affine @ center_ijk)[:3]
        else:
            import numpy as np
            global_centroid = np.zeros(3, dtype=np.float32)

        # Only update phantom actors and skip everything else if dragging
        is_dragging = getattr(self, "_is_dragging_explode", False)
        if is_dragging:
            if hasattr(self, "_mesh_renderer") and self._mesh_renderer is not None:
                for key, phantom_actor in self._phantom_actors.items():
                    centroid = self._mesh_renderer.centroids.get(key)
                    if centroid is not None:
                        direction = centroid - global_centroid
                        import numpy as np
                        norm = np.linalg.norm(direction)
                        if norm > 0.001:
                            direction = direction / norm
                        else:
                            direction = np.array([0.0, 0.0, 0.0])
                        
                        translation = direction * (factor * max_dist)
                        tx = float(translation[0])
                        ty = float(translation[1])
                        tz = float(translation[2])
                        
                        phantom_actor.SetPosition(tx, ty, tz)
                        phantom_actor.Modified()
            self._vtk_widget.GetRenderWindow().Render()
            return

        # 1. Explode mesh if available
        if hasattr(self, "_mesh_renderer") and self._mesh_renderer is not None:
            for key, centroid in self._mesh_renderer.centroids.items():
                actor = self._scene.get_actor(key)
                if actor is not None:
                    direction = centroid - global_centroid
                    import numpy as np
                    norm = np.linalg.norm(direction)
                    if norm > 0.001:
                        direction = direction / norm
                    else:
                        direction = np.array([0.0, 0.0, 0.0])
                    
                    translation = direction * (factor * max_dist)
                    
                    # Force conversion to native Python float to avoid silent VTK wrapper bugs with NumPy scalars
                    tx = float(translation[0])
                    ty = float(translation[1])
                    tz = float(translation[2])
                    
                    actor.SetPosition(tx, ty, tz)
                    actor.Modified()
                    
                    # Sync tumor fragments perfectly with their corresponding mesh fragments
                    if key.startswith("surface_mesh_"):
                        label = key.split("_")[-1]
                        tumor_key = f"tumor_overlay_{label}"
                        tumor_actor = self._scene.get_actor(tumor_key)
                        if tumor_actor is not None:
                            tumor_actor.SetPosition(tx, ty, tz)
                            tumor_actor.Modified()
                    
        # Extract mesh centroids for synchronization
        mesh_centroids = None
        if hasattr(self, "_mesh_renderer") and self._mesh_renderer is not None:
            mesh_centroids = {}
            for key, centroid in self._mesh_renderer.centroids.items():
                if key.startswith("surface_mesh_"):
                    try:
                        lbl = int(key.split("_")[-1])
                        mesh_centroids[lbl] = centroid
                    except ValueError:
                        continue

        # 2. Explode connectome if available
        if hasattr(self, "_connectome_renderer") and self._connectome_renderer is not None:
            self._connectome_renderer.explode(factor, max_dist, global_centroid, mesh_centroids)

        # 3. Explode voxel points if available
        if hasattr(self, "_voxel_renderer") and self._voxel_renderer is not None:
            self._voxel_renderer.explode(factor, max_dist, global_centroid, mesh_centroids)

        # 4. Explode tractography streamlines if available
        if self._tract_renderer is not None:
            self._tract_renderer.explode(factor, max_dist, global_centroid, mesh_centroids)

        self._vtk_widget.GetRenderWindow().Render()

    def _on_explode_slider_pressed(self) -> None:
        self._is_dragging_explode = True
        
        # Clear any existing phantom actors from renderer just in case
        for phantom_actor in self._phantom_actors.values():
            self._scene.renderer.RemoveActor(phantom_actor)
        self._phantom_actors.clear()
        
        # Create phantom actors for all non-deleted surface mesh actors
        for key, original_actor in self._scene._actors.items():
            if key.startswith("surface_mesh_") and key not in self._deleted_keys:
                phantom_actor = vtk.vtkActor()
                phantom_actor.SetMapper(original_actor.GetMapper())
                
                # Style as solid red, 35% opacity
                phantom_actor.SetProperty(vtk.vtkProperty())
                phantom_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
                phantom_actor.GetProperty().SetOpacity(0.35)
                phantom_actor.GetProperty().SetInterpolationToPhong()
                phantom_actor.GetProperty().SetAmbient(0.1)
                phantom_actor.GetProperty().SetDiffuse(0.7)
                phantom_actor.GetProperty().SetSpecular(0.2)
                phantom_actor.GetProperty().SetSpecularPower(10.0)
                phantom_actor.GetProperty().SetRepresentation(
                    original_actor.GetProperty().GetRepresentation()
                )
                
                # Match position
                phantom_actor.SetPosition(original_actor.GetPosition())
                
                self._scene.renderer.AddActor(phantom_actor)
                self._phantom_actors[key] = phantom_actor
                
        self._vtk_widget.GetRenderWindow().Render()

    def _on_explode_slider_released(self) -> None:
        self._is_dragging_explode = False
        
        # Clean up phantom actors
        for phantom_actor in self._phantom_actors.values():
            self._scene.renderer.RemoveActor(phantom_actor)
        self._phantom_actors.clear()
        
        # Force a full update to the current value on release
        if hasattr(self, "_explode_slider") and self._explode_slider is not None:
            self._on_mesh_explode_changed(self._explode_slider.value())

    def _on_part_picked(self, scene_key: str) -> None:
        """Called when PickingController emits part_picked(scene_key)."""
        self._selected_key = scene_key
        # Show a short label (truncate long keys for display)
        display_name = scene_key.replace("surface_mesh_", "Part ")
        if hasattr(self, "_selected_label"):
            self._selected_label.setText(f"Selected: {display_name}")
        if hasattr(self, "_delete_btn"):
            self._delete_btn.setEnabled(True)
        self._vtk_widget.GetRenderWindow().Render()

    def _on_delete_selected(self) -> None:
        """Hide the selected mesh part and its corresponding voxel region."""
        if self._selected_key is None:
            return

        key = self._selected_key

        # 1. Clear picking highlight before hiding the actor
        if self._picking_controller is not None:
            self._picking_controller.clear_highlight()

        # 2. Hide the mesh actor in the scene
        self._scene.set_visible(key, False)
        self._deleted_keys.append(key)

        # 3. Sync voxel point cloud — hide the matching parcellation label
        if hasattr(self, "_voxel_renderer") and self._voxel_renderer is not None:
            if key.startswith("surface_mesh_"):
                label_str = key.split("_")[-1]
                try:
                    label = int(label_str)
                    self._voxel_renderer.hide_label(label)
                except ValueError:
                    pass  # key format unexpected — no voxel sync

        # 4. Reset UI state
        self._selected_key = None
        if hasattr(self, "_selected_label"):
            self._selected_label.setText("Selected: none")
        if hasattr(self, "_delete_btn"):
            self._delete_btn.setEnabled(False)
        if hasattr(self, "_restore_btn"):
            self._restore_btn.setEnabled(bool(self._deleted_keys))

        self._vtk_widget.GetRenderWindow().Render()

    def _on_restore_all(self) -> None:
        """Restore all deleted mesh parts and voxel regions."""
        for key in self._deleted_keys:
            self._scene.set_visible(key, True)
        self._deleted_keys.clear()

        if hasattr(self, "_voxel_renderer") and self._voxel_renderer is not None:
            self._voxel_renderer.restore_all_labels()

        if hasattr(self, "_restore_btn"):
            self._restore_btn.setEnabled(False)
        if hasattr(self, "_selected_label"):
            self._selected_label.setText("Selected: none")

        self._vtk_widget.GetRenderWindow().Render()

    def _on_hand_tracking_toggled(self, enabled: bool) -> None:
        """Start or stop the hand tracking worker thread."""
        if enabled:
            try:
                from app.tracking.gesture_mapper import GestureMapper
                from app.tracking.hand_worker import HandTrackingWorker
            except ImportError:
                if hasattr(self, "_hand_panel"):
                    self._hand_panel.set_error("Tracking dependencies missing.\nRun: pip install -e .[tracking]")
                return

            self._gesture_mapper = GestureMapper()
            self._hand_worker = HandTrackingWorker()
            self._hand_thread = QThread()
            self._hand_worker.moveToThread(self._hand_thread)

            # Wire signals
            self._hand_thread.started.connect(self._hand_worker.run)
            self._hand_worker.frame_ready.connect(self._on_hand_frame)
            self._hand_worker.landmarks_detected.connect(self._on_hand_landmarks)
            self._hand_worker.two_hands_detected.connect(self._on_two_hands)
            self._hand_worker.hand_lost.connect(self._on_hand_lost)
            self._hand_worker.error.connect(self._on_hand_error)

            self._hand_thread.start()
        else:
            self._stop_hand_tracking()

    def _stop_hand_tracking(self) -> None:
        """Cleanly stop the hand tracking thread."""
        if self._hand_worker is not None:
            self._hand_worker.stop()
        if self._hand_thread is not None:
            self._hand_thread.quit()
            self._hand_thread.wait(3000)
            self._hand_thread = None
        self._hand_worker = None
        if self._gesture_mapper is not None:
            self._gesture_mapper.reset()
        if hasattr(self, "_hand_panel"):
            self._hand_panel.set_reset_progress(0.0)

    def _on_hand_frame(self, image: QImage) -> None:
        """Update the video feed in the hand tracking panel."""
        if hasattr(self, "_hand_panel"):
            self._hand_panel.update_frame(image)

    def _on_hand_landmarks(self, data: dict) -> None:
        """Process landmarks through GestureMapper and update VTK camera."""
        if self._gesture_mapper is None:
            return

        cmd = self._gesture_mapper.process(data["landmarks"])

        # Update gesture display
        if hasattr(self, "_hand_panel"):
            self._hand_panel.set_gesture(cmd.gesture)
            self._hand_panel.set_status("ACTIVE", "#0f0")
            self._hand_panel.set_reset_progress(cmd.reset_progress)

        # Apply camera commands
        camera = self._scene.renderer.GetActiveCamera()
        if cmd.reset_camera:
            if hasattr(self, "_default_camera_state") and self._default_camera_state is not None:
                self._default_camera_state.restore(camera)
            self._scene.reset_camera()
            self._world_up = tuple(camera.GetViewUp())
        else:
            has_rot = (abs(cmd.azimuth_delta) > 0.01 or abs(cmd.elevation_delta) > 0.01)
            if has_rot:
                if not hasattr(self, "_world_up") or self._world_up is None:
                    self._world_up = tuple(camera.GetViewUp())

                # Normalize world up vector
                w_up_len = sum(x**2 for x in self._world_up)**0.5
                w_up = [x / w_up_len for x in self._world_up] if w_up_len > 0.001 else [0.0, 0.0, 1.0]

                pos = camera.GetPosition()
                fp = camera.GetFocalPoint()
                up = camera.GetViewUp()

                # Check if camera is currently upside-down relative to world vertical
                is_upside_down = False
                w_up_dot_up = sum(w_up[i] * up[i] for i in range(3))
                if w_up_dot_up < 0.0:
                    is_upside_down = True

                # Invert azimuth delta if upside-down to keep horizontal screen-space controls intuitive.
                # Elevation does not need inversion because rotation is around the camera's local right vector,
                # which naturally flips when the camera is upside-down.
                azimuth_delta = -cmd.azimuth_delta if is_upside_down else cmd.azimuth_delta
                elevation_delta = cmd.elevation_delta

                # Calculate normalized look vector
                look = [fp[i] - pos[i] for i in range(3)]
                look_len = sum(x**2 for x in look)**0.5
                look_norm = [x / look_len for x in look] if look_len > 0.001 else [0.0, 1.0, 0.0]

                # Right vector: look x up
                right = [
                    look_norm[1] * up[2] - look_norm[2] * up[1],
                    look_norm[2] * up[0] - look_norm[0] * up[2],
                    look_norm[0] * up[1] - look_norm[1] * up[0]
                ]
                right_len = sum(x**2 for x in right)**0.5
                right_norm = [x / right_len for x in right] if right_len > 0.001 else [1.0, 0.0, 0.0]

                import math
                def rotate_vector(v: list[float], axis: list[float], angle_degrees: float) -> list[float]:
                    angle_rad = math.radians(angle_degrees)
                    cos_a = math.cos(angle_rad)
                    sin_a = math.sin(angle_rad)
                    cross = [
                        axis[1] * v[2] - axis[2] * v[1],
                        axis[2] * v[0] - axis[0] * v[2],
                        axis[0] * v[1] - axis[1] * v[0]
                    ]
                    dot_val = sum(axis[i] * v[i] for i in range(3))
                    return [
                        v[i] * cos_a + cross[i] * sin_a + axis[i] * dot_val * (1 - cos_a)
                        for i in range(3)
                    ]

                # 1. Rotate position and up around world vertical (azimuth)
                p = [pos[i] - fp[i] for i in range(3)]
                if abs(azimuth_delta) > 0.01:
                    p = rotate_vector(p, w_up, azimuth_delta)
                    up = rotate_vector(list(up), w_up, azimuth_delta)

                # 2. Rotate position and up around camera horizontal (elevation)
                if abs(elevation_delta) > 0.01:
                    p = rotate_vector(p, right_norm, elevation_delta)
                    up = rotate_vector(up, right_norm, elevation_delta)

                # Normalize view up to prevent numerical drift
                up_len = sum(x**2 for x in up)**0.5
                up_norm = [x / up_len for x in up] if up_len > 0.001 else [0.0, 0.0, 1.0]

                # 3. Apply view-up alignment to prevent roll drift when not near pole singularity
                look_new = [-p[i] for i in range(3)]
                look_new_len = sum(x**2 for x in look_new)**0.5
                if look_new_len > 0.001:
                    look_new_norm = [x / look_new_len for x in look_new]
                    dot = sum(look_new_norm[i] * w_up[i] for i in range(3))
                    if abs(dot) < 0.99:
                        proj_up = [w_up[i] - dot * look_new_norm[i] for i in range(3)]
                        proj_len = sum(x**2 for x in proj_up)**0.5
                        if proj_len > 0.001:
                            proj_up_norm = [x / proj_len for x in proj_up]
                            # Compare direction with current up_norm to preserve upside-down state
                            dot_direction = sum(up_norm[i] * proj_up_norm[i] for i in range(3))
                            if dot_direction < 0.0:
                                up_norm = [-x for x in proj_up_norm]
                            else:
                                up_norm = proj_up_norm

                camera.SetPosition(fp[0] + p[0], fp[1] + p[1], fp[2] + p[2])
                camera.SetViewUp(up_norm[0], up_norm[1], up_norm[2])

            if abs(cmd.dolly_factor - 1.0) > 0.001:
                camera.Dolly(cmd.dolly_factor)

        self._vtk_widget.GetRenderWindow().Render()

    def _on_two_hands(self, data: dict) -> None:
        """Process two-hand landmarks for pinch-zoom gesture."""
        if self._gesture_mapper is None:
            return

        cmd = self._gesture_mapper.process_two_hands(
            data["right_landmarks"], data["left_landmarks"]
        )

        # Update gesture display
        if hasattr(self, "_hand_panel"):
            self._hand_panel.set_gesture(cmd.gesture)
            self._hand_panel.set_status("ACTIVE", "#0f0")
            if cmd.gesture != "open_palm":
                self._hand_panel.set_reset_progress(0.0)

        # Apply camera commands
        camera = self._scene.renderer.GetActiveCamera()
        if abs(cmd.dolly_factor - 1.0) > 0.001:
            camera.Dolly(cmd.dolly_factor)
            self._scene.renderer.ResetCameraClippingRange()

        self._vtk_widget.GetRenderWindow().Render()

    def _on_hand_lost(self) -> None:
        """Handle loss of hand detection."""
        if hasattr(self, "_hand_panel"):
            self._hand_panel.set_status("SEARCHING...", "#ffcc00")
            self._hand_panel.set_gesture("none")
            self._hand_panel.set_reset_progress(0.0)
        if self._gesture_mapper is not None:
            self._gesture_mapper.reset()

    def _on_hand_error(self, message: str) -> None:
        """Handle hand tracking errors."""
        if hasattr(self, "_hand_panel"):
            self._hand_panel.set_error(message)
        self._stop_hand_tracking()

    def resizeEvent(self, event) -> None:
        """Reposition the floating Hand HUD in the top-right corner of the central widget."""
        super().resizeEvent(event)
        if hasattr(self, "_hand_panel") and self._hand_panel is not None:
            pw = self._hand_panel.width()
            self._hand_panel.move(self._vtk_widget.width() - pw - 20, 20)
        if hasattr(self, "_vtk_widget") and self._vtk_widget is not None:
            self._vtk_widget.GetRenderWindow().Render()

    def closeEvent(self, event) -> None:
        """Stop hand tracking thread before closing."""
        self._stop_hand_tracking()
        super().closeEvent(event)
