# NeuroNautics — Technical Architecture & Optimization Guide

This document provides a detailed explanation of the architecture, mathematics, rendering algorithms, hand-tracking pipeline, and performance optimizations in **NeuroNautics** — a multimodal 3D brain data navigator built with VTK and PySide6. It is intended for developers, clinicians, and reviewers.

---

## 1. Application Overview

NeuroNautics loads patient neuroimaging data (MRI volumes, white-matter tractograms, structural connectomes, cortical surface meshes, and lesion masks), co-registers them into a shared physical coordinate space, and renders them as interactive 3D visualizations. Users can scroll through MRI slices, toggle visualization layers, explode parcellation regions, select and hide cortical areas, and control the camera using webcam-based hand gestures.

### 1.1 Technology Stack

| Layer | Technology | Role |
| :--- | :--- | :--- |
| GUI framework | PySide6 (Qt 6) | Widgets, docks, sliders, signals/slots |
| 3D rendering | VTK ≥ 9.3 | GPU-accelerated volume/mesh/point-cloud rendering |
| Neuroimaging I/O | NiBabel, DIPY | NIfTI / TCK file parsing and coordinate transforms |
| Surface meshes | PyVista | OBJ loading, mesh partitioning, cell thresholding |
| Spatial indexing | SciPy `cKDTree` | Nearest-neighbor parcellation mapping |
| Graph algorithms | SciPy `dijkstra` | Tumor spiderweb shortest-path tree |
| Hand tracking | MediaPipe, OpenCV | Webcam gesture recognition |
| Build system | Hatch (`pyproject.toml`) | Packaging, `neuronautics` CLI entry point |

Dependencies are split into required (`nibabel`, `numpy`, `vtk`, `PySide6`, `dipy`), development (`pytest`), and optional tracking extras (`mediapipe`, `opencv-python`).

### 1.2 Entry Point & CLI

The application starts from `app/main.py`, exposed as the `neuronautics` console script via `pyproject.toml`. It accepts arguments through two modes:

**Command-line arguments:**
```
neuronautics --t1 brain.nii.gz --tck tract.tck --connectome matrix.csv \
             --parcellation-volume parc_fs.mif
```

**Manifest file** (recommended):
```
neuronautics --manifest patient/manifest.json
```

 *View README.md for complete example and setup*

The manifest is a JSON file that bundles all data paths and visual configuration into a single portable specification. Relative paths are resolved against the manifest's parent directory, allowing self-contained patient data folders. The manifest parser in `app/io/manifest.py` supports:
- Data paths: `t1`, `tck`, `connectome`, `parcellation_volume`, `tumor`, `mesh`
- Rendering config: `voxel_stride`, `voxel_point_size`, `voxel_spheres`, contrast values
- Color overrides: `mesh_color`, `tumor_color`, `mesh_lut` (per-label color dictionaries)
- Tumor rendering: `tumor_opacity`, `tumor_line_width`, `tumor_num_nodes`

**Automatic color normalization:** When manifest colors are specified in the common 0–255 integer range (e.g. `[255, 50, 50]`), the parser detects values exceeding 1.0 and normalizes them to VTK's 0.0–1.0 float range automatically.

If no T1 path is provided, a native file-chooser dialog opens.

---

## 2. Layered Architecture

NeuroNautics follows strict separation of concerns across five layers. Each layer depends only on the one above it:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          I/O LAYER  (app/io/)                              │
│  Loads binary files (NIfTI, MIF, TCK, CSV, OBJ, JSON) into pure Python    │
│  domain objects. No VTK, PySide6, or UI dependencies.                     │
└──────────────────────┬─────────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         CORE LAYER  (app/core/)                            │
│  Domain models: Volume, TractographyBundle, SurfaceMesh, ConnectomeMatrix. │
│  Pure mathematical data objects with spatial helper methods.               │
└──────────────────────┬─────────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                          VIZ LAYER  (app/viz/)                             │
│  Transforms core objects into VTK actors. Handles shading, coloring,      │
│  transfer functions, and in-place coordinate manipulation.                │
└──────────────────────┬─────────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        SCENE LAYER  (app/scene/)                           │
│  Manages the VTK renderer, actor registries, camera state preservation,   │
│  mode switching, and picking interaction.                                 │
└──────────────────────┬─────────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                          UI LAYER  (app/ui/)                               │
│  PySide6 widgets: main window, collapsible mode panels, sliders, picking  │
│  controls, and the floating hand-tracking HUD overlay.                    │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 File Inventory

#### `app/io/` — File Readers

| File | Purpose |
| :--- | :--- |
| `nifti_loader.py` | Reads `.nii` / `.nii.gz` via NiBabel. Provides `load_mif()` for MRtrix `.mif` files via `mrconvert` subprocess conversion. |
| `tck_loader.py` | Loads streamlines from `.tck` using NiBabel header parsing + DIPY's `StatefulTractogram`. |
| `csv_loader.py` | Reads connectome adjacency matrices. Auto-detects header rows, validates square shape. |
| `obj_loader.py` | Loads `.obj` surface meshes via PyVista, extracting vertices and VTK-format face arrays. |
| `parcellation_mapper.py` | Maps world-space coordinates to parcellation labels using a $K$-D Tree of non-zero voxels. |
| `manifest.py` | Parses patient `manifest.json`. Resolves relative paths, normalizes colors, loads visual settings. |

#### `app/core/` — Domain Models

| File | Purpose |
| :--- | :--- |
| `volume.py` | 3D intensity array + affine. Provides `slice_x/y/z()`, `voxel_sizes_mm`, `percentile_window()`. |
| `tractography.py` | Streamline bundle with `color_by_direction()` tangent-based RGB computation. |
| `mesh.py` | Dataclass: `vertices`, `faces`, optional `labels` array. |
| `connectome.py` | Symmetric weight matrix with `degree()`, `threshold()`, `strongest_connections(k)`. |
| `spatial.py` | `apply_affine()`, `voxel_to_world()`, `world_to_voxel()`, `voxel_size()`. |

#### `app/viz/` — Actor Builders

| File | Purpose |
| :--- | :--- |
| `slice_viewer.py` | Three orthogonal MRI slice planes using `vtkImageSlice` with native direction cosines. |
| `tractography_renderer.py` | Streamlines as direction-colored `vtkPolyData` lines. Single-actor exploded view. |
| `voxel_model.py` | Stride-decimated point cloud with quadratic contrast. NaN sentinel hiding. |
| `connectome_graph.py` | Degree-colored sphere nodes + merged cylinder edges. Fibonacci fallback layout. |
| `mesh_renderer.py` | Parcellation-split surface mesh with per-label sub-actors and Phong shading. |
| `tumor_overlay.py` | Binary lesion mask → Dijkstra shortest-path spiderweb network. |

#### `app/scene/` — Scene Management

| File | Purpose |
| :--- | :--- |
| `scene_manager.py` | `vtkRenderer` wrapper. Named actor registry with `add/remove/set_visible`. |
| `camera.py` | `CameraState` dataclass: captures and restores camera position/focal/view-up. |
| `mode_controller.py` | Groups actors into named modes (e.g. "T1 Slices"). Camera-stable visibility toggling. |
| `picking_controller.py` | Ctrl+Click → `vtkPropPicker` → scene key identification. Highlight with yellow + ambient boost. |

#### `app/ui/` — User Interface

| File | Purpose |
| :--- | :--- |
| `main_window.py` | Central `QVTKRenderWindowInteractor`, controls dock, slider callbacks, hand-tracking camera orbit. |
| `mode_selector.py` | Checkbox panel wired to `ModeController`. Collapses sub-widgets when unchecked. |
| `hand_tracking_panel.py` | Floating glassmorphism HUD: toggle, video feed, gesture label, reset progress bar. Minimizable. |
| `scale_indicator.py` | 2D text overlay displaying voxel dimensions in mm. |

#### `app/tracking/` — Hand Tracking

| File | Purpose |
| :--- | :--- |
| `hand_worker.py` | QThread: webcam capture → horizontal mirror → MediaPipe inference → Qt signals. |
| `gesture_mapper.py` | Gesture classification, EMA smoothing, camera command generation. |
| `model_manager.py` | Auto-downloads the MediaPipe `hand_landmarker.task` model on first use. |

---

## 3. Coordinate Co-Registration & Affine Mathematics

All neuroimaging structures must reside in the exact same physical **RAS+ coordinate space** (Right, Anterior, Superior) measured in millimeters.

```
                          Superior (+Z)
                               ▲
                               │     Anterior (+Y)
                               │    /
                               │   /
      Left (-X) ◄──────────────┼──────────────► Right (+X)
                              /│
                             / │
                            /  ▼
                    Posterior   Inferior (-Z)
                     (-Y)
```

### 3.1 The Voxel-to-World Affine Matrix

Volumetric datasets store data in a discrete 3D grid indexed by voxel coordinates $(i, j, k)$. The NIfTI header provides a $4 \times 4$ homogeneous transformation matrix $\mathbf{A}$ that maps each voxel to physical RAS+ world coordinates $(x, y, z)$ in millimeters:

$$\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix} = \mathbf{A} \cdot \begin{bmatrix} i \\ j \\ k \\ 1 \end{bmatrix} = \begin{bmatrix} r_{00} & r_{01} & r_{02} & t_x \\ r_{10} & r_{11} & r_{12} & t_y \\ r_{20} & r_{21} & r_{22} & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} i \\ j \\ k \\ 1 \end{bmatrix}$$

This matrix decomposes into three components:

1. **Translation ($\mathbf{t}$)** — the world-space position of the first voxel:
   $$\mathbf{t} = \begin{bmatrix} t_x & t_y & t_z \end{bmatrix}^T$$

2. **Voxel Spacing ($\mathbf{s}$)** — the physical size of each voxel, computed as column norms:
   $$s_x = \sqrt{r_{00}^2 + r_{10}^2 + r_{20}^2}, \quad s_y = \sqrt{r_{01}^2 + r_{11}^2 + r_{21}^2}, \quad s_z = \sqrt{r_{02}^2 + r_{12}^2 + r_{22}^2}$$

3. **Direction Cosines ($\mathbf{R}$)** — the orientation of each voxel axis relative to world axes:
   $$\mathbf{R} = \begin{bmatrix} r_{00}/s_x & r_{01}/s_y & r_{02}/s_z \\ r_{10}/s_x & r_{11}/s_y & r_{12}/s_z \\ r_{20}/s_x & r_{21}/s_y & r_{22}/s_z \end{bmatrix}$$

The implementation in `app/core/spatial.py` applies this in vectorized form:
```python
def apply_affine(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((len(points), 1))])  # (N, 4)
    transformed = (affine @ homogeneous.T).T                       # (N, 4)
    return transformed[:, :3]
```

> [!NOTE]
> This matrix acts as a universal coordinate translator. In a raw NIfTI file, voxels are stored like cells in a 3D spreadsheet (Row 50, Column 60, Depth 70). This matrix stretches cell coordinates by the voxel size (scale), rotates them to match how the patient's head was oriented in the scanner (rotation), and shifts them to the scanner's absolute center position (translation). The result is that every point maps to the exact same anatomical location in millimeters regardless of acquisition parameters.

### 3.2 Native VTK Slice Placement

When displaying an MRI brain scan, the software needs to show three orthogonal "slices" — like cutting the brain horizontally, vertically, and sideways at the same time. The challenge is that MRI scans are not always acquired with the same orientation: a patient's head may be slightly tilted in the scanner, or a different machine may store data in a different order. Naively re-rotating every slice pixel-by-pixel on the CPU every time the user scrolls would be extremely slow.

**The smarter approach — letting the graphics engine do the work:**

Instead of manually computing rotated images, NeuroNautics extracts the three pieces of positioning information already present in the affine matrix (see §3.1) and hands them directly to VTK, the 3D rendering library:

- **Origin** — where in 3D space the very first voxel of the MRI sits (think: the "anchor point" of the entire volume).
- **Spacing** — how many millimeters wide/tall/deep each individual voxel is (e.g., 1.0 mm × 1.0 mm × 1.2 mm).
- **Direction cosines** — three unit vectors that describe how the rows, columns, and slices of the scan are oriented relative to the real-world axes (Right, Anterior, Superior). This is the compact representation of the rotation extracted from the affine matrix.

```python
vtk_image = vtk.vtkImageData()
vtk_image.SetDimensions(nx, ny, nz)          # number of voxels on each axis
vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])   # physical size of each voxel (mm)
vtk_image.SetOrigin(origin[0], origin[1], origin[2])       # world position of voxel (0,0,0)

# Load direction cosines (VTK ≥ 9.0)
m = vtk.vtkMatrix3x3()
for i in range(3):
    for j in range(3):
        m.SetElement(i, j, dirs[i, j])
vtk_image.SetDirectionMatrix(m)
```

Once VTK knows the origin, spacing, and orientation, it handles all slice placement on the GPU automatically. The `vtkImageSliceMapper` uses this information to cut the volume at exactly the right angle and position — no manual rotation code required.

**Memory layout:** VTK reads voxel intensity values in a specific order (column-major, or "x-fastest"). The voxel array is rearranged to match this expectation before being handed to VTK, to avoid invisible data corruption.

---

**Intensity windowing — automatic contrast adjustment:**

Raw MRI voxel values span a very wide numerical range. If displayed linearly, the image would look washed out or completely dark — much like a photo taken with incorrect exposure. To fix this automatically, the software computes the 1st and 99th percentile of all non-zero voxel intensities:

- Anything below the 1st percentile is treated as pure black (background noise).
- Anything above the 99th percentile is treated as pure white (outlier bright spots).
- Everything in between is mapped linearly to the visible grayscale range.

```python
wmin, wmax = volume.percentile_window(1.0, 99.0)
prop.SetColorWindow(wmax - wmin)   # the total contrast range
prop.SetColorLevel((wmax + wmin) / 2.0)  # the midpoint of the range
```

### 3.3 MRtrix MIF Parsing

MRtrix `.mif` files have complex, variable text-based headers with layout permutations and axis flips that make direct parsing error-prone. Two strategies handle them:

**Primary method (`nifti_loader.py`):** Run MRtrix3's `mrconvert` via subprocess to convert `.mif` to a temporary `.nii.gz` file, then load with NiBabel. This delegates the layout/orientation logic entirely to MRtrix3's own tools.

```python
subprocess.run(
    ["mrconvert", str(path), tmp_nii, "-force"],
    check=True, env=env,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
```

**Windows compatibility trick:** MRtrix3 command-line tools crash on Windows if the `HOME` environment variable is not set. The loader intercepts this:
```python
env = os.environ.copy()
if "HOME" not in env:
    env["HOME"] = env.get("USERPROFILE", os.path.expanduser("~"))
```

**Fallback parser (`nifti_loader.py` → `_parse_mif_binary`):** If `mrconvert` is not installed, a built-in binary MIF parser reads the text header directly (key-value pairs until `END`), resolves axis permutations from the `layout` field (e.g. `-0,+2,-1`), applies axis flips, and transposes the data back to standard X,Y,Z order.

### 3.4 TCK Streamline Alignment

TCK streamline files encode raw millimeter coordinate arrays. The alignment is handled by `app/io/tck_loader.py`, which uses two strategies depending on what metadata the file provides:

1. **Header-based alignment:** The loader reads `vox_to_ras` and `dimensions` from the TCK header, constructs a minimal NIfTI reference image with these values, and feeds it to DIPY's `load_tractogram(..., to_space=Space.RASMM)`. This resolves coordinates to the exact physical RAS+ space.

2. **Identity fallback:** If the TCK header lacks `vox_to_ras` or `dimensions` (legacy files), the loader falls back to a 256³ isotropic 1mm identity grid (`np.eye(4)`), parsing raw stored coordinates directly as world millimeter values.

### 3.5 Camera Initialization from Affine

On startup, the camera is automatically oriented to show the left side of the brain. The code extracts the three anatomical direction vectors (Right, Anterior, Superior) from the affine's direction cosines:

```python
# Identify which affine column corresponds to each world axis
col_r = argmax(|dirs[0, :]|)   # column most aligned with X (Right)
col_a = argmax(|dirs[1, :]|)   # column most aligned with Y (Anterior)
col_s = argmax(|dirs[2, :]|)   # column most aligned with Z (Superior)

# Position camera looking from the left (negative Right direction)
camera.SetPosition(-r_vec)
camera.SetViewUp(s_vec)
```

This works correctly regardless of how the original scan was oriented in the scanner.

---

## 4. Parcellation Mapping & KD-Trees

Parcellation assigns each spatial point (surface vertex, voxel center, streamline point) to an anatomical region label. This mapping is the foundation of the exploded view, region picking, and voxel hiding.

### 4.1 The Boundary Problem

Surface mesh vertices do not align perfectly with discrete voxel boundaries due to mesh smoothing and interpolation. Direct coordinate lookup (rounding to the nearest voxel index) frequently classifies border vertices as background (label 0), producing gaps and visual artifacts.

### 4.2 World-Space KD-Tree Solution

**What is a "background voxel"?** In a parcellation volume, every voxel is assigned an integer label: label 1 might be "left frontal cortex", label 42 might be "right hippocampus", etc. But the majority of voxels in the volume are *empty space* — air around the skull, ventricles, or regions outside the brain. These are assigned label 0 (background). The problem is that when a surface mesh vertex sits near the edge of a brain region, a naive nearest-voxel lookup often snaps to one of these label-0 background voxels, making the vertex "disappear" from any region — creating visible holes in the mesh.

NeuroNautics resolves this by building a `cKDTree` (a spatial search structure that efficiently finds the closest point in a large dataset) of all **non-zero** labeled voxels in world space, and querying each vertex's nearest neighbor:

```python
# Build tree from labeled voxels only (skip background)
nonzero_indices = np.argwhere(data > 0)                       # only voxels with a real label
world_coords = apply_affine(parcellation_vol.affine, nonzero_indices)  # convert to mm
tree = cKDTree(world_coords)                                  # build spatial index

# Match each point to the nearest labeled voxel
_, nearest_idx = tree.query(points, k=1)                      # find closest real voxel
labels = nonzero_labels[nearest_idx]                          # assign its label
```

By excluding background voxels from the search tree, border vertices are always assigned to the nearest *real* anatomical region rather than falling into background gaps. The implementation lives in `app/io/parcellation_mapper.py`.

### 4.3 Mesh Parcellation Splitting

When a parcellation is available, the surface mesh is split into separate VTK actors per label to enable per-region manipulation (exploded views, individual deletion).

**The label averaging problem:** In a triangle mesh, labels are stored *per vertex* (per corner point). But a single triangle can have three vertices belonging to three different brain regions. If we naively average those labels to decide which region the triangle belongs to (e.g. averaging labels 10, 12, and 14 into label 12), the triangle would be assigned to the wrong region — or to a region that doesn't even exist. This is especially problematic at region borders, where adjacent triangles share vertices across two labels.

The solution uses PyVista's `point_data_to_cell_data` with `categorical=True`. Instead of averaging the vertex labels, this assigns each triangle (cell) the label that appears most often among its vertices — a "majority vote" approach. This ensures that every triangle gets a valid, real label, even at boundaries:

```python
polydata.point_data["Labels"] = mesh.labels
polydata = polydata.point_data_to_cell_data(categorical=True)  # majority vote, no averaging

for label in unique_labels:
    sub_mesh = polydata.threshold([label, label], scalars="Labels", preference="cell")
    sub_mesh_poly = sub_mesh.extract_surface(algorithm="dataset_surface")
    # Build vtkActor for this region...
```

**Phong shading** controls how each mesh surface reacts to light, making it look like a solid 3D object rather than a flat shape. It works by combining three lighting components:

- **Ambient (0.1)** — a small constant base brightness, simulating indirect light bouncing off walls. Ensures surfaces are never completely black even when facing away from the light.
- **Diffuse (0.7)** — the main "matte" illumination. Surfaces facing the light source appear bright; surfaces angled away appear dark. This is what gives the brain its 3D shape perception.
- **Specular (0.2)** — a subtle shiny highlight where light reflects directly toward the camera, like a faint gloss. Adds realism without making the surface look metallic.

Colors are either loaded from the manifest's `mesh_lut` (a dictionary mapping each label number to a specific RGB color) or generated procedurally using HSV with a seeded random hue per label to ensure consistent colors across sessions.

---

## 5. Rendering Algorithms

### 5.1 Tractography Direction Colorization

Streamline fibers are colored by their local tangent direction — the standard DTI convention. For a streamline containing points $\mathbf{p}_0, \mathbf{p}_1, \dots, \mathbf{p}_N$, the tangent at point $i$ is:

$$\mathbf{t}_i = \mathbf{p}_{i+1} - \mathbf{p}_i$$

The normalized absolute direction maps to RGB:

$$\text{Color}(i) = \begin{bmatrix} R \\ G \\ B \end{bmatrix} = \begin{bmatrix} |\hat{t}_{i, x}| \\ |\hat{t}_{i, y}| \\ |\hat{t}_{i, z}| \end{bmatrix} \times 255$$

This encodes anatomical directions as:
- **Red:** Left-Right axis ($X$)
- **Green:** Anterior-Posterior axis ($Y$)
- **Blue:** Superior-Inferior axis ($Z$)

Colors are stored as `vtkUnsignedCharArray` with lighting disabled (`LightingOff()`) so the RGB values display exactly as computed, unaffected by light position.

### 5.2 Voxel Point Cloud & Quadratic Contrast

The `VoxelModelRenderer` converts a structural volume into an interactive point cloud:

**Stride decimation:** To manage the ~16 million voxels in a typical 256³ MRI volume, the renderer samples every $s$-th voxel along each axis ($s=4$ by default), reducing the point count by $s^3 = 64\times$. Only voxels above a 5th-percentile intensity threshold are kept, removing background noise.

**Quadratic contrast curve (γ = 2.0):** MRI intensities are mapped to grayscale using a power-law transfer function that emphasizes white matter over gray matter:

$$I_{\text{mapped}} = \left(\frac{I - I_{\text{black}}}{I_{\text{white}} - I_{\text{black}}}\right)^{2.0}$$

This curve is implemented as a `vtkColorTransferFunction` with five evenly spaced sample points:
```python
for pct in [0.0, 0.25, 0.5, 0.75, 1.0]:
    val = s_min + pct * (s_max - s_min)
    intensity = pct ** 2.0  # quadratic gamma
    lut.AddRGBPoint(val, intensity, intensity, intensity)
```

Users can dynamically adjust `s_min` (black cutoff) and `s_max` (white cutoff) via UI sliders, updating the transfer function in real time.

### 5.3 Tumor Spiderweb (Dijkstra Shortest-Path Tree)

The `TumorOverlayRenderer` converts a binary lesion mask into a 3D network:

```
Step 1: SAMPLE           Step 2: KD-TREE          Step 3: DIJKSTRA
                                                   
 ░░██░░                  ●───●───●                     ●
 ░████░      →           │ ╲ │ ╱ │        →          ╱ │ ╲
 ░██░░░                  ●───●───●                  ●   ●   ●
                              (15-NN graph)          (shortest-path tree)
```

1. **Sampling:** Extract foreground voxel coordinates. If count exceeds `num_nodes` (default 1500), downsample randomly.
2. **Graph construction:** Build a $K$-D Tree and query the 15 nearest neighbors per point to create a distance-weighted adjacency graph:
   $$W_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|$$
3. **Root selection:** Find the point closest to the tumor's center of mass.
4. **Dijkstra's spanning tree:** Run Dijkstra's algorithm from the root. Connect each node to its predecessor, forming a minimum-distance spanning tree of `vtkLine` cells.
5. **Parcellation fragmentation:** If a parcellation is loaded, map tumor points to label zones and split the network into sub-actors (e.g. `tumor_overlay_42`) that move in sync with the corresponding mesh fragments during explosion.

> [!NOTE]
> This spiderweb representation is a choice we made when developing the software. This visualisation is a demo that can be changed in later versions if a better representation is necessary.

### 5.4 Connectome Graph

The connectome renderer creates a 3D node-edge graph:

**Nodes** are sphere glyphs (`vtkGlyph3D`) colored by connection degree using a blue → yellow → red transfer function:
- Blue $(0.1, 0.2, 0.8)$: low-degree peripheral nodes
- Yellow $(0.9, 0.9, 0.2)$: medium-degree nodes
- Red $(0.9, 0.1, 0.1)$: high-degree hub regions

**Edges** are drawn for the top-$K$ strongest connections (default $K=200$). Each edge is a VTK cylinder with weight-proportional radius:
$$r = 0.3 + 1.7 \times \frac{w}{w_{\max}}$$

All cylinder polydata is merged into a single actor via `vtkAppendPolyData`, reducing draw calls from hundreds of separate actors to one.

**Performance Trade-offs & Interaction Design**:
- **Single-Actor Merging (Draw Call Reduction)**: In a typical connectome display, drawing hundreds of individual cylinder actors creates a major CPU bottleneck, as the graphics pipeline must bind individual transforms and issue a draw call for every separate actor. Merging all cylinder geometries into a single `vtkPolyData` allows VTK to upload the entire edge network to the GPU as a single vertex buffer object (VBO), executing the draw in a single call and enabling smooth interactive framerates.
- **CPU-Side Geometry Recalculation**: Since all cylinders are merged into one static mesh, individual edges do not have their own transform matrices and cannot be shifted independently using simple actor transforms. Consequently, during dynamic changes—such as when triggering the radial "exploded view"—the node and edge endpoints must be recalculated on the CPU, and the combined geometry must be regenerated via `vtkAppendPolyData` and swapped back into the mapper.

### 5.5 Fibonacci Sphere Fallback

If parcellation centroids are unavailable, $N$ nodes are distributed uniformly on a sphere using the golden-angle algorithm:

$$z_i = 1 - \frac{2(i + 0.5)}{N}, \quad \phi_i = \arccos(z_i)$$
$$\theta_i = \frac{2\pi i}{\varphi}, \quad \text{where } \varphi = \frac{1 + \sqrt{5}}{2}$$
$$x_i = R \sin\phi_i \cos\theta_i, \quad y_i = R \sin\phi_i \sin\theta_i, \quad z_i = R \cos\phi_i$$

#### Mathematical Intuition (Golden section spiral)
The Fibonacci sphere algorithm (often called a Golden Section Spiral) distributes $N$ points almost perfectly evenly across the surface of a 3D sphere. It works by:
1. Vertically slicing the sphere from bottom to top, distributing the heights ($z_i$) linearly.
2. Wrapping a spiral around the vertical axis where each point is rotated relative to the last by the **golden angle** ($\approx 137.51^\circ$, or $\frac{2\pi}{\varphi}$). Because the golden ratio is highly irrational, consecutive points never align in radial columns, minimizing clustering and maximizing distance between all adjacent nodes.

#### Fallback Behavior & Parcellation Alignment
When a `--parcellation-volume` (or `--parcellation` CSV) is omitted, the application exhibits the following behaviors:
- **Anatomical Disconnection**: The connectome nodes are positioned at these calculated coordinates on a regular sphere of radius $R = 80\text{ mm}$ around the global origin instead of their true anatomical centres-of-mass.
- **Topology Preservation**: The underlying adjacency network topology (degree coloring, edges drawn between nodes $i$ and $j$, edge thickness based on connection strength) remains identical, but is mapped onto the abstract sphere.
- **Visual Misalignment**: If other layers (e.g., T1 MRI slices, voxel grids, or cortical meshes) are rendered simultaneously, the connectome graph will not align with them physically. The brain layers will render at their true coordinates, while the connectome will hover as a large, abstract sphere around the center.


### 5.6 Connectome Centroid Computation

When a `--parcellation-volume` is provided, `centroids_from_parcellation()` computes each node's position as the center-of-mass of its corresponding label region.

#### Mathematical Formulation

Because the voxel-to-world affine transformation is linear, the centroid in physical world space $\mathbf{C}_L \in \mathbb{R}^3$ for a parcellation label $L$ is computed by first calculating the center of mass in 3D voxel space and then transforming the resulting homogeneous coordinate using the affine matrix:

$$\mathbf{C}_L = \text{proj}_3 \left( \mathbf{A} \cdot \tilde{\mathbf{i}}_{\text{CoM}} \right)$$

where the voxel center of mass $\mathbf{i}_{\text{CoM}} \in \mathbb{R}^3$ is defined as:

$$\mathbf{i}_{\text{CoM}} = \frac{1}{|V_L|} \sum_{\mathbf{w} \in V_L} \mathbf{w}$$

and $\tilde{\mathbf{i}}_{\text{CoM}}$ is its homogeneous representation:

$$\tilde{\mathbf{i}}_{\text{CoM}} = \begin{bmatrix} \mathbf{i}_{\text{CoM}} \\ 1.0 \end{bmatrix} = \begin{bmatrix} i_{\text{CoM}} \\ j_{\text{CoM}} \\ k_{\text{CoM}} \\ 1.0 \end{bmatrix}$$

#### Explicit Term Definitions

- **$\mathbf{C}_L = [x_L, y_L, z_L]^T \in \mathbb{R}^3$**: The final 3D world-space coordinates (in RAS scanner millimeters) representing the centroid of parcellation label $L$.
- **$V_L$**: The set of all voxel coordinate index vectors $\mathbf{w} = [i, j, k]^T$ inside the parcellation grid that belong to parcel label $L$:
  $$V_L = \{ [i, j, k]^T \in \mathbb{Z}^3 \mid \text{data}[i, j, k] = L \}$$
- **$|V_L|$**: The cardinality of the set $V_L$ (i.e., the total number of voxels assigned to label $L$).
- **$\mathbf{A} \in \mathbb{R}^{4 \times 4}$**: The 4x4 voxel-to-world affine transformation matrix loaded from the parcellation image volume header.
- **$\text{proj}_3(\mathbf{x})$**: The projection operator that drops the homogeneous 4th coordinate to yield a 3D coordinate vector:
  $$\text{proj}_3 \left( \begin{bmatrix} x \\ y \\ z \\ 1.0 \end{bmatrix} \right) = \begin{bmatrix} x \\ y \\ z \end{bmatrix}$$

---

## 6. Exploded View Mathematics

The parcellation-based exploded view separates cortical regions radially from the brain's center of mass, revealing deep internal structures. All four visualization layers (mesh, voxels, streamlines, connectome) explode in sync.

```
                         Displaced Position (C'_L)
                               ┌──────────┐
                               │ Region L │
                               └────┬─────┘
                                    ▲
                                    │
                                    │ Displacement: f · D_max · d̂_L
                                    │
                                Centroid L (C_L)
                                    ●
                                   /
                                  /  Direction: d̂_L
                                 /
                                ● Global Centroid (C_global)
```

### 6.1 Radial Displacement Formulation

Let $\mathbf{C}_{\text{global}}$ be the brain's center of mass, $\mathbf{C}_L$ the centroid of region $L$.

1. **Direction vector:**
   $$\mathbf{d}_L = \mathbf{C}_L - \mathbf{C}_{\text{global}}$$

2. **Normalization** (with degeneracy guard):
   $$\hat{\mathbf{d}}_L = \begin{cases} \frac{\mathbf{d}_L}{\|\mathbf{d}_L\|}, & \text{if } \|\mathbf{d}_L\| > 0.001\text{ mm} \\ \mathbf{0}, & \text{otherwise} \end{cases}$$

3. **Displacement** given explosion factor $f \in [0, 1]$ and maximum distance $D_{\max} = 150$ mm:
   $$\text{Offset}(L) = f \cdot D_{\max} \cdot \hat{\mathbf{d}}_L$$

> [!NOTE]
> The global centroid is the brain's center. For each region, the algorithm draws a line from the brain center to that region's center. Sliding the explosion slider pushes each region outward along this line.

### 6.2 Implementation Per Layer

The four layers use different strategies to optimize for their data structures:

#### A. Mesh: Actor Translation (GPU-side)

Because each parcellation region is a separate `vtkActor`, the mesh renderer moves entire actors using `SetPosition()` — a GPU-side operation that avoids touching vertex data:

```python
offset = direction * (factor * max_dist)
actor.SetPosition(float(offset[0]), float(offset[1]), float(offset[2]))
actor.Modified()
```

Tumor overlay fragments are synchronized by matching label keys (`surface_mesh_42` → `tumor_overlay_42`) and applying the same position.

> [!IMPORTANT]
> VTK's Python wrappers silently fail when receiving NumPy scalar types instead of native Python floats for `SetPosition()`. All translation values are explicitly cast with `float()`.

#### B. Voxels & Streamlines: In-Place Point Array Modification (CPU → GPU)

For high-density point clouds (voxels) and poly-lines (streamlines), creating thousands of individual actors would cause major rendering lag. Instead, all coordinates reside in a single `vtkPoints` array within one actor. The points are displaced on the CPU using vectorized NumPy, then pushed to the GPU via direct pointer replacement:

```python
displaced = self._original_points.copy()

for lbl, centroid in label_centroids.items():
    mask = self._labels == lbl
    direction = centroid - global_centroid
    norm = np.linalg.norm(direction)
    if norm > 0.001:
        offset = (direction / norm) * (factor * max_dist)
        displaced[mask] += offset

vtk_data = numpy_support.numpy_to_vtk(
    np.ascontiguousarray(displaced, dtype=np.float32), deep=True
)
self._vtk_points.SetData(vtk_data)
self._vtk_points.Modified()
```

#### C. Connectome: Node Displacement + Edge Rebuild

Node spheres are displaced using the same `vtkPoints` in-place technique as voxels. Edges (cylinders) require geometric recalculation:

1. **Displaced endpoints:**
   $$\mathbf{p}_{1, \text{new}} = \mathbf{p}_1 + \text{Offset}(L_1), \quad \mathbf{p}_{2, \text{new}} = \mathbf{p}_2 + \text{Offset}(L_2)$$

2. **Midpoint translation:**
   $$\text{Mid} = \frac{\mathbf{p}_{1, \text{new}} + \mathbf{p}_{2, \text{new}}}{2}$$

3. **Rotation alignment:** Rotate each cylinder from its default Y-axis to align with the new edge direction vector $\mathbf{v}$:
   $$\text{axis} = \hat{\mathbf{y}} \times \hat{\mathbf{v}}, \quad \theta = \arctan2\left(\|\text{axis}\|, \hat{\mathbf{y}} \cdot \hat{\mathbf{v}}\right)$$
   
   Applied via `vtkTransform`:
   ```python
   transform = vtk.vtkTransform()
   transform.Translate(*mid)
   if cross_norm > 1e-6:
       transform.RotateWXYZ(angle_deg, *(cross / cross_norm))
   elif dot < 0:
       transform.RotateX(180.0)  # anti-parallel edge
   ```

   The rebuilt edge polydata is merged via `vtkAppendPolyData` and swapped into the existing mapper.

---

## 7. Interactive Picking & Region Hiding

### 7.1 Ctrl+Click Picking

To prevent selection events from interfering with normal camera orbits, picking only triggers when **Ctrl** is held during a left-click. The click coordinate is projected into the scene using `vtkPropPicker`:

```python
x, y = interactor.GetEventPosition()
hit = self._picker.Pick(x, y, 0, self._renderer)
if hit:
    actor = self._picker.GetViewProp()
    scene_key = self._actor_to_key.get(id(actor))  # e.g. "surface_mesh_42"
```

### 7.2 Highlight & State Caching

When a region is selected, its current color and ambient coefficient are cached, then overwritten with a bright yellow highlight:

```python
self._orig_color = tuple(vtk_prop.GetColor())   # cache original
self._orig_ambient = vtk_prop.GetAmbient()
vtk_prop.SetColor(1.0, 1.0, 0.0)                # yellow
vtk_prop.SetAmbient(0.6)                         # boost ambient
```

When selection is cleared or "Restore All" is pressed, the cached values are restored exactly.

### 7.3 NaN Sentinel Hiding

When a cortical region is deleted, the voxel model must hide its corresponding voxels. Rebuilding the point cloud array would cause visible lag. Instead, NeuroNautics exploits the fact that **VTK skips rendering points with NaN coordinates**:

```python
if self._hidden_labels:
    for label in self._hidden_labels:
        mask = self._labels == label
        displaced[mask] = np.nan

vtk_data = numpy_support.numpy_to_vtk(displaced, deep=True)
self._vtk_points.SetData(vtk_data)
```

This hides voxels in under 1 ms without modifying cell arrays, index buffers, or memory layout. The approach is compatible with the exploded view: `hide_label()` re-runs `explode()` with the stored parameters to apply NaN masking *after* displacement, keeping both features in sync.

---

## 8. Hand Tracking & Camera Kinematics

The hand-tracking subsystem lets users orbit, zoom, and reset the camera using webcam-detected hand gestures. It runs entirely in a background thread to keep the UI responsive.

```
   [OpenCV Frame Capture]
             │
             ▼
    [Horizontal Mirror (cv2.flip)]
             │
             ▼
  [MediaPipe HandLandmarker (2 hands, 21 landmarks each)]
             │
             ├──── right hand only ────► [landmarks_detected signal]
             │                                    │
             ├──── both hands ────────► [two_hands_detected signal]
             │                                    │
             └──── no hands ──────────► [hand_lost signal]
                                                  │
                                                  ▼
                                    [GestureMapper (classify + EMA smooth)]
                                                  │
                                                  ▼
                                    [CameraCommand → Rodrigues orbit math]
```

### 8.1 Horizontal Mirroring & Handedness Correction

Webcam frames are mirrored so left/right hand movement matches on-screen motion:
```python
frame = cv2.flip(frame, 1)
```

Because mirroring flips coordinate space, MediaPipe labels hands backwards. The worker corrects this:
```python
if category == "Left":     # MediaPipe says Left
    hand_map["right"] = lms  # but it's actually the user's right hand
else:
    hand_map["left"] = lms
```

### 8.2 Gesture Classification

The `GestureMapper` classifies each frame into one of five gestures using geometric heuristics on the 21 MediaPipe hand landmarks:

| Gesture | Detection Rule | Action |
| :--- | :--- | :--- |
| **Pinch** 🤏 | Thumb tip ↔ Index tip distance < 0.05 (normalized) | Grab & drag to orbit camera |
| **Open Palm** 🖐 | All 4 fingers + thumb extended (tip.y < pip.y) | Hold both palms for 1.5s → reset camera |
| **Fist** ✊ | No fingers extended, thumb not extended | Pause tracking (no-op) |
| **Point** ✋ | Default (none of the above) | Hovering — no camera action |
| **Two-Hand Zoom** 🔍 | Both hands pinching simultaneously | Hands apart = zoom in, together = zoom out |

Finger extension is detected by comparing the Y coordinate of each fingertip with its PIP joint — in camera space, lower Y means the finger is higher on screen:
```python
extended = sum(1 for tip, pip in zip(tips, pips) 
               if landmarks[tip].y < landmarks[pip].y)
```

### 8.3 Exponential Moving Average Smoothing

Raw hand coordinates are noisy due to camera jitter and micro-tremors. An EMA filter smooths all coordinate streams:

$$S_t = \alpha \cdot Y_t + (1 - \alpha) \cdot S_{t-1}$$

Where $\alpha = 0.6$ (default). This blends 60% of the new reading with 40% of the previous smoothed value.

> [!NOTE]
> Instead of snapping the camera instantly to each new hand position, the algorithm blends new coordinates (60% weight) with the previous position (40% weight). This dampens sudden jumps, producing smooth camera movement that still feels responsive.

**Drag anchoring:** When a pinch gesture starts, the smoothed position is initialized to the raw position (no blending lag). The camera delta is computed between the current smoothed position and the previous drag anchor. When the pinch is released, drag state is cleared to prevent velocity jumps on re-pinch.

### 8.4 Camera Orbit (Rodrigues' Rotation Formula)

Right-hand pinch coordinates $(\Delta x, \Delta y)$ are converted to camera rotation around the brain center using **Rodrigues' Rotation Formula**:

$$\mathbf{v}_{\text{rot}} = \mathbf{v} \cos\theta + (\mathbf{k} \times \mathbf{v}) \sin\theta + \mathbf{k}(\mathbf{k} \cdot \mathbf{v})(1 - \cos\theta)$$

Where $\mathbf{v}$ is the vector to rotate, $\mathbf{k}$ is the unit rotation axis, and $\theta$ is the rotation angle.

**Azimuth** (horizontal orbit around the vertical world axis $\mathbf{W}_{\text{up}} = [0, 0, 1]^T$):
$$\mathbf{p}_{\text{new}} = \text{Rotate}(\mathbf{Pos} - \mathbf{FP}, \, \mathbf{W}_{\text{up}}, \, \Delta_{\text{azimuth}})$$
$$\mathbf{Up}_{\text{new}} = \text{Rotate}(\mathbf{Up}, \, \mathbf{W}_{\text{up}}, \, \Delta_{\text{azimuth}})$$

**Elevation** (vertical orbit around the camera's local right axis $\mathbf{Right} = \hat{\mathbf{Look}} \times \mathbf{Up}$):
$$\mathbf{p}_{\text{final}} = \text{Rotate}(\mathbf{p}_{\text{new}}, \, \mathbf{Right}, \, \Delta_{\text{elevation}})$$
$$\mathbf{Up}_{\text{final}} = \text{Rotate}(\mathbf{Up}_{\text{new}}, \, \mathbf{Right}, \, \Delta_{\text{elevation}})$$


#### Anti-Inversion Control

When the camera orbits past the poles (upside-down, where $\mathbf{W}_{\text{up}} \cdot \mathbf{Up} < 0$), moving your hand right would rotate the scene left. NeuroNautics detects this and inverts the azimuth to keep controls intuitive:

```python
is_upside_down = sum(w_up[i] * up[i] for i in range(3)) < 0.0
azimuth_delta = -cmd.azimuth_delta if is_upside_down else cmd.azimuth_delta
```

Elevation is not inverted because the right-axis naturally flips when the camera is upside-down.

#### Roll Drift Correction

Accumulated floating-point errors cause the view-up vector to drift, gradually tilting the scene. The correction projects the world-up vector back onto the plane perpendicular to the look direction:

$$\mathbf{Up}_{\text{corrected}} = \mathbf{W}_{\text{up}} - (\mathbf{W}_{\text{up}} \cdot \hat{\mathbf{Look}}) \, \hat{\mathbf{Look}}$$
$$\mathbf{Up}_{\text{final}} = \frac{\mathbf{Up}_{\text{corrected}}}{\|\mathbf{Up}_{\text{corrected}}\|}$$

A pole singularity guard ($|\hat{\mathbf{Look}} \cdot \mathbf{W}_{\text{up}}| < 0.99$) prevents the correction from snapping when looking straight up or down. The direction of the corrected vector is matched to the current up vector (via dot product sign) to preserve upside-down orientation.

### 8.5 Two-Hand Zoom

When both hands are pinching, inter-hand distance drives zoom:

$$\text{Ratio} = \frac{d_t}{d_{t-1}}, \quad D_{\text{dolly}} = 1.0 + (\text{Ratio} - 1.0) \times 1.5$$

Where $d_t$ is the EMA-smoothed Euclidean distance between the two hands' pinch midpoints (thumb-index average per hand). Hands moving apart → $D_{\text{dolly}} > 1$ (zoom in). Together → $D_{\text{dolly}} < 1$ (zoom out).

While two-hand zoom is active, single-hand rotation is suppressed to prevent conflicting camera commands. Drag state is cleared on transition to prevent velocity jumps.

### 8.6 Camera Reset Gesture

When both hands show open palms simultaneously, a 1.5-second hold timer starts. A progress bar fills in the HUD. If held for the full duration, the camera resets to its default position (saved at startup). The timer resets if either hand changes gesture.

---

## 9. Performance Optimizations

NeuroNautics implements several targeted optimizations to maintain high framerates during interactive use:

| Category | Technique | Problem Solved | Benefit |
| :--- | :--- | :--- | :--- |
| **Voxel Rendering** | Flat vertex points (`SetVerts`) instead of `vtkGlyph3D` cubes | Instancing geometry for ~65K voxels overloaded the GPU pipeline | ~100× speedup; memory from ~500 MB to ~15 MB |
| **Voxel Hiding** | NaN coordinate sentinel masking | Rebuilding point arrays on hide caused visible lag | Hides voxels in <1 ms without modifying cell indices |
| **Exploded View** | In-place `vtkPoints.SetData()` pointer swap | Rebuilding VTK arrays on each slider tick caused lag | Avoids array reallocation; updates in <1 ms |
| **Explosion Dragging** | Phantom actor overlay during slider drag | Updating all 4 layers (mesh + voxels + streamlines + connectome) on every tick dropped framerate | Only lightweight mesh phantom actors update during drag; full sync on release. Maintains 60 FPS |
| **Connectome Edges** | `vtkAppendPolyData` merges all cylinders into one actor | Drawing 200 separate cylinder actors caused excessive draw calls | Single draw call for all edges |
| **Webcam Frames** | `.copy()` on `QImage` to decouple from numpy buffer | Frame array was overwritten before Qt finished painting | Prevents race conditions and visual glitches |
| **Video Feed** | Frame throttling to ~15 FPS (emit every 2nd frame) | Drawing camera feed at 30 FPS consumed too much main-thread time | Halves UI paint overhead while keeping video visually smooth |
| **Landmark Signal** | Throttle `landmarks_detected` to ~33 Hz | Emitting every frame (~30 FPS) could saturate the Qt signal queue | Prevents main-thread event queue backup |

### 9.1 Phantom Actor Optimization (Detail)

During explosion slider dragging, updating all four visualization layers for every slider tick (mesh repositioning + voxel displacement + streamline displacement + connectome rebuild) drops the framerate significantly. NeuroNautics solves this with a **phantom actor overlay**:

1. **On drag start (`sliderPressed`):** For each visible mesh region, create a lightweight phantom `vtkActor` sharing the same mapper (geometry) as the original, styled as a solid red outline at 35% opacity. Only these phantoms are added to the renderer.

2. **During drag:** Only the phantoms are repositioned (simple `SetPosition()` calls — pure GPU operations). The voxel, streamline, and connectome layers are frozen.

3. **On drag release (`sliderReleased`):** Remove all phantoms and perform a single full synchronization of all four layers at the final slider value.

This keeps the interaction at 60 FPS because phantom positioning is purely GPU-side, while the expensive CPU-to-GPU point array transfers only happen once on release.