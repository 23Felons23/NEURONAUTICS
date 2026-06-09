# NeuroNautics — Multimodal Brain Data Navigator

NeuroNautics is a real-time 3D brain data navigator designed for neurosurgical planning and multimodal neuroimaging visualization. It co-registers and visualizes 3D MRI scans, white-matter tractography (streamlines), structural connectome graphs, cortical surface meshes, and tumor/lesion overlays in a single, coordinate-aligned interactive workspace.

---

## 1. Installation & Setup

### Prerequisites
- **Python:** version `3.10` or higher (recommended: `3.11` or `3.12`).
- **System Packages (Optional):**
  - **Webcam:** Required for webcam-based hand-tracking features.
  - **MRtrix3:** (Optional) If you plan to load `.mif` parcellation volumes, having `mrconvert` from [MRtrix3](https://www.mrtrix.org/) installed and available in your system `PATH` is recommended. If `mrconvert` is not found, NeuroNautics will automatically fall back to its built-in Python-native binary `.mif` parser (which should work fine).

### Installing MRtrix3 (Optional)
If you want to use the primary `mrconvert` tool for converting `.mif` files, you can install MRtrix3 using one of the following methods:

#### Windows
* **Using Conda (Recommended):**
  ```bash
  conda install -c mrtrix3 mrtrix3
  ```
* **Using MSYS2:** Follow the step-by-step MSYS2 installation tutorial on the [official MRtrix3 Windows guide](https://mrtrix.readthedocs.io/en/latest/installation/windows_install.html).

#### macOS
* **Using Homebrew:**
  ```bash
  brew tap MRtrix3/mrtrix3
  brew install mrtrix3
  ```
* **Using Standalone Package:** Download the `.pkg` installer directly from the [MRtrix3 Downloads page](https://www.mrtrix.org/download/).

#### Linux
* **Using Conda:**
  ```bash
  conda install -c mrtrix3 mrtrix3
  ```
* **Using APT (Ubuntu/Debian):** Follow the repository registration instructions on the [Downloads page](https://www.mrtrix.org/download/), then install via:
  ```bash
  sudo apt-get install mrtrix3
  ```

### Installation Options
We recommend using [uv](https://github.com/astral-sh/uv) or standard Python virtual environments. 

#### Option A: Development / Editable Install (Recommended)
1. Navigate to the cloned project directory (which contains `pyproject.toml`):
   ```bash
   # Replace this path with the path where you cloned the project on your machine
   cd cloned-repository
   ```
   
   > [!IMPORTANT]
   > Make sure you are in the directory containing `pyproject.toml`. You can verify this by running `dir` (on Windows) or `ls` (on macOS/Linux). If you do not run `pip install` from the directory containing `pyproject.toml`, pip will throw an error stating: `"The project does not appear to be a python project neither setup.py nor pyproject.toml found"`.

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv

   # Then :
   # on Windows (PowerShell):
   .venv\Scripts\activate

   # on Windows (Command Prompt):
   .venv\Scripts\Activate.bat

   # on macOS/Linux:
   source .venv/bin/activate
   ```
3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

#### Option B: Installing Extras (Hand Tracking & Testing)
To install package extras such as MediaPipe/OpenCV for gesture tracking, or pytest for development:
- **With Gesture/Hand-Tracking:**
  ```bash
  pip install -e .[tracking]
  ```
- **With Development & Test suites:**
  ```bash
  pip install -e .[dev]
  ```
- **Install All Packages:**
  ```bash
  pip install -e .[dev,tracking]
  ```

---

## 2. CLI Usage & Commands

After installing the package, a console script shortcut `neuronautics` is added to your path. You can run the application using either `neuronautics` or by invoking the module directly with `python -m app`.

### Command Syntax & Options
```bash
neuronautics [options]
# OR
python -m app [options]
```

| Argument / Flag | Destination / Variable | Type | Description |
| :--- | :--- | :--- | :--- |
| `--t1 PATH` | `t1` | `str` | Path to the T1-weighted structural MRI scan (`.nii` or `.nii.gz`). If omitted, a file picker dialog will pop up on launch. |
| `--manifest PATH` | `manifest` | `str` | Path to a patient dataset JSON manifest bundle. (Recommended for complex sessions, overrides CLI args). |
| `--tck PATH` | `tck` | `str` | Path to the tractography streamline fiber bundle file (`.tck`). |
| `--connectome PATH` | `connectome` | `str` | Path to the connectome adjacency connectivity matrix file (`.csv`). |
| `--parcellation-volume PATH` | `parcellation_volume` | `str` | Path to the parcellation label volume (`.nii`, `.nii.gz`, or `.mif`). Node centroids will be computed as the per-parcel centre-of-mass. |
| `--parcellation PATH` | `parcellation` | `str` | Legacy method: Path to a CSV file specifying pre-computed node centroids (`label,x,y,z`). *Preferred: Use `--parcellation-volume` instead.* |
| `--no-voxel` | `voxel` | Flag | Disables discrete voxel point-cloud grid visualization on startup (enabled by default). |
| `--voxel-stride N` | `voxel_stride` | `int` | Decimation stride factor for the point cloud (default: `4` i.e. samples every 4th voxel). |
| `--no-hand-tracking` | `hand_tracking` | Flag | Disables webcam-based hand gesture controls (which are enabled by default; requires `[tracking]` dependencies). |

### CLI Examples
*The manifest file is a better option most of the times, read below*

1. **Launch with T1 MRI only (opens interactive Orthogonal slice planes):**
   ```bash
   neuronautics --t1 data/mri/t1_brain.nii.gz
   ```

2. **Launch with T1 MRI and Tractography Overlay:**
   ```bash
   neuronautics --t1 data/mri/t1_brain.nii.gz --tck data/connectome/exemplars.tck
   ```

3. **Launch Connectome Graph with anatomically mapped nodes from parcellation volume:**
   ```bash
   neuronautics --t1 data/mri/t1_brain.nii.gz \
                --connectome data/connectome/connectome_fs.csv \
                --parcellation-volume data/connectome/parc_fs.mif
   ```

4. **Launch using a Patient Manifest file (Recommended):**
   ```bash
   neuronautics --manifest data/patient_manifest.json
   ```

---

## 3. The Manifest File (`manifest.json`)  * RECOMMENDED *

The dataset manifest is a JSON file that bundles all multi-modal neuroimaging datasets, rendering styles, colors, and thresholds for a dataset. 

### Why Use a Manifest?
- **Relative Path Resolution:** All data file paths are resolved relative to the manifest file's containing directory. This makes patient data folders fully portable.
- **Custom Coloring:** Allows overriding default colors for meshes, tumor overlays, and parcellated regions.
- **Color Normalization:** If colors are entered in standard 0–255 RGB integer formats, the loader automatically detects values greater than 1.0 and normalizes them to VTK's 0.0–1.0 float ranges.
- **Persistent Contrast Configurations:** Pre-configure MRI intensity cutoffs and contrast parameters.

### Complete `manifest.json` Example

Place this file (e.g., `manifest.json`) in the same root folder as your dataset `mri/` and `connectome/` subdirectories:

```json
{
  "t1": "mri/t1_brain.nii.gz",
  "tck": "connectome/exemplars.tck",
  "connectome": "connectome/connectome_fs.csv",
  "parcellation_volume": "connectome/parc_fs.mif",
  "tumor": "tumor_fake.nii.gz",
  "mesh": "connectome/mesh_smoothed.obj",
  
  "voxel_stride": 4,
  "voxel_point_size": 3.0,
  "voxel_spheres": true,
  "voxel_black_value": 1000.0,
  "voxel_white_value": 2500.0,
  
  "mesh_color": [180, 180, 200],
  "mesh_lut": {
    "11": [255, 100, 100],
    "12": [100, 255, 100],
    "13": [100, 100, 255],
    "17": [240, 240, 50],
    "18": [240, 50, 240]
  },
  
  "tumor_color": [255, 0, 0],
  "tumor_opacity": 0.65,
  "tumor_line_width": 2.5,
  "tumor_num_nodes": 1200
}
```

### Manifest Key Reference

#### Data Paths (Paths resolved relative to the manifest folder)
* **`t1`** (`string`, optional): Path to the T1-weighted NIfTI image (`.nii` / `.nii.gz`).
* **`tck`** (`string`, optional): Path to the tractography streamline bundle (`.tck`).
* **`connectome`** (`string`, optional): Path to the connectivity CSV matrix (`.csv`).
* **`parcellation_volume`** (`string`, optional): Path to the parcellation labels volume (`.nii` / `.nii.gz` / `.mif`). Also supports key name alias: `parcellation`.
* **`tumor`** (`string`, optional): Path to a binary tumor/lesion mask volume (`.nii` / `.nii.gz`).
* **`mesh`** (`string`, optional): Path to the cortical surface/subcortical `.obj` mesh file.

#### Voxel Rendering Settings
* **`voxel_stride`** (`int`, default: `4`): Decimation factor. Samples every $N$-th voxel in the cloud.
* **`voxel_point_size`** (`float`, default: `3.0`): Graphical point size of each voxel in the point cloud.
* **`voxel_spheres`** (`bool`, default: `false`): Renders voxels as spheres if `true`, or standard square pixels if `false`.
* **`voxel_black_value`** (`float`, optional): Lower boundary threshold for the quadratic contrast transfer function. Voxel intensities below this are clipped to black.
* **`voxel_white_value`** (`float`, optional): Upper boundary threshold for the contrast transfer function. Voxel intensities above this are clipped to white.

#### Mesh & Region Coloring
* **`mesh_color`** (`array` of 3 floats/ints, optional): Fallback RGB color for the cortical mesh surface.
* **`mesh_lut`** (`object`, optional): A dictionary mapping stringified region labels to custom 3-element RGB arrays. Used to color specific parcellated regions of interest.

#### Tumor Overlay Settings
* **`tumor_color`** (`array` of 3 floats/ints, default: `[255, 0, 0]`): RGB color for the tumor network edges.
* **`tumor_opacity`** (`float`, default: `0.6`): Opacity transparency level of the tumor visualization.
* **`tumor_line_width`** (`float`, default: `1.5`): Line width weight of the tumor network segments.
* **`tumor_num_nodes`** (`int`, default: `1500`): Maximum target nodes to sample from the tumor mask boundary for Dijkstra network construction.

> **Note**: A fake tumor is available in the repo under the name `tumor_fake.nii.gz`. You can use it to test the tumor visualization capabilities of the app by adding it to your `manifest.json`.

---

## 4. Interaction & Controls

### Keyboard & Mouse
- **Left-Click + Drag:** Orbit camera.
- **Scroll Wheel / Right-Click + Drag:** Zoom camera in/out.
- **Ctrl + Left-Click:** Pick region of interest. Highlights the selected parcellated region in yellow, displays its details in the info panel, and enables you to hide it.
- **Explode Slider:** Pushes parcellated region components radially outwards from the brain's center of mass, allowing visual inspection of internal white-matter structures.

### Webcam Gesture Controls (Enabled by Default)
Webcam-based hand tracking gesture controls are enabled by default. To run NeuroNautics without hand tracking, launch it with the `--no-hand-tracking` flag.
When active, the camera feed shows in a glassmorphism floating panel. Use the following gestures with your hand visible in the webcam:

1. **Pinch (Index to Thumb)** 🤏: Hold right-hand pinch and drag left/right/up/down to orbit the camera view.
2. **Two-Hand Pinch** 🔍: Pinch with both hands. Separate hands to zoom in (dolly), or pull hands closer to zoom out.
3. **Open Palm** 🖐: Hold both palms open and facing the camera for 1.5 seconds to reset the camera back to the default sagittal view.
4. **Fist** ✊: Pause tracking and lock camera rotation/zoom.
5. **Point / Open Hand** ✋: Hovering mode.

---

## 5. Development & Running Tests

The project includes an extensive suite of unit and integration tests covering coordinate transformations, file parsing, and scene management.

Run tests using Pytest:
```bash
# Verify setup and run the test suite
pytest -v
```
