"""Application entry point.

Usage
-----
    # Run directly (T1 only):
    python -m app --t1 path/to/brain.nii.gz

    # With tractography overlay:
    python -m app --t1 path/to/brain.nii.gz --tck path/to/tract.tck

    # Legacy positional arg still works:
    python -m app path/to/brain.nii.gz

    # After pip install:
    neuronautics --t1 path/to/brain.nii.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from app.io.nifti_loader import load_nifti
from app.ui.main_window import MainWindow


def _pick_file() -> Path | None:
    """Open a file-chooser dialog to select a NIfTI file."""
    path, _ = QFileDialog.getOpenFileName(
        None,
        "Open NIfTI file",
        "",
        "NIfTI files (*.nii *.nii.gz)",
    )
    return Path(path) if path else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="neuronautics",
        description="Multimodal brain data navigator",
    )
    # Support both positional (legacy) and --t1 flag
    parser.add_argument(
        "t1_positional",
        nargs="?",
        metavar="T1",
        help="Path to T1 NIfTI file (positional, deprecated — use --t1).",
    )
    parser.add_argument(
        "--t1",
        dest="t1",
        metavar="PATH",
        help="Path to T1 NIfTI file.",
    )
    parser.add_argument(
        "--manifest",
        dest="manifest",
        metavar="PATH",
        help="Path to JSON manifest file.",
    )
    parser.add_argument(
        "--tck",
        dest="tck",
        metavar="PATH",
        help="Path to tractogram (.tck) file (optional).",
    )
    parser.add_argument(
        "--voxel",
        dest="voxel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable discrete voxel cube visualization (default: enabled; stride=4 by default).",
    )
    parser.add_argument(
        "--voxel-stride",
        dest="voxel_stride",
        type=int,
        default=4,
        metavar="N",
        help="Sample every N voxels along each axis for voxel model (default: 4).",
    )
    parser.add_argument(
        "--connectome",
        dest="connectome",
        metavar="PATH",
        help="Path to connectome adjacency matrix (.csv).",
    )
    parser.add_argument(
        "--parcellation",
        dest="parcellation",
        metavar="PATH",
        help="Path to parcellation centroid CSV file (label,x,y,z). "
             "Optional — use --parcellation-volume instead for .nii/.mif files.",
    )
    parser.add_argument(
        "--parcellation-volume",
        dest="parcellation_volume",
        metavar="PATH",
        help="Path to parcellation label volume (.nii, .nii.gz, .mif). "
             "Node centroids are computed as per-parcel centre-of-mass in world space. "
             "Use parc_fs.mif from your connectome folder for anatomically correct node positions.",
    )
    parser.add_argument(
        "--hand-tracking",
        dest="hand_tracking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable hand tracking gesture controls (default: enabled; requires webcam, mediapipe, opencv).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the NeuroNautics application."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    args = _parse_args()

    # Load manifest if provided
    bundle = None
    if args.manifest:
        from app.io.manifest import load_manifest
        try:
            bundle = load_manifest(args.manifest)
        except Exception as exc:
            QMessageBox.critical(None, "NeuroNautics — Manifest Error", f"Could not load manifest:\n{exc}")
            sys.exit(1)
            
    # Resolve paths from manifest or args
    if bundle and bundle.tck:
        args.tck = bundle.tck
    if bundle and bundle.connectome:
        args.connectome = bundle.connectome
        args.parcellation_volume = bundle.parcellation_volume

    # Resolve T1 path: manifest wins over --t1 wins over positional fallback
    if bundle and bundle.t1:
        nifti_path = Path(bundle.t1)
    else:
        t1_str = args.t1 or args.t1_positional
        if t1_str:
            nifti_path = Path(t1_str)
        else:
            nifti_path = _pick_file()
            if nifti_path is None:
                sys.exit(0)

    # Load the volume
    try:
        volume = load_nifti(nifti_path)
    except (FileNotFoundError, ValueError) as exc:
        QMessageBox.critical(None, "NeuroNautics — Load Error", str(exc))
        sys.exit(1)

    print(f"Loaded: {volume}")

    # Parse parcellation volume early so it can be shared with all renderers
    parc_vol = None
    parc_vol_path = getattr(args, "parcellation_volume", None)
    if parc_vol_path:
        try:
            parc_vol = load_nifti(Path(parc_vol_path))
        except Exception as exc:
            print(f"Warning: Could not load parcellation volume: {exc}")

    # Optionally load tractography
    tract_renderer = None
    if args.tck:
        tck_path = Path(args.tck)
        try:
            from app.io.tck_loader import load_tck
            from app.viz.tractography_renderer import TractographyRenderer

            tck_bundle = load_tck(tck_path)
            tract_renderer = TractographyRenderer.from_bundle(tck_bundle, parc_vol=parc_vol)
            print(f"Loaded: {tck_bundle}")
        except (FileNotFoundError, ValueError, ImportError) as exc:
            QMessageBox.warning(
                None,
                "NeuroNautics — Tractography Warning",
                f"Could not load tractogram:\n{exc}",
            )

    # Optionally load voxel model
    voxel_renderer = None
    if args.voxel or (bundle is not None):
        from app.viz.voxel_model import VoxelModelRenderer
        
        # Load defaults from bundle if available, else from command line args
        stride = bundle.voxel_stride if (bundle is not None) else args.voxel_stride
        point_size = bundle.voxel_point_size if (bundle is not None) else 3.0
        black_val = bundle.voxel_black_value if (bundle is not None) else None
        white_val = bundle.voxel_white_value if (bundle is not None) else None
        spheres = bundle.voxel_spheres if (bundle is not None) else False
        
        voxel_renderer = VoxelModelRenderer.from_volume(
            volume,
            stride=stride,
            parc_vol=parc_vol,
            point_size=point_size,
            black_value=black_val,
            white_value=white_val,
            render_spheres=spheres,
        )
        print(f"Voxel model: stride={stride}, point_size={point_size}, spheres={spheres}")


    # Optionally load connectome graph
    connectome_renderer = None
    if args.connectome:
        try:
            import numpy as np
            from app.io.csv_loader import load_csv
            from app.viz.connectome_graph import ConnectomeGraphRenderer, centroids_from_parcellation

            matrix = load_csv(Path(args.connectome))
            centroids = None

            if parc_vol_path:
                print(f"Computing parcel centroids from {parc_vol_path} ...")
                centroids = centroids_from_parcellation(
                    parc_vol_path, matrix.node_count
                )
                print(f"  → {matrix.node_count} parcel centroids extracted")
            elif args.parcellation:
                # Legacy: centroid CSV (label, x, y, z)
                raw = np.loadtxt(
                    args.parcellation, delimiter=",",
                    skiprows=1, usecols=(1, 2, 3), dtype=np.float32,
                )
                centroids = raw

            connectome_renderer = ConnectomeGraphRenderer.from_matrix(
                matrix, centroids=centroids
            )
            print(f"Connectome: {matrix}")
        except (FileNotFoundError, ValueError, ImportError) as exc:
            QMessageBox.warning(
                None,
                "NeuroNautics — Connectome Warning",
                f"Could not load connectome:\n{exc}",
            )

    # Optionally load tumor overlay
    tumor_renderer = None
    if bundle and bundle.tumor:
        try:
            from app.viz.tumor_overlay import TumorOverlayRenderer
            kwargs = {}
            if bundle.tumor_color is not None:
                kwargs['color'] = bundle.tumor_color
            if bundle.tumor_opacity is not None:
                kwargs['opacity'] = bundle.tumor_opacity
            if bundle.tumor_line_width is not None:
                kwargs['line_width'] = bundle.tumor_line_width
            if bundle.tumor_num_nodes is not None:
                kwargs['num_nodes'] = bundle.tumor_num_nodes
            tumor_renderer = TumorOverlayRenderer.from_nifti(bundle.tumor, parc_vol=parc_vol, **kwargs)
            print(f"Loaded Tumor Overlay: {bundle.tumor}")
        except Exception as exc:
            print(f"Warning: Could not load tumor overlay: {exc}")

    # Optionally load surface mesh
    mesh_renderer = None
    if bundle and bundle.mesh:
        try:
            from app.io.obj_loader import load_obj
            from app.viz.mesh_renderer import MeshRenderer
            from app.io.parcellation_mapper import map_parcellation_to_mesh
            mesh = load_obj(bundle.mesh)
            
            # Map parcellation
            if parc_vol:
                try:
                    mesh = map_parcellation_to_mesh(mesh, parc_vol)
                    print(f"Mapped parcellation from {parc_vol_path} to mesh")
                except Exception as map_exc:
                    print(f"Warning: Could not map parcellation: {map_exc}")

            kwargs = {}
            if bundle.mesh_color is not None:
                kwargs["color"] = bundle.mesh_color
            if bundle.mesh_lut is not None:
                kwargs["lut"] = bundle.mesh_lut

            mesh_renderer = MeshRenderer.from_mesh(mesh, **kwargs)
            print(f"Loaded Surface Mesh: {bundle.mesh}")
        except Exception as exc:
            print(f"Warning: Could not load surface mesh: {exc}")

    # Build and show the window
    window = MainWindow(
        volume,
        tract_renderer=tract_renderer,
        voxel_renderer=voxel_renderer,
        connectome_renderer=connectome_renderer,
        tumor_renderer=tumor_renderer,
        mesh_renderer=mesh_renderer,
        hand_tracking=args.hand_tracking,
    )
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
