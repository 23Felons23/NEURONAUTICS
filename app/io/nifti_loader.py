"""Volume loader for NIfTI (.nii) and MRtrix (.mif) files."""

from __future__ import annotations

from pathlib import Path
import nibabel as nib
import numpy as np

from app.core.volume import Volume


def load_nifti(path: str | Path) -> Volume:
    """Load a volume file (.nii, .nii.gz, .mif) and return a Volume.

    Parameters
    ----------
    path:
        Path to the volume file.

    Returns
    -------
    Volume
        Voxel data array (x, y, z) and the 4×4 voxel-to-world affine.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Volume file not found: {path}")

    ext = path.name.lower()
    if ext.endswith(".mif"):
        data, affine = load_mif(path)
    else:
        try:
            img = nib.load(str(path))
            data = np.asarray(img.dataobj, dtype=np.float32)
            affine = np.asarray(img.affine, dtype=np.float64)
        except Exception as exc:
            raise ValueError(f"Failed to load NIfTI file '{path}': {exc}") from exc

    return Volume(data=data, affine=affine, path=path)


def load_mif(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a MRtrix .mif file and return (data, affine).

    **Primary method:** Uses MRtrix3's ``mrconvert`` to convert the file to a
    temporary ``.nii.gz``, then loads it with NiBabel.  This delegates layout
    permutations and axis flips entirely to MRtrix3's own tools.

    **Fallback method:** If ``mrconvert`` is not installed, a built-in binary
    parser reads the MIF text header directly, resolves axis permutations from
    the ``layout`` field (e.g. ``-0,+2,-1``), applies axis flips, and
    transposes the data back to standard X,Y,Z order.

    Returns
    -------
    data : np.ndarray  — voxel data array, shape (X, Y, Z)
    affine : np.ndarray — (4, 4) float64 voxel-to-world matrix
    """
    import os
    import tempfile
    import subprocess

    path = Path(path)

    # --- Primary: mrconvert ------------------------------------------------
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_nii = os.path.join(tmpdir, "temp_parc.nii.gz")

            # MRtrix3 tools on Windows crash if HOME is not set
            env = os.environ.copy()
            if "HOME" not in env:
                env["HOME"] = env.get("USERPROFILE", os.path.expanduser("~"))

            subprocess.run(
                ["mrconvert", str(path), tmp_nii, "-force"],
                check=True,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            img = nib.load(tmp_nii)
            data = np.asarray(img.dataobj, dtype=np.float32)
            affine = np.asarray(img.affine, dtype=np.float64)

        return data, affine

    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # mrconvert unavailable — fall through to built-in parser
        pass

    # --- Fallback: built-in binary MIF parser ------------------------------
    return _parse_mif_binary(path)


def _parse_mif_binary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Built-in binary parser for MRtrix .mif files.

    Reads the text header (key-value lines until ``END``), resolves axis
    permutations from the ``layout`` field, applies axis flips, and
    transposes the data back to standard X,Y,Z order.

    Returns
    -------
    data : np.ndarray  — int32 voxel label array, shape (X, Y, Z)
    affine : np.ndarray — (4, 4) float64 voxel-to-world matrix
    """
    _MIF_DTYPES: dict[str, str] = {
        "Int8": "i1", "UInt8": "u1",
        "Int16LE": "<i2", "Int16BE": ">i2",
        "UInt16LE": "<u2", "UInt16BE": ">u2",
        "Int32LE": "<i4", "Int32BE": ">i4",
        "UInt32LE": "<u4", "UInt32BE": ">u4",
        "Float32LE": "<f4", "Float32BE": ">f4",
        "Float64LE": "<f8", "Float64BE": ">f8",
    }

    header: dict[str, list[str]] = {}

    with open(path, "rb") as fh:
        while True:
            line_bytes = fh.readline()
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if line == "END":
                break
            if ":" in line:
                key, _, val = line.partition(":")
                header.setdefault(key.strip(), []).append(val.strip())

    dim = [int(d) for d in header["dim"][0].split(",")][:3]  # (X, Y, Z)
    dtype_str = _MIF_DTYPES.get(header["datatype"][0], "<i4")

    # Transform rows → 4×4 affine (MRtrix stores 3 rows; last is [0,0,0,1])
    transform_rows = []
    for row_str in header.get("transform", []):
        transform_rows.append([float(v) for v in row_str.split(",")])
    if len(transform_rows) == 3:
        transform_rows.append([0.0, 0.0, 0.0, 1.0])
    affine = np.array(transform_rows, dtype=np.float64) if transform_rows else np.eye(4)

    # `file: . OFFSET` — byte offset of binary data within this file
    file_val = header.get("file", [". 0"])[0].split()
    data_offset = int(file_val[1]) if len(file_val) > 1 else 0

    # Layout: e.g. "-0,+2,-1" — axis permutation/flip
    layout_str = header.get("layout", ["+0,+1,+2"])[0]
    layout_strs = layout_str.split(",")
    layout = [int(s) for s in layout_strs]

    n_voxels = dim[0] * dim[1] * dim[2]
    with open(path, "rb") as fh:
        fh.seek(data_offset)
        raw = np.frombuffer(
            fh.read(n_voxels * np.dtype(dtype_str).itemsize), dtype=dtype_str
        )

    # Apply layout permutation (MRtrix stores axis order / flips in layout)
    abs_layout = [abs(l) for l in layout]
    
    # Stored shape on disk from slowest to fastest (matching C-contiguous order):
    # slowest (stride 2), middle (stride 1), fastest (stride 0)
    shape_stored = [
        dim[abs_layout.index(2)],
        dim[abs_layout.index(1)],
        dim[abs_layout.index(0)]
    ]
    data = raw.reshape(shape_stored, order="C")

    # Transpose to canonical X, Y, Z order.
    # The axes of data_stored are (slowest, middle, fastest).
    # So the axis of data_stored corresponding to canonical axis c is (2 - abs_layout[c]).
    perm = [2 - abs_layout[0], 2 - abs_layout[1], 2 - abs_layout[2]]
    data = np.transpose(data, perm)

    # Apply axis flips to the canonical axes based on layout signs.
    for c in range(3):
        if layout_strs[c].strip().startswith("-"):
            data = np.flip(data, axis=c)

    return data.astype(np.int32), affine
