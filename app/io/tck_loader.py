"""TCK (MRtrix) tractogram loader.

TCK files embed their own coordinate-space header, so **no external
reference volume is required**.  We read the raw header via nibabel to
reconstruct a minimal NIfTI reference image that DIPY accepts, then let
DIPY stream the points into a ``StatefulTractogram``.

If the TCK header does not contain a valid voxel grid (older files), we
fall back to a 1 mm³ isotropic identity space in RASmm, which keeps
coordinates in mm world-space exactly as stored in the file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.tractography import TractographyBundle


def load_tck(path: Path | str) -> TractographyBundle:
    """Load a ``.tck`` tractogram and return a ``TractographyBundle``.

    Parameters
    ----------
    path:
        Absolute or relative path to the ``.tck`` file.

    Returns
    -------
    TractographyBundle
        Streamlines in world (mm) space, as stored in the TCK header.

    Raises
    ------
    FileNotFoundError
        If *path* does not point to an existing file.
    ValueError
        If the file cannot be parsed as a valid tractogram.
    ImportError
        If nibabel or DIPY are not installed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TCK file not found: {path}")

    # ------------------------------------------------------------------ #
    # 1. Read the TCK header via nibabel to get the voxel grid / affine.  #
    # ------------------------------------------------------------------ #
    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("nibabel is required to load .tck files: pip install nibabel") from exc

    try:
        tck_img = nib.streamlines.load(str(path), lazy_load=True)
        header = tck_img.header
    except Exception as exc:
        raise ValueError(f"Could not read TCK header from {path}: {exc}") from exc

    # Pull voxel-to-rasmm affine from the header if available.
    vox_to_ras = header.get("vox_to_ras", None)
    dimensions  = header.get("dimensions", None)

    if (
        vox_to_ras is not None
        and not np.allclose(vox_to_ras, 0)
        and dimensions is not None
    ):
        ref_affine = np.asarray(vox_to_ras, dtype=np.float64)
        ref_shape  = tuple(int(d) for d in dimensions)
    else:
        # Fallback: 256³ isotropic 1 mm identity (RASmm = world mm coords)
        ref_affine = np.eye(4, dtype=np.float64)
        ref_shape  = (256, 256, 256)

    # Build a minimal NIfTI1Image that DIPY accepts as a reference.
    ref_data = np.zeros(ref_shape, dtype=np.uint8)
    reference = nib.Nifti1Image(ref_data, affine=ref_affine)

    # ------------------------------------------------------------------ #
    # 2. Load streamlines with DIPY in RASmm world space.                 #
    # ------------------------------------------------------------------ #
    try:
        from dipy.io.streamline import load_tractogram  # type: ignore
        from dipy.io.stateful_tractogram import Space   # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("DIPY is required to load .tck files: pip install dipy") from exc

    try:
        sft = load_tractogram(
            str(path),
            reference=reference,
            to_space=Space.RASMM,
            bbox_valid_check=False,
        )
    except Exception as exc:
        raise ValueError(f"Could not parse tractogram {path}: {exc}") from exc

    # DIPY returns False (not an SFT) when loading silently fails.
    if sft is False or sft is None:
        raise ValueError(
            f"DIPY could not load {path} — "
            "the file may be corrupt or not a valid TCK tractogram."
        )

    streamlines: list[np.ndarray] = [
        np.asarray(sl, dtype=np.float32) for sl in sft.streamlines
    ]
    return TractographyBundle(streamlines, affine=ref_affine)
