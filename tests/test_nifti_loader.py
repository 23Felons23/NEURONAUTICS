"""Tests for app.io.nifti_loader."""

from __future__ import annotations

import gzip
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from app.io.nifti_loader import load_nifti
from app.core.volume import Volume


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_nifti(tmp_path: Path, shape=(8, 10, 6), suffix=".nii") -> tuple[Path, np.ndarray, np.ndarray]:
    """Write a minimal NIfTI-1 file and return (path, data, affine)."""
    rng = np.random.default_rng(0)
    data = rng.random(shape).astype(np.float32)
    affine = np.diag([2.0, 2.0, 3.0, 1.0])  # 2 × 2 × 3 mm voxels
    img = nib.Nifti1Image(data, affine)
    path = tmp_path / f"test{suffix}"
    nib.save(img, str(path))
    return path, data, affine


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestLoadNiftiSuccess:
    def test_returns_volume(self, tmp_path):
        path, _, _ = _write_nifti(tmp_path)
        vol = load_nifti(path)
        assert isinstance(vol, Volume)

    def test_shape_preserved(self, tmp_path):
        shape = (8, 10, 6)
        path, _, _ = _write_nifti(tmp_path, shape=shape)
        vol = load_nifti(path)
        assert vol.shape == shape

    def test_data_dtype_is_float32(self, tmp_path):
        path, _, _ = _write_nifti(tmp_path)
        vol = load_nifti(path)
        assert vol.data.dtype == np.float32

    def test_affine_dtype_is_float64(self, tmp_path):
        path, _, _ = _write_nifti(tmp_path)
        vol = load_nifti(path)
        assert vol.affine.dtype == np.float64

    def test_affine_values_match(self, tmp_path):
        path, _, affine = _write_nifti(tmp_path)
        vol = load_nifti(path)
        np.testing.assert_allclose(vol.affine, affine, atol=1e-6)

    def test_data_values_match(self, tmp_path):
        path, data, _ = _write_nifti(tmp_path)
        vol = load_nifti(path)
        np.testing.assert_allclose(vol.data, data, atol=1e-6)

    def test_path_attribute_set(self, tmp_path):
        path, _, _ = _write_nifti(tmp_path)
        vol = load_nifti(path)
        assert vol.path == path

    def test_accepts_string_path(self, tmp_path):
        path, _, _ = _write_nifti(tmp_path)
        vol = load_nifti(str(path))
        assert isinstance(vol, Volume)

    def test_voxel_sizes_from_affine(self, tmp_path):
        path, _, _ = _write_nifti(tmp_path)
        vol = load_nifti(path)
        # affine diagonal is [2, 2, 3, 1]
        np.testing.assert_allclose(vol.voxel_sizes_mm, [2.0, 2.0, 3.0], atol=1e-5)

    def test_gz_compressed(self, tmp_path):
        """load_nifti must handle .nii.gz files."""
        path, data, _ = _write_nifti(tmp_path, suffix=".nii.gz")
        vol = load_nifti(path)
        assert vol.shape == data.shape


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------

class TestLoadNiftiErrors:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Volume file not found"):
            load_nifti(tmp_path / "ghost.nii")

    def test_corrupt_file_raises_value_error(self, tmp_path):
        bad = tmp_path / "bad.nii"
        bad.write_bytes(b"this is not a nifti file")
        with pytest.raises(ValueError, match="Failed to load NIfTI"):
            load_nifti(bad)

    def test_empty_file_raises_value_error(self, tmp_path):
        empty = tmp_path / "empty.nii"
        empty.write_bytes(b"")
        with pytest.raises(ValueError, match="Failed to load NIfTI"):
            load_nifti(empty)


class TestLoadMifFallback:
    def _write_mif(
        self,
        tmp_path: Path,
        filename: str,
        dim: tuple[int, int, int],
        layout_str: str,
        canonical_data: np.ndarray,
        transform_rows: list[list[float]],
    ) -> Path:
        """Helper to write a custom MIF file for testing."""
        path = tmp_path / filename
        header_lines = [
            "mrtrix image",
            f"dim: {','.join(str(d) for d in dim)}",
            "datatype: Int32LE",
            f"layout: {layout_str}",
        ]
        for row in transform_rows:
            header_lines.append(f"transform: {','.join(str(v) for v in row)}")
        
        header_lines.append("file: . 1024")
        header_lines.append("END")
        
        header_text = "\n".join(header_lines) + "\n"
        header_bytes = header_text.encode("utf-8")
        
        # Pad header to exactly 1024 bytes
        padding = b"\x00" * (1024 - len(header_bytes))
        
        # Serialize canonical_data to stored memory order
        layout_strs = layout_str.split(",")
        layout_vals = [int(s) for s in layout_strs]
        abs_layout = [abs(l) for l in layout_vals]
        
        stored_data = canonical_data.copy()
        for c in range(3):
            if layout_strs[c].strip().startswith("-"):
                stored_data = np.flip(stored_data, axis=c)
                
        c_slowest = abs_layout.index(2)
        c_middle = abs_layout.index(1)
        c_fastest = abs_layout.index(0)
        
        stored_data = np.transpose(stored_data, [c_slowest, c_middle, c_fastest])
        flat_bytes = stored_data.astype(np.int32).tobytes()
        
        with open(path, "wb") as f:
            f.write(header_bytes)
            f.write(padding)
            f.write(flat_bytes)
            
        return path

    @pytest.mark.parametrize(
        "layout_str,dim",
        [
            ("+0,+1,+2", (4, 5, 6)),
            ("+2,-0,-1", (4, 5, 6)),
            ("-1,+2,-0", (3, 4, 5)),
            ("-0,-1,-2", (4, 4, 4)),
        ],
    )
    def test_mif_fallback_parser(self, tmp_path, layout_str, dim):
        from unittest.mock import patch
        from app.io.nifti_loader import load_mif

        canonical_data = np.arange(np.prod(dim), dtype=np.int32).reshape(dim)
        transform_rows = [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
        ]
        expected_affine = np.array(transform_rows + [[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

        mif_path = self._write_mif(
            tmp_path, "test.mif", dim, layout_str, canonical_data, transform_rows
        )

        # Force fallback parser by patching subprocess.run to raise FileNotFoundError
        with patch("subprocess.run", side_effect=FileNotFoundError):
            parsed_data, parsed_affine = load_mif(mif_path)

        assert parsed_data.shape == dim
        np.testing.assert_array_equal(parsed_data, canonical_data)
        np.testing.assert_allclose(parsed_affine, expected_affine)
