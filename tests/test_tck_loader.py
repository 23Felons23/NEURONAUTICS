"""Unit tests for app.io.tck_loader."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from app.core.tractography import TractographyBundle
from app.io.tck_loader import load_tck


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_synthetic_tck(path: Path) -> None:
    """Write a minimal valid .tck file with 3 synthetic streamlines."""
    rng = np.random.default_rng(0)
    streamlines = [
        rng.uniform(0, 100, (10, 3)).astype(np.float32)
        for _ in range(3)
    ]
    tractogram = nib.streamlines.Tractogram(
        streamlines,
        affine_to_rasmm=np.eye(4),
    )
    nib.streamlines.save(tractogram, str(path))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadTck:
    def test_load_tck_returns_bundle(self, tmp_path: Path):
        """load_tck() on a valid .tck returns a non-empty TractographyBundle."""
        tck_path = tmp_path / "test.tck"
        _write_synthetic_tck(tck_path)

        bundle = load_tck(tck_path)

        assert isinstance(bundle, TractographyBundle)
        assert bundle.n_streamlines > 0

    def test_load_tck_bad_path(self, tmp_path: Path):
        """load_tck() raises FileNotFoundError for a non-existent file."""
        bad_path = tmp_path / "does_not_exist.tck"

        with pytest.raises(FileNotFoundError):
            load_tck(bad_path)
