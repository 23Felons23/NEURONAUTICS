"""Connectome CSV loader.

Reads a symmetric adjacency matrix from a comma-separated file.
Rows and columns correspond to parcellation regions; cell values
are connection strengths (e.g. fibre count, FA-weighted density).

Format
------
Optional single header row of parcel label strings::

    region_A, region_B, region_C
    0.0,      1.2,      0.4
    1.2,      0.0,      3.1
    0.4,      3.1,      0.0

If the first row is entirely numeric the loader infers integer labels
``["0", "1", ..., "N-1"]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.connectome import ConnectomeMatrix


def load_csv(path: Path | str) -> ConnectomeMatrix:
    """Load a connectome adjacency matrix from a ``.csv`` file.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    ConnectomeMatrix
        Symmetric (N × N) float32 weight matrix with parcel labels.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be parsed as a square numeric matrix.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Connectome CSV not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            first_line = fh.readline().strip()
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc

    # Detect header: if any cell in the first row is non-numeric → labels row
    cells = [c.strip() for c in first_line.split(",")]
    has_header = False
    labels: list[str] = []
    try:
        [float(c) for c in cells if c]
    except ValueError:
        has_header = True
        labels = [c.strip('"').strip("'").strip() for c in cells]

    try:
        weights = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1 if has_header else 0,
            dtype=np.float32,
        )
    except Exception as exc:
        raise ValueError(f"Failed to parse {path} as a numeric matrix: {exc}") from exc

    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(
            f"Connectome matrix must be square, got shape {weights.shape} in {path}"
        )

    n = weights.shape[0]
    if not labels:
        labels = [str(i) for i in range(n)]

    if len(labels) != n:
        raise ValueError(
            f"Header has {len(labels)} labels but matrix has {n} rows in {path}"
        )

    return ConnectomeMatrix(weights=weights, labels=labels)
