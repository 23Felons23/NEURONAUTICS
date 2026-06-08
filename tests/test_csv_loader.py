"""Unit tests for load_csv()."""

from __future__ import annotations

import numpy as np
import pytest

from app.io.csv_loader import load_csv


class TestLoadCsv:
    def test_load_square_matrix_no_header(self, tmp_path):
        csv = tmp_path / "conn.csv"
        csv.write_text("0.0,1.2,0.4\n1.2,0.0,3.1\n0.4,3.1,0.0\n")
        cm = load_csv(csv)
        assert cm.node_count == 3
        assert cm.weights.shape == (3, 3)
        assert cm.labels == ["0", "1", "2"]

    def test_load_with_header_labels(self, tmp_path):
        csv = tmp_path / "conn.csv"
        csv.write_text("A,B,C\n0.0,1.0,0.5\n1.0,0.0,2.0\n0.5,2.0,0.0\n")
        cm = load_csv(csv)
        assert cm.labels == ["A", "B", "C"]
        assert cm.node_count == 3

    def test_load_values_correct(self, tmp_path):
        csv = tmp_path / "conn.csv"
        csv.write_text("0.0,5.0\n5.0,0.0\n")
        cm = load_csv(csv)
        assert cm.weights[0, 1] == pytest.approx(5.0)
        assert cm.weights[1, 0] == pytest.approx(5.0)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path / "no_such_file.csv")

    def test_non_square_raises_value_error(self, tmp_path):
        csv = tmp_path / "bad.csv"
        # 3 rows × 2 cols — not square
        csv.write_text("1.0,2.0\n3.0,4.0\n5.0,6.0\n")
        with pytest.raises(ValueError):
            load_csv(csv)
