import json
import os
from pathlib import Path
import pytest

from app.io.manifest import DatasetBundle, load_manifest


def test_load_manifest_valid(tmp_path):
    # Create a dummy manifest file
    manifest_data = {
        "t1": "mri/t1.nii.gz",
        "tumor": "lesion.nii.gz",
        "connectome": "connectome.csv"
    }
    manifest_file = tmp_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)
        
    # The expected absolute paths
    expected_t1 = str((tmp_path / "mri" / "t1.nii.gz").resolve())
    expected_tumor = str((tmp_path / "lesion.nii.gz").resolve())
    expected_connectome = str((tmp_path / "connectome.csv").resolve())
    
    bundle = load_manifest(manifest_file)
    
    assert bundle.t1 == expected_t1
    assert bundle.tumor == expected_tumor
    assert bundle.connectome == expected_connectome
    # Fields not in JSON should be None
    assert bundle.tck is None
    assert bundle.mesh is None
    assert bundle.parcellation_volume is None


def test_load_manifest_empty(tmp_path):
    manifest_file = tmp_path / "empty.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump({}, f)
        
    bundle = load_manifest(manifest_file)
    assert bundle.t1 is None
    assert bundle.tumor is None


def test_load_manifest_alternative_keys(tmp_path):
    manifest_data = {
        "parcellation": "parc.mif"
    }
    manifest_file = tmp_path / "manifest2.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)
        
    bundle = load_manifest(manifest_file)
    expected_parc = str((tmp_path / "parc.mif").resolve())
    assert bundle.parcellation_volume == expected_parc


def test_load_manifest_colors_and_lut(tmp_path):
    manifest_data = {
        "mesh_color": [255, 128, 0],
        "mesh_lut": {
            "1": [0.1, 0.2, 0.3],
            "2": [255, 0, 128]
        }
    }
    manifest_file = tmp_path / "manifest3.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)
        
    bundle = load_manifest(manifest_file)
    
    assert bundle.mesh_color == (1.0, 128 / 255.0, 0.0)
    assert bundle.mesh_lut["1"] == (0.1, 0.2, 0.3)
    assert bundle.mesh_lut["2"] == (1.0, 0.0, 128 / 255.0)


def test_load_manifest_tumor_properties(tmp_path):
    manifest_data = {
        "tumor_color": [255, 51, 51],
        "tumor_opacity": 0.8,
        "tumor_line_width": 3.5,
        "tumor_num_nodes": 1200
    }
    manifest_file = tmp_path / "manifest_tumor.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)
        
    bundle = load_manifest(manifest_file)
    
    assert bundle.tumor_color == (1.0, 0.2, 0.2)
    assert bundle.tumor_opacity == 0.8
    assert bundle.tumor_line_width == 3.5
    assert bundle.tumor_num_nodes == 1200

