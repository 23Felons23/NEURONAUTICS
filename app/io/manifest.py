import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class DatasetBundle:
    """Holds absolute paths and settings loaded from a manifest file."""
    t1: Optional[str] = None
    tck: Optional[str] = None
    connectome: Optional[str] = None
    parcellation_volume: Optional[str] = None
    tumor: Optional[str] = None
    mesh: Optional[str] = None
    voxel_stride: int = 4
    voxel_point_size: float = 3.0
    voxel_black_value: Optional[float] = None
    voxel_white_value: Optional[float] = None
    voxel_spheres: bool = False
    mesh_color: Optional[tuple[float, float, float]] = None
    mesh_lut: Optional[dict[str, tuple[float, float, float]]] = None
    tumor_color: Optional[tuple[float, float, float]] = None
    tumor_opacity: Optional[float] = None
    tumor_line_width: Optional[float] = None
    tumor_num_nodes: Optional[int] = None


def load_manifest(path: Union[str, Path]) -> DatasetBundle:
    """
    Load a JSON manifest file and resolve any relative paths to absolute paths
    relative to the directory containing the manifest file.
    
    Args:
        path: Path to the JSON manifest file.
        
    Returns:
        DatasetBundle containing the absolute paths.
    """
    manifest_path = Path(path).resolve()
    base_dir = manifest_path.parent
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    bundle = DatasetBundle()
    
    # Map JSON keys to DatasetBundle fields
    path_mapping = {
        't1': 't1',
        'tck': 'tck',
        'connectome': 'connectome',
        'parcellation_volume': 'parcellation_volume',
        'tumor': 'tumor',
        'mesh': 'mesh',
        # Accept alternative keys
        'parcellation': 'parcellation_volume'
    }
    
    config_mapping = {
        'voxel_stride': ('voxel_stride', int),
        'voxel_point_size': ('voxel_point_size', float),
        'voxel_black_value': ('voxel_black_value', float),
        'voxel_white_value': ('voxel_white_value', float),
        'voxel_spheres': ('voxel_spheres', bool),
        'tumor_opacity': ('tumor_opacity', float),
        'tumor_line_width': ('tumor_line_width', float),
        'tumor_num_nodes': ('tumor_num_nodes', int)
    }
    
    for json_key, value in data.items():
        if value is None:
            continue
        if json_key in path_mapping:
            field = path_mapping[json_key]
            # Resolve relative paths against the manifest's parent directory
            abs_path = str((base_dir / value).resolve())
            setattr(bundle, field, abs_path)
        elif json_key in config_mapping:
            field, cast_fn = config_mapping[json_key]
            setattr(bundle, field, cast_fn(value))
        elif json_key == 'mesh_color':
            if isinstance(value, (list, tuple)) and len(value) == 3:
                vals = [float(v) for v in value]
                if any(v > 1.0 for v in vals):
                    vals = [v / 255.0 for v in vals]
                bundle.mesh_color = tuple(vals)
            else:
                raise ValueError("mesh_color must be a list/tuple of 3 RGB values")
        elif json_key == 'tumor_color':
            if isinstance(value, (list, tuple)) and len(value) == 3:
                vals = [float(v) for v in value]
                if any(v > 1.0 for v in vals):
                    vals = [v / 255.0 for v in vals]
                bundle.tumor_color = tuple(vals)
            else:
                raise ValueError("tumor_color must be a list/tuple of 3 RGB values")
        elif json_key == 'mesh_lut':
            if isinstance(value, dict):
                lut = {}
                for k, v in value.items():
                    if isinstance(v, (list, tuple)) and len(v) == 3:
                        vals = [float(val) for val in v]
                        if any(val > 1.0 for val in vals):
                            vals = [val / 255.0 for val in vals]
                        lut[str(k)] = tuple(vals)
                    else:
                        raise ValueError(f"mesh_lut color for key '{k}' must be a list/tuple of 3 RGB values")
                bundle.mesh_lut = lut
            else:
                raise ValueError("mesh_lut must be a dictionary mapping labels to 3-value RGB lists")
            
    return bundle

