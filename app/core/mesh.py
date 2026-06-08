import numpy as np
from dataclasses import dataclass


@dataclass
class SurfaceMesh:
    """Domain object representing a 3D surface mesh."""
    vertices: np.ndarray  # shape (N, 3), float
    faces: np.ndarray     # shape (M, 3) or flat array of polygons, int
    labels: np.ndarray | None = None  # shape (N,), int
    
    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]
        
    @property
    def num_faces(self) -> int:
        # If faces is (M, 3)
        if self.faces.ndim == 2 and self.faces.shape[1] == 3:
            return self.faces.shape[0]
        # Otherwise it might be a VTK cell array format where each face is preceded by its vertex count
        # This is harder to count simply without knowing the format.
        return len(self.faces)
