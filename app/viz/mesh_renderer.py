import numpy as np
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkProperty
import pyvista as pv

from app.core.mesh import SurfaceMesh


class MeshRenderer:
    """Renders a SurfaceMesh as one or more 3D solid objects.
    If the mesh has parcellation labels, it is split into sub-actors for exploded viewing.
    """

    def __init__(self, actors: dict[str, vtkActor], centroids: dict[str, np.ndarray], global_centroid: np.ndarray):
        self.actors = actors
        self.centroids = centroids
        self.global_centroid = global_centroid
        # For backward compatibility with tests/code expecting a single .actor
        self.actor = next(iter(actors.values())) if actors else vtkActor()

    @staticmethod
    def from_mesh(
        mesh: SurfaceMesh,
        color: tuple[float, float, float] = (0.95, 0.85, 0.75),
        opacity: float = 1.0,
        lut: dict[str, tuple[float, float, float]] | None = None,
    ) -> "MeshRenderer":
        """
        Creates VTK actors from a SurfaceMesh.
        
        Args:
            mesh: The SurfaceMesh domain object.
            color: Default RGB tuple for the mesh color if unparcellated.
            opacity: Opacity level from 0.0 (transparent) to 1.0 (solid).
            lut: Optional dictionary mapping label indices to custom RGB color tuples.
            
        Returns:
            MeshRenderer instance exposing the .actors dictionary.
        """
        polydata = pv.PolyData(mesh.vertices, mesh.faces)
        global_centroid = np.mean(mesh.vertices, axis=0)
        
        actors_dict = {}
        centroids_dict = {}
 
        if mesh.labels is not None:
            # We have parcellation labels, split the mesh
            # To prevent gaps between parcels, we map point data to cell data 
            # using PyVista's built-in filter with categorical=True to prevent averaging.
            polydata.point_data["Labels"] = mesh.labels
            polydata = polydata.point_data_to_cell_data(categorical=True)
            
            face_labels = polydata.cell_data["Labels"]
            unique_labels = np.unique(face_labels).astype(int)
            
            for label in unique_labels:
                # Extract sub-mesh using CELL thresholding to ensure clean partition
                sub_mesh = polydata.threshold([label, label], scalars="Labels", preference="cell")
                if sub_mesh.n_points == 0:
                     continue
                 
                # Extract surface to ensure it's PolyData not UnstructuredGrid
                sub_mesh_poly = sub_mesh.extract_surface(algorithm="dataset_surface")
                
                # Check for custom LUT color, otherwise generate procedurally
                patch_color = None
                if lut is not None:
                    patch_color = lut.get(str(label))
                
                if patch_color is None:
                    # Generate a consistent, distinct, and bright color per label
                    import colorsys
                    np.random.seed(int(label) * 12345)
                    h = np.random.random()
                    s = np.random.uniform(0.5, 1.0)
                    v = np.random.uniform(0.7, 1.0)
                    patch_color = colorsys.hsv_to_rgb(h, s, v)
                 
                mapper = vtkPolyDataMapper()
                mapper.SetInputData(sub_mesh_poly)
                # CRITICAL: Disable scalar visibility so VTK uses patch_color, not the Labels array
                mapper.ScalarVisibilityOff()
                
                prop = vtkProperty()
                prop.SetColor(patch_color)
                prop.SetOpacity(opacity)
                prop.SetInterpolationToPhong()
                prop.SetAmbient(0.1)
                prop.SetDiffuse(0.7)
                prop.SetSpecular(0.2)
                prop.SetSpecularPower(10.0)
                
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.SetProperty(prop)
                
                key = f"surface_mesh_{label}"
                actors_dict[key] = actor
                centroids_dict[key] = np.mean(sub_mesh_poly.points, axis=0)
                
        else:
            # No labels, render as a single mesh
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            
            prop = vtkProperty()
            prop.SetColor(color)
            prop.SetOpacity(opacity)
            prop.SetInterpolationToPhong()
            prop.SetAmbient(0.1)
            prop.SetDiffuse(0.7)
            prop.SetSpecular(0.2)
            prop.SetSpecularPower(10.0)
            
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.SetProperty(prop)
            
            key = "surface_mesh"
            actors_dict[key] = actor
            centroids_dict[key] = global_centroid
            
        return MeshRenderer(actors_dict, centroids_dict, global_centroid)

    def set_opacity(self, opacity: float) -> None:
        """Sets the opacity for all mesh actors."""
        for actor in self.actors.values():
            actor.GetProperty().SetOpacity(opacity)

    def explode(self, factor: float, max_dist: float = 150.0) -> None:
        """
        Translates sub-actors outwards from the global center to create an exploded view.
        
        Args:
            factor: Explode factor from 0.0 (assembled) to 1.0 (fully exploded).
            max_dist: Maximum translation distance in world units.
        """
        if len(self.actors) <= 1:
            return  # Nothing to explode if it's a single mesh
            
        for key, actor in self.actors.items():
            centroid = self.centroids.get(key)
            if centroid is None:
                continue
                
            # Vector from global center to actor centroid
            direction = centroid - self.global_centroid
            norm = np.linalg.norm(direction)
            
            if norm > 0.001:
                direction = direction / norm
            else:
                direction = np.array([0.0, 0.0, 0.0])
                
            # Calculate translation
            translation = direction * (factor * max_dist)
            actor.SetPosition(translation[0], translation[1], translation[2])
