import numpy as np
import scipy.spatial
import pyvista as pv
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkLine, vtkCellArray
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor, vtkProperty

from app.io.nifti_loader import load_nifti
from app.core.volume import Volume
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class TumorOverlayRenderer:
    """Renders a binary lesion mask as a 3D spiderweb network.
    If a parcellation volume is provided, the web is fragmented into sub-actors for exploded viewing.
    """

    def __init__(self, actors: dict[str, vtkActor], centroids: dict[str, np.ndarray], global_centroid: np.ndarray):
        self.actors = actors
        self.centroids = centroids
        self.global_centroid = global_centroid
        # For backward compatibility
        self.actor = next(iter(actors.values())) if actors else vtkActor()

    @staticmethod
    def from_nifti(
        path: str,
        color: tuple[float, float, float] = (1.0, 0.2, 0.2),
        num_nodes: int = 1500,
        k_neighbors: int = 4,
        parc_vol: Volume = None,
        opacity: float = 0.5,
        line_width: float = 2.0
    ) -> "TumorOverlayRenderer":
        """
        Loads a binary NIfTI mask and creates a 3D spiderweb network.
        If parc_vol is provided, fragments the network by anatomical label.
        """
        vol = load_nifti(path)
        binary_mask = vol.data > 0
        
        actors_dict = {}
        centroids_dict = {}
        
        if not np.any(binary_mask):
            return TumorOverlayRenderer(actors_dict, centroids_dict, np.array([0,0,0]))
            
        indices = np.argwhere(binary_mask)
        
        if len(indices) > num_nodes:
            choices = np.random.choice(len(indices), size=num_nodes, replace=False)
            sampled_indices = indices[choices]
        else:
            sampled_indices = indices
            
        ones = np.ones((len(sampled_indices), 1))
        coords_4d = np.hstack((sampled_indices, ones))
        world_coords = coords_4d.dot(vol.affine.T)[:, :3]
        
        tree = scipy.spatial.KDTree(world_coords)
        distances, neighbors = tree.query(world_coords, k=15)
        
        points = vtkPoints()
        for pt in world_coords:
            points.InsertNextPoint(*pt)
            
        row_indices = []
        col_indices = []
        data_weights = []
        
        for i, (point_neighbors, point_dists) in enumerate(zip(neighbors, distances)):
            for j, dist in zip(point_neighbors[1:], point_dists[1:]):
                if j >= len(world_coords) or j < 0:
                    continue
                row_indices.append(i)
                col_indices.append(j)
                data_weights.append(dist)
                
        graph = csr_matrix((data_weights, (row_indices, col_indices)), shape=(len(world_coords), len(world_coords)))
        
        global_centroid = world_coords.mean(axis=0)
        root_idx = tree.query(global_centroid)[1]
        
        dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, indices=root_idx, return_predecessors=True)
        
        lines = vtkCellArray()
        for i, pred in enumerate(predecessors):
            if pred >= 0:
                line = vtkLine()
                line.GetPointIds().SetId(0, int(i))
                line.GetPointIds().SetId(1, int(pred))
                lines.InsertNextCell(line)
                    
        polydata = vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        
        if parc_vol is not None:
            # Fragment the web
            from app.io.parcellation_mapper import map_points_to_parcellation
            labels = map_points_to_parcellation(world_coords, parc_vol)
            
            pv_poly = pv.PolyData(polydata)
            pv_poly.point_data["Labels"] = labels
            pv_poly = pv_poly.point_data_to_cell_data(categorical=True)
            
            face_labels = pv_poly.cell_data["Labels"]
            unique_labels = np.unique(face_labels).astype(int)
            
            for label in unique_labels:
                sub_mesh = pv_poly.threshold([label, label], scalars="Labels", preference="cell")
                if sub_mesh.n_points == 0:
                    continue
                    
                sub_mesh_poly = sub_mesh.extract_surface(algorithm="dataset_surface")
                
                mapper = vtkPolyDataMapper()
                mapper.SetInputData(sub_mesh_poly)
                mapper.ScalarVisibilityOff()
                
                property = vtkProperty()
                property.SetColor(*color)
                property.SetOpacity(opacity)
                property.SetLineWidth(line_width)
                property.SetRenderLinesAsTubes(True) 
                
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.SetProperty(property)
                
                key = f"tumor_overlay_{label}"
                actors_dict[key] = actor
                centroids_dict[key] = np.array(sub_mesh_poly.center)
                
        else:
            # Single mesh
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            
            property = vtkProperty()
            property.SetColor(*color)
            property.SetOpacity(opacity)
            property.SetLineWidth(line_width)
            property.SetRenderLinesAsTubes(True) 
            
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.SetProperty(property)
            
            key = "tumor_overlay"
            actors_dict[key] = actor
            centroids_dict[key] = global_centroid
        
        return TumorOverlayRenderer(actors_dict, centroids_dict, global_centroid)
