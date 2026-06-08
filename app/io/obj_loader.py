import pyvista as pv
from app.core.mesh import SurfaceMesh


def load_obj(path: str) -> SurfaceMesh:
    """
    Loads a .obj file from the given path into a SurfaceMesh domain object.
    
    Args:
        path: Absolute or relative path to the .obj file.
        
    Returns:
        SurfaceMesh containing vertices and faces.
    """
    # PyVista provides a very robust way to load OBJ files
    mesh = pv.read(path)
    
    # mesh.points contains the vertices as a numpy array
    # mesh.faces contains the face definitions in VTK format (N, v1, v2, v3, ...)
    # where N is the number of vertices in the face (e.g. 3 for triangles)
    
    return SurfaceMesh(
        vertices=mesh.points,
        faces=mesh.faces
    )
