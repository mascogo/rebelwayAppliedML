import open3d as o3d
import numpy as np

point_cloud = o3d.io.read_point_cloud("point_cloud.ply")

point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)

bbox = point_cloud.get_axis_aligned_bounding_box()
mesh = mesh.crop(bbox)

o3d.io.write_triangle_mesh("mesh.obj", mesh)

# Visualize the mesh
o3d.visualization.draw_geometries([point_cloud], window_name="Point Cloud")
o3d.visualization.draw_geometries([mesh], window_name="Mesh from Point Cloud")

