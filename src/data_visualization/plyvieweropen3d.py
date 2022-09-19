import open3d as o3d

file_path = "../../data/bag_files/R6_G4_L_14_9/scene/integrated.ply"
print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud(file_path)
o3d.visualization.draw_geometries([pcd])