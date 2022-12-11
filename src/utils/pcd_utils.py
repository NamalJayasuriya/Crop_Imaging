import open3d as o3d
import numpy as np

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    # print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def degree(x):
    return np.pi*(x)/180

def o3d_potcloud(color, depth, intrinsics, max_d=4, v_size=0.001, rotation=[0,0,0], crop=[], vis=True):
    intrinsic = o3d.camera.PinholeCameraIntrinsic(1280, 720, intrinsics.fx,
                                                intrinsics.fy, intrinsics.ppx,
                                                intrinsics.ppy)
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    # print(intrinsic)
    pcd = o3d.geometry.PointCloud()
    color = o3d.geometry.Image(color)
    depth = o3d.geometry.Image(depth)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=1000,
        depth_trunc=max_d,
        convert_rgb_to_intensity=False)

    temp = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    temp.transform(flip_transform)
    pcd.points = temp.points
    pcd.colors = temp.colors
    pcd = pcd.voxel_down_sample(v_size)
    R = pcd.get_rotation_matrix_from_xzy((degree(rotation[0]), degree(rotation[1]), degree(rotation[2])))
    pcd.rotate(R, center=(0, 0, 0))
    if vis:
        o3d.visualization.draw_geometries([pcd])
    return pcd

