import numpy as np
import open3d as o3d
import cv2
from matplotlib import pyplot as plt
import pyrealsense2 as rs
# from skimage.feature import canny
from skimage import data,morphology, segmentation
from skimage.color import rgb2gray, label2rgb
from skimage.filters import sobel
# from skimage.exposure import histogram
import scipy.ndimage as nd
import multiprocessing as mp
import time
from os import listdir
from src.utils.preprocessing_utils import *

file_path = "../../data/bag_files/leaves/l1.bag" #R6_G4_R_7_9  R6_G4_L_21_9

FILTERING = True
LOAD_BAG = True
DECIMATION = 4
HOLE_FILLING = True

plt.rcParams.update({'font.size': 8})

def imshow(imgs, titles=None, rotate=True):
    plt.figure()
    #subplot(r,c) provide the no. of rows and columns
    n= len(imgs)
    if n == 2:
        x, y = 7, 15
    else:
        x,y = 17, 35
    rows = int(np.ceil(n/4))
    f, axarr = plt.subplots(rows,n, figsize=(x, y))
    for i in range(n):
        if rotate:
            img = cv2.rotate(imgs[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            img = imgs[i]
        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        axarr[i].imshow(img)
        if not titles==None:
            axarr[i].set_title(titles[i])

def load_bag(file_path):
    pipeline = rs.pipeline()
    config = rs.config()
    if LOAD_BAG:
        config.enable_device_from_file(file_path)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    if DECIMATION:
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, DECIMATION)
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()


    depth_scale = depth_sensor.get_depth_scale()

    align_to = rs.stream.color
    align = rs.align(align_to)

    for i in range(50):
        frames = pipeline.wait_for_frames()

    #-------------- filtering ----------------

    if DECIMATION:
        frames = decimation.process(frames).as_frameset()

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame().as_video_frame()
    color_frame = aligned_frames.get_color_frame().as_video_frame()

    # Added by Namal
    if FILTERING:
        #aligned_depth_frame = decimation.process(aligned_depth_frame)
        aligned_depth_frame = depth_to_disparity.process(aligned_depth_frame)
        aligned_depth_frame = spatial.process(aligned_depth_frame)
        aligned_depth_frame = temporal.process(aligned_depth_frame)
        aligned_depth_frame = disparity_to_depth.process(aligned_depth_frame)
    if HOLE_FILLING:
        aligned_depth_frame = hole_filling.process(aligned_depth_frame)

    # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        print("Frame not found")

    depth_image1 = np.asanyarray(aligned_depth_frame.get_data())
    color_image1 = np.asanyarray(color_frame.get_data())

    pc = rs.pointcloud()
    pc.map_to(color_frame)

    points = pc.calculate(aligned_depth_frame)
    verts1 =  np.asanyarray(points.get_vertices())
    verts1 = verts1.view(np.float32).reshape(720, 1280, 3)
    # print(color_image1.shape)
    # print(verts1.shape)


    offset = 1
    res_x_mn, res_x_mx, res_y_mn, res_y_mx = 350, 700, 250, 600

    # color_image =color_image1[res_y_mn: res_y_mx, res_x_mn:res_x_mx]
    # depth_image =depth_image1[res_y_mn: res_y_mx, res_x_mn:res_x_mx]
    # verts =verts1[res_y_mn: res_y_mx, res_x_mn:res_x_mx]

    color_image =color_image1[res_y_mn:res_y_mx:offset, res_x_mn:res_x_mx:offset]
    depth_image =depth_image1[res_y_mn:res_y_mx:offset, res_x_mn:res_x_mx:offset]
    verts =verts1[res_y_mn:res_y_mx:offset, res_x_mn:res_x_mx:offset]

    print("shape c,d, v : {} {} {}".format(color_image.shape, depth_image.shape, verts.shape))

    # imshow([color_image1,depth_image1, color_image, depth_image], ['','','',''], False)
    intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

    return color_image, depth_image, verts, depth_scale, intrinsics


def preprocess():
    ## ---------- PRE PROCESSING ---------------- #
    GREEN_MASK = True
    ER_MASK_COLOR = False
    ER_MASK_DEPTH = False
    DEPTH_FILTERING = True
    D_MIN = 0.5
    D_MAX = 0.8

    # imshow([color_image, depth_image],["color", "depth"])

    MASK = False


    if DEPTH_FILTERING:
        # depth filtering
        depth_fed = depth_filter(depth_image, D_MIN, D_MAX, depth_scale)

    #     KERNAL_SIZE = 3
    #     kernel = np.ones((KERNAL_SIZE, KERNAL_SIZE), np.uint8)
    #     # Using cv2.erode() method
    #     depth_fe = cv2.erode(depth_f, kernel, cv2.BORDER_REFLECT, iterations=12)
    #     depth_fed = cv2.dilate(depth_fe, kernel, iterations=12)
    #     imshow([ depth_fe, depth_fed], ["depth eroted", "depth dialated"])

    # get green mask
    if GREEN_MASK:
        #depth filtered color
        color_d  = color_filter_by_mask(color_image, depth_fed )
        mask = green_color_mask(color_d, [20,50,20], [75,255,200])
        MASK = True

    # get region and edge based mask
    if ER_MASK_COLOR:
        #depth filtered color
        color_d  = color_filter_by_mask(color_image, depth_fed)
        mask = region_and_edge_based_segmentation_mask(color_d, True, ret='markers')
        MASK = True

    if ER_MASK_DEPTH:
        mask = region_and_edge_based_segmentation_mask(depth_image, ret='markers')
        MASK = True

    # mask = np.where(mask >0 and  mask2>0, 1, 0)
    print(mask[0][0])

    if MASK:
        depth_m = depth_filter_by_mask(depth_image, mask)
        color_m = color_filter_by_mask(color_image, mask)
        verts_m = color_filter_by_mask(verts, mask)

    # imshow([ depth_fed, mask], ['Depth Filtered','mask '])

    # imshow([ color_image, depth_image, depth_m, color_m,], ['color','depth', "depth masked", "color_masked"])

    # background remove color
    # color_m = bg_remove_color(depth_m, color_m)
    # imshow([color_m, get_colorized_depth(depth_m)], ["color bg_", "depth erode"])

    return color_m, depth_m, verts_m, mask

def o3d_potcloud(color, depth, intrinsics, vis=True):
    intrinsic = o3d.camera.PinholeCameraIntrinsic(1280, 720, intrinsics.fx,
                                                intrinsics.fy, intrinsics.ppx,
                                                intrinsics.ppy)
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    print(intrinsic)
    pcd = o3d.geometry.PointCloud()
    color = o3d.geometry.Image(color)
    depth = o3d.geometry.Image(depth)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=1000,
        depth_trunc=3,
        convert_rgb_to_intensity=False)

    temp = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    temp.transform(flip_transform)
    pcd.points = temp.points
    pcd.colors = temp.colors
    # pcd = pcd.voxel_down_sample(0.001)
    if vis:
        o3d.visualization.draw_geometries([pcd])
    return pcd

def o3d_area1(cloud, vis=True):
    # cloud = pcd #voxel_down_pcd # pcd2 voxel_down_pcd
    cloud.estimate_normals()
    distances = cloud.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 5 * avg_dist #
    # radii = [0.005, 0.01, 0.02, 0.04]
    # inlier_cloud.estimate_normals()
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, o3d.utility.DoubleVector([radius, radius * 2]))
    if vis:
        o3d.visualization.draw_geometries([bpa_mesh], mesh_show_back_face=False)
    print("area : {}".format(bpa_mesh.get_surface_area()))
    return bpa_mesh.get_surface_area()


if __name__ == "__main__":
    path = '../../data/bag_files/leaves/'
    files = listdir(path)
    for f in files:
        f='l1.bag'
        print(f)
        color_image, depth_image, verts, depth_scale, intrinsics = load_bag(path+f)
        color_m, depth_m, verts_m, mask = preprocess()
        # cv2imshow(mask, True)
        # show_pcd()
        # find_trs()

        pcd = o3d_potcloud(color_m, depth_m, intrinsics, True)
        if len(pcd.points)>0:
            area1 = o3d_area1(pcd, True)
        else:
            print(f+ "pcd error")
        break



