import cv2
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
import pyrealsense2 as rs
from skimage.feature import canny
from skimage import data,morphology, segmentation
from skimage.color import rgb2gray, label2rgb
from skimage.filters import sobel
from skimage.exposure import histogram
import scipy.ndimage as nd
import copy
from src.utils.preprocessing_utils import *
from src.utils.pcd_utils import *

FILTERING = True
LOAD_BAG = True
DECIMATION = 4
HOLE_FILLING = True

def load_bag(file_path, offset=15, seq_n=50):
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

    heights = []

    for i in range(seq_n):
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

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # imshow([color_image1,depth_image1, color_image, depth_image], ['','','',''], False)
    intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

    return color_image, depth_image, depth_scale, intrinsics

def load_bag2(file_path, offset=15, seq_n=50, vis=False):
    pipeline = rs.pipeline()
    config = rs.config()
    if LOAD_BAG:
        config.enable_device_from_file(file_path,repeat_playback=False)

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

    heights = []

    frame_count = 0
    for i in range(15):
        frames = pipeline.wait_for_frames()
    while True:
        frames = pipeline.wait_for_frames()
        frame_count+=1

        if frame_count%offset == 0:

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

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # imshow([color_image1,depth_image1, color_image, depth_image], ['','','',''], False)
            intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

            # return color_image, depth_image, depth_scale, intrinsics
            color_m, depth_m, mask = preprocess(depth_image, color_image, depth_scale, vis=False) #set the configurations within the function

            pcd = o3d_potcloud(color_m, depth_m, intrinsics, max_d=2.2, v_size=0.01, rotation=[0,90,36], vis=True)
            pcd = radius_outlier_remove(pcd, 40, 0.08, vis=True)
            pcd = dbscan_outlier_remove(pcd, 80, 0.05, vis)
            height = crop_and_measure_height(pcd, -1.2, -2.25, 0.25, vis) *1000# dc, hc, lc
            print("seq_n: {}   height: {}".format(frame_count/offset, height))
            heights.append(height)
    return heights

def preprocess(depth_image, color_image, depth_scale, vis=False):
    ## ---------- PRE PROCESSING ---------------- #
    GREEN_MASK = False
    ER_MASK_COLOR = False
    ER_MASK_DEPTH = False
    DEPTH_FILTERING = True
    D_MIN =0.5
    D_MAX = 2.2

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

    if MASK:
        depth_m = depth_filter_by_mask(depth_image, mask)
        color_m = color_filter_by_mask(color_image, mask)
    else:
        depth_m = depth_fed
        color_m = color_filter_by_mask(color_image, depth_fed)
        mask = np.where(depth_fed>0 ,1 ,0)
    # imshow([ depth_fed, mask], ['Depth Filtered','mask '])

    if vis:
        imshow([ color_image, depth_image, depth_m, color_m,], ['color','depth', "depth masked", "color_masked"])
        plt.show()
    # background remove color
    # color_m = bg_remove_color(depth_m, color_m)
    # imshow([color_m, get_colorized_depth(depth_m)], ["color bg_", "depth erode"])

    return color_m, depth_m, mask

def radius_outlier_remove(pcd, m_points, radius, vis=False):
    cl, ind = pcd.remove_radius_outlier(nb_points=m_points, radius=radius)
    # display_inlier_outlier(pcd_vd1, ind)
    if vis:
        display_inlier_outlier(pcd, ind)
    return pcd.select_by_index(ind)

def dbscan_outlier_remove(pcd, m_points, eps, vis=False):
    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=m_points, print_progress=False)) # 0.05 ,80

    mask = np.where((labels > -1), 1, 0)
    # mask = np.where((labels < 0) | (labels >= labels.max()-5), 0, 1)
    m=list(np.nonzero(mask)[0])
    pcd2 = pcd.select_by_index(m)

    max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")

    if vis:
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        pcd.translate((2, 0, 0))
        o3d.visualization.draw_geometries([pcd, pcd2])
    return pcd2

def crop_and_measure_height(pcd, dc1, hc1, lc, vis):
    # dc1, hc1, lc = -2.0, -2.25, 0.25
    d=np.asarray(pcd.points).T[2]
    l=np.asarray(pcd.points).T[0]
    h=np.asarray(pcd.points).T[1]

    ind_d = np.where((d > dc1), 1, 0) #filter by depth(gutter width)
    ind_l = np.where(( (l > l.max()-lc) | (l < l.min()+lc)), 0, 1) # eliminate begining and end of frame length
    ind_h = np.where((h > hc1), 1, 0) #eliminate base area

    ind =  ind_d & ind_h  & ind_l

    ind_d=list(np.nonzero(ind)[0])

    pcd = pcd.select_by_index(ind_d)

    max_p=pcd.get_max_bound()
    min_p=pcd.get_min_bound()
    height = max_p[1]-min_p[1]

    if vis:
        aabb = pcd.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)
        # obb = mesh_r2.get_oriented_bounding_box()
        # obb.color = (0, 1, 0)
        o3d.visualization.draw_geometries([pcd, aabb])
    return height

if __name__ == "__main__":
    file_path = "../../data/bag_files/16_11_2022/R6_G3_L_16_11_35d.bag" #R6_G4_R_7_9  R6_G4_L_21_9
    color_image, depth_image, depth_scale, intrinsics = load_bag(file_path, offset=10, seq_n=15)

    color_m, depth_m, mask = preprocess(depth_image, color_image, depth_scale, vis=True) #set the configurations within the function

    pcd = o3d_potcloud(color_m, depth_m, intrinsics, max_d=2.2, v_size=0.01, rotation=[0,90,36], vis=True)
    pcd = radius_outlier_remove(pcd, 40, 0.08, vis=True)
    pcd = dbscan_outlier_remove(pcd, 80, 0.05, True)
    height = crop_and_measure_height(pcd, -2.0, -2.25, 0.25, True)

    # heights = load_bag2(file_path, offset=25, vis=True)
    # print(heights)

#%%
