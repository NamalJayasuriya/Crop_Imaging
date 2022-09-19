# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/sensors/realsense_pcd_visualizer.py

# pyrealsense2 is required.
# Please see instructions in https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
import pyrealsense2 as rs
import numpy as np
from enum import IntEnum

from datetime import datetime
import open3d as o3d

from os.path import abspath
import sys
sys.path.append(abspath(__file__))
from realsense_helper import get_profiles

FILTERING = True
LOAD_BAG = True
#file_path = '/media/namal/Data/Phd/Datasets/realsense/g3_l_d455_15fps_300lsr.bag'
# file_path = '/media/namal/Data/Phd/Datasets/realsense/g3_r_test_23aug_6fps_300lsr.bag'
# file_path = '/media/namal/Data/Phd/Datasets/realsense/g3_d415_L_24auf_test_300lsr_10am_15fps.bag'
# file_path = '/media/namal/Data/Phd/Datasets/realsense/g3_L_24auf_test_300lsr_9.24am_15fps.bag'
#
# file_path = '/media/namal/Data/Phd/Datasets/realsense/g3_d455_L_24auf_test_300lsr_10am_15fps.bag'
# file_path = '/media/namal/Data/Phd/Datasets/realsense/g3_d415_L_24auf_test_300lsr_10am_15fps.bag'
# file_path = '/media/namal/Data/Phd/Datasets/realsense/D415_single_plant2.bag'
# file_path = '/media/namal/Data/Phd/Datasets/realsense/D455_single_plant2.bag'

# file_path = '/media/namal/Data/Phd/Datasets/realsense/D415_single_plant2.bag'
# file_path = '/media/namal/Data/Phd/Datasets/realsense/G6_L3_L_D455_5fps_5.24pm.bag'
#file_path = '/media/namal/Data/Phd/Datasets/realsense/R6_G4_R_16_300_D415_443PM.bag'
# file_path = '/media/namal/Data/Phd/Datasets/realsense/.bag'
# file_path = '/media/namal/Data/Phd/Datasets/realsense/.bag'
# file_path = '/media/namal/Data/Phd/Datasets/realsense/.bag'
# file_path = '/media/namal/Data/Phd/Datasets/realsense/.bag'
# file_path = '/media/namal/Data/Phd/Datasets/realsense/.bag'
# file_path = '/media/namal/Data/Phd/Datasets/realsense/.bag'
# file_path = '/media/namal/Data/Phd/Datasets/realsense/.bag'
# file_path = '/media/namal/Data/Phd/Datasets/realsense/.bag'

# file_path = '/media/namal/Data/Phd/Codes/Open3D/examples/python/reconstruction_system/dataset/realsense_bag/G4_R_16_8_no_pp_1-40.bag'

file_path = "../../data/bag_files/R6_G4_L_14_9.bag"

class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5

res_x, res_y, fps = 1280, 720, 6
#res_x, res_y, fps = 848, 480, 5


def get_intrinsic_matrix(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = o3d.camera.PinholeCameraIntrinsic(res_x, res_y, intrinsics.fx,
                                            intrinsics.fy, intrinsics.ppx,
                                            intrinsics.ppy)
    return out


if __name__ == "__main__":

    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    if LOAD_BAG:
        config.enable_device_from_file(file_path)
    else:
        color_profiles, depth_profiles = get_profiles()
        print('Using the default profiles: \n  color:{}, depth:{}'.format(
            color_profiles[0], depth_profiles[0]))
        w, h, fps, fmt = res_x, res_y, fps, rs.format.z16  #depth_profiles[0]
        config.enable_stream(rs.stream.depth, w, h, fmt, fps)
        w, h, fps, fmt = res_x, res_y, fps, rs.format.rgb8  #color_profiles[0]
        config.enable_stream(rs.stream.color, w, h, fmt, fps)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    if not LOAD_BAG:
        # Using preset HighAccuracy for recording
        depth_sensor.set_option(rs.option.visual_preset, Preset.Default)

        # Set Controls : Added by NAmal
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        depth_sensor.set_option(rs.option.depth_units, 0.001)
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
        depth_sensor.set_option(rs.option.laser_power, 300)

    # Depth filtering:added by Namal
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 2)
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = depth_sensor.get_depth_scale()

    # We will not display the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 3  # 3 meter
    clipping_distance = clipping_distance_in_meters / depth_scale
    # print(depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    pcd = o3d.geometry.PointCloud()
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

    # Streaming loop
    frame_count = 0
    try:
        while True:
            #input_ = input()

            dt0 = datetime.now()

            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            rames = decimation.process(frames).as_frameset()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()

            #Depth filtering:added by Namal

            if FILTERING:
                #aligned_depth_frame = decimation.process(aligned_depth_frame)
                aligned_depth_frame = depth_to_disparity.process(aligned_depth_frame)
                aligned_depth_frame = spatial.process(aligned_depth_frame)
                aligned_depth_frame = temporal.process(aligned_depth_frame)
                aligned_depth_frame = disparity_to_depth.process(aligned_depth_frame)
                #aligned_depth_frame = hole_filling.process(aligned_depth_frame)


            color_frame = aligned_frames.get_color_frame()
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                get_intrinsic_matrix(aligned_depth_frame)) #(color_frame)) ToDo: changed

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = o3d.geometry.Image(np.array(aligned_depth_frame.get_data()))
            color_temp = np.asarray(color_frame.get_data())
            color_image = o3d.geometry.Image(color_temp)

            #print("Depth Shape",np.array(aligned_depth_frame.get_data()).shape)
            #print("Color Shape",color_temp.shape)

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image,
                depth_image,
                depth_scale=1.0 / depth_scale, # ToDo
                depth_trunc=clipping_distance_in_meters,
                convert_rgb_to_intensity=False)
            temp = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, intrinsic)
            temp.transform(flip_transform)
            pcd.points = temp.points
            pcd.colors = temp.colors

            if frame_count == 0:
                vis.add_geometry(pcd)

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            process_time = datetime.now() - dt0
            print("\rFPS: " + str(1 / process_time.total_seconds()), end='')
            frame_count += 1

    finally:
        pipeline.stop()
    vis.destroy_window()
