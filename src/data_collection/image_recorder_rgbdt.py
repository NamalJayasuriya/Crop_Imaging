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

# examples/python/reconstruction_system/sensors/realsense_recorder.py

# pyrealsense2 is required.
# Please see instructions in https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
from os import makedirs
from os.path import exists, join, abspath
import shutil
import json
from enum import IntEnum
from flirpy.camera.lepton import Lepton

import sys
sys.path.append(abspath(__file__))
from src.utils.realsense_helper import get_profiles



CLIPPING_DISTANCE = 3
FPS = 15 #record from device
RES_X =  1280 #record from device
REX_Y = 720 #record from device
BAG_FILE_NAME_READ = '../../data/bag_files/R6_G4_L_21_9.bag' #R6_G4_l_16_300_D415_441PM  D415_single_plant2
SAVE_IMGS_ON_PLAYBACK = False # only for  playback rosbag
RECORDING = False # should be False for realtime recording
THERMAL = False
if not THERMAL and SAVE_IMGS_ON_PLAYBACK:
    RECORDING = True
if SAVE_IMGS_ON_PLAYBACK:
    DIRECTORY_OUT_DEFAULT =  '../../data/bag_files/R6_G4_L_21_9'
else:
    DIRECTORY_OUT_DEFAULT = '../../data/image_files/t'
DECIMATION = True
FILTERING = True
OFFSET = 2 # Offset foe saving images

try:
    # Python 2 compatible
    input = raw_input
except NameError:
    pass

class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def make_clean_folder(path_folder):
    if not exists(path_folder):
        makedirs(path_folder)
    else:
        user_input = input("%s not empty. Overwrite? (y/n) : " % path_folder)
        if user_input.lower() == 'y':
            shutil.rmtree(path_folder)
            makedirs(path_folder)
        else:
            exit()


def save_intrinsic_as_json(filename, frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    with open(filename, 'w') as outfile:
        obj = json.dump(
            {
                'width':
                    intrinsics.width,
                'height':
                    intrinsics.height,
                'intrinsic_matrix': [
                    intrinsics.fx, 0, 0, 0, intrinsics.fy, 0, intrinsics.ppx,
                    intrinsics.ppy, 1
                ]
            },
            outfile,
            indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
        "Realsense Recorder. Please select one of the optional arguments")
    parser.add_argument("--output_folder",
                        default=DIRECTORY_OUT_DEFAULT,
                        help="set output folder")
    parser.add_argument("--record_rosbag",
                        action='store_true',
                        help="Recording rgbd stream into realsense.bag")
    parser.add_argument("--record_imgs",
        action='store_true',
        help="Recording save color and depth images into realsense folder")
    parser.add_argument("--playback_rosbag",
                        action='store_true',
                        help="Play recorded realsense.bag file")
    args = parser.parse_args()

    if sum(o is not False for o in vars(args).values()) != 2:
        parser.print_help()
        exit()

    path_output = args.output_folder
    path_depth = join(args.output_folder, "depth")
    path_color = join(args.output_folder, "color")
    if THERMAL:
        path_thermal = join(args.output_folder, "thermal")
    if args.record_imgs or SAVE_IMGS_ON_PLAYBACK:
        make_clean_folder(path_output)
        make_clean_folder(path_depth)
        make_clean_folder(path_color)
        if THERMAL:
            make_clean_folder(path_thermal)

    path_bag = join(args.output_folder, 'realsense.bag')
    if args.record_rosbag:
        if exists(path_bag):
            user_input = input("%s exists. Overwrite? (y/n) : " % path_bag)
            if user_input.lower() == 'n':
                exit()
    if THERMAL:
        # Initialize leption thermal camera by Namal
        thermal_cam = Lepton()

    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    color_profiles, depth_profiles = get_profiles()

    if args.record_imgs or args.record_rosbag:
        # note: using 640 x 480 depth resolution produces smooth depth boundaries
        #       using rs.format.bgr8 for color image format for OpenCV based image visualization
        print('Using the default profiles: \n  color:{}, depth:{}'.format(
            color_profiles[0], depth_profiles[0]))
        w, h, fps, fmt = RES_X, REX_Y, FPS, rs.format.z16 #depth_profiles[0] #ToDo: hardcorded by namal
        config.enable_stream(rs.stream.depth, w, h, fmt, fps)
        w, h, fps, fmt = RES_X, REX_Y, FPS, rs.format.rgb8 #color_profiles[0] # #ToDo: hardcorded by namal
        config.enable_stream(rs.stream.color, w, h, fmt, fps)
        if args.record_rosbag:
            config.enable_record_to_file(path_bag)
    if args.playback_rosbag:
        config.enable_device_from_file(BAG_FILE_NAME_READ, repeat_playback=False)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    # Using preset HighAccuracy for recording
    if args.record_rosbag or args.record_imgs:
        depth_sensor.set_option(rs.option.visual_preset, Preset.Default) # ToDO: change High Accuracy -> Default

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
    clipping_distance_in_meters = CLIPPING_DISTANCE  # 3 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)


    # Streaming loop
    frame_count = 0
    fc2 = 0
    try:

        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            if THERMAL:
                thermal_img = thermal_cam.grab().astype(np.float32)

            #Decimation Added  by Namal
            if DECIMATION:
                frames = decimation.process(frames).as_frameset()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Added by Namal
            if FILTERING:
                #aligned_depth_frame = decimation.process(aligned_depth_frame)
                aligned_depth_frame = depth_to_disparity.process(aligned_depth_frame)
                aligned_depth_frame = spatial.process(aligned_depth_frame)
                aligned_depth_frame = temporal.process(aligned_depth_frame)
                aligned_depth_frame = disparity_to_depth.process(aligned_depth_frame)
                #aligned_depth_frame = hole_filling.process(aligned_depth_frame)

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())


            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            #depth image is 1 channel, color is 3 channels
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0.3), grey_color, color_image)

            #added by Namal
            depth_filtered = np.where((depth_image > clipping_distance) | (depth_image <= 0.3), 0, depth_image)

            if THERMAL:
                #thermal resize by namal
                #thermal_img = cv2.resize(thermal_img, (1280, 720))
                thermal_img2 = 255 * (thermal_img - thermal_img.min()) / (thermal_img.max() - thermal_img.min())
                thermal_img2 = cv2.resize(thermal_img2, (1280, 720))
            if (args.record_imgs or SAVE_IMGS_ON_PLAYBACK) and RECORDING:
                if fc2 == 0:
                    save_intrinsic_as_json(
                        join(args.output_folder, "camera_intrinsic.json"),
                        color_frame)
                if frame_count%OFFSET == 0:
                    cv2.imwrite("%s/%06d.png" % \
                            (path_depth, fc2), depth_image)
                    cv2.imwrite("%s/%06d.jpg" % \
                            (path_color, fc2), cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
                    if THERMAL:
                        cv2.imwrite("%s/%06d.png" % \
                                    (path_thermal, fc2), thermal_img2)
                        cv2.imwrite("%s/%06d_.png" % \
                                    (path_thermal, fc2), thermal_img)
                    print("Saved color + depth + thermal images %06d" % fc2)
                    fc2 += 1
                frame_count += 1

            # Render images
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)
            if THERMAL:
                thermal_colormap = cv2.applyColorMap(thermal_img2.astype(np.uint8), cv2.COLORMAP_INFERNO)
                #thermal_colormap = cv2.applyColorMap(cv2.convertScaleAbs(thermal_img, alpha=
            if THERMAL:
                images = np.hstack((color_image, depth_colormap, thermal_colormap)) #ToDo: change bg_removed->
            else:
                images = np.hstack((color_image, depth_colormap))

            if not SAVE_IMGS_ON_PLAYBACK:
                win_name ='Recorder Realsense'
                cv2.namedWindow(win_name, cv2.WINDOW_FULLSCREEN)
                # Move it to (X,Y)
                cv2.moveWindow(win_name, -400, 0)

                # Resize the Window
                images = cv2.resize(images, (1800, 380))

                # Show the Image in the Window
                cv2.imshow(win_name, images)

                #time.sleep(1)
                key = cv2.waitKey(1)

                # if 'esc' button pressed, escape loop and exit program
                if key == 27:
                    cv2.destroyAllWindows()
                    thermal_cam.close()
                    break
                elif key == ord('p'):
                    RECORDING = False
                elif key == ord('s'):
                    file = str(input("Enter A Directory Name: "))
                    #DIRECTORY_OUT_DEFAULT = '../dataset/'+file
                    path_output = '../../data/image_fies/'+file
                    path_depth = join(path_output, "depth")
                    path_color = join(path_output, "color")
                    make_clean_folder(path_output)
                    make_clean_folder(path_depth)
                    make_clean_folder(path_color)
                    if THERMAL:
                        path_thermal = join(path_output, "thermal")
                        make_clean_folder(path_thermal)
                    frame_count = 0
                    fc2 =0
                    RECORDING = True
                else:
                    continue
    finally:
        pipeline.stop()
