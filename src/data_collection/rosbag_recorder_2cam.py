import pyrealsense2 as rs
import time

res_x, res_y, color_format, depth_format, fps = 1280, 720, rs.format.rgb8, rs.format.z16, 30
file_path = "../../data/bag_files/record_"

context = rs.context()
devices = context.devices

print(len(devices))

dev1 = devices[0].get_info(rs.camera_info.serial_number)
dev2 = devices[1].get_info(rs.camera_info.serial_number)

print(dev1, dev2)

pipe1 = rs.pipeline()
pipe2 = rs.pipeline()

config1 = rs.config()
config2= rs.config()

config1.enable_stream(rs.stream.color, res_x, res_y, color_format, fps)
config1.enable_stream(rs.stream.depth, res_x, res_y, depth_format, fps)

config2.enable_stream(rs.stream.color, res_x, res_y, color_format, fps)
config2.enable_stream(rs.stream.depth, res_x, res_y, depth_format, fps)

config1.enable_record_to_file(file_path+str(dev1)+"12.bag")
config2.enable_record_to_file(file_path+str(dev2)+"12.bag")

config1.enable_device(dev1)
config2.enable_device(dev2)

profile1 = pipe1.start(config1)
profile2 = pipe2.start(config2)

sensors   =  [profile1.get_device().first_depth_sensor(), profile2.get_device().first_depth_sensor()]
# sensor1.set_option(rs.option.inter_cam_sync_mode, 0)
# sensor2  =  profile2.get_device().first_depth_sensor()
# sensor2.set_option(rs.option.inter_cam_sync_mode, 0)

for sensor in sensors:
    sensor.set_option(rs.option.inter_cam_sync_mode, 0)
    sensor.set_option(rs.option.visual_preset, 1) # 1: default

    # Set Controls : Added by NAmal
    sensor.set_option(rs.option.enable_auto_exposure, 1)
    sensor.set_option(rs.option.depth_units, 0.001)
    sensor.set_option(rs.option.emitter_enabled, 1)
    sensor.set_option(rs.option.laser_power, 300)

time.sleep(30)
