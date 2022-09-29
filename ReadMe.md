# Crop Imaging System for Tall Vertically Supported Plants Under Protected Cropping

### Record RGB + depth + thermal Images
`python image_recorder_rgbdt.py --record_imgs`
`python image_recorder_rgbdt.py --playback_rosbag`

### Run Open3d Reconstruction system
`python run_system.py --config='config/realsense.json' --make --register --refine --integrate`
