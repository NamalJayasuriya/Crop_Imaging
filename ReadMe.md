# Crop Imaging System for Tall Vertically Supported Plants Under Protected Cropping

### Record RGB + depth + thermal Images
`python realsense_recorder2.py --record_imgs`

### Run Open3d Reconstruction system
`python run_system.py --config='config/realsense.json' --make --register --refine --integrate`