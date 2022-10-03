import cv2 as cv
import numpy as np
from os import listdir, makedirs

PATH = "../../data/image_files/"
DATA_DIR = "TEST_1280/"
RES_X = 640
REX_Y = 360
ROTATE = False



def preprocess_files(dir_in, dir_out, type):
    dir_out = dir_out+"/"+type
    makedirs(dir_out,exist_ok=True)
    files = listdir(dir_in+type)
    print(dir_in+type)
    for f in files:
        img = cv.imread(dir_in+type+"/"+f)
        img = cv.resize(img, (RES_X, REX_Y))
        if ROTATE:
            img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
        cv.imwrite(dir_out+"/"+f.split('/')[-1], img)



out_dir = PATH+"TEST_"+str(RES_X)
preprocess_files(PATH+DATA_DIR, out_dir, 'color')
preprocess_files(PATH+DATA_DIR, out_dir, 'depth')

