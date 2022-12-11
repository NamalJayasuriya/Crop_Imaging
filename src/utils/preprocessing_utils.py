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

def depth_filter(depth_image, min_d, max_d, depth_scale):
    max_d = max_d / depth_scale
    min_d = min_d /depth_scale
    return np.where((depth_image > max_d) | (depth_image <= min_d), 0, depth_image)

def bg_remove_color(depth_image, color_image):
    grey_color = 0
    #depth image is 1 channel, color is 3 channels
    depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
    return np.where(depth_image_3d, color_image, grey_color)

def green_color_mask(color, lower=[5,35,0], upper=[75,255,200]):

    #convert the BGR image to HSV colour space
    hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)

    #set the lower and upper bounds for the green hue
    lower_green = np.array(lower)  #[50,100,50]
    upper_green = np.array(upper)  #[70,255,255]

    #create a mask for green colour using inRange function
    mask = cv2.inRange(hsv, lower_green, upper_green)
    g_filtered = cv2.bitwise_and(color, color, mask=mask)
    mask = cv2.cvtColor(g_filtered, cv2.COLOR_RGB2GRAY)

    return mask

def depth_filter_by_mask(depth, mask):
    return np.where(mask, depth, 0)

def color_filter_by_mask(color, mask):
    mask_3d = np.dstack((mask, mask, mask))
    return np.where(mask_3d, color, 0)


def region_and_edge_based_segmentation_mask(image, negation=False, m1=10, m2=150, ret='segment'):
    if type(image[0][0]) == np.uint16:
        img_wh = image
    else:
        img_wh = rgb2gray(image)

#     # apply edge segmentation
#     edges = canny(img_wh)

#     # fill regions to perform edge segmentation
#     fill_im = nd.binary_fill_holes(edges)
#     # imshow([edges,fill_im])
#     # plt.title('edges, Region Filling')

    # Region Segmentation
    elevation_map = sobel(img_wh)
#     plt.imshow(elevation_map)

    # Since, the contrast difference is not much. Anyways we will perform it
    markers = np.zeros_like(img_wh)
    m1 = m1/255
    m2 = m2/255
    markers[img_wh < m1] = 1 # 30/255
    markers[img_wh > m2] = 2 # 150/255
    # plt.imshow(markers)
    # plt.title('markers')

    # Perform watershed region segmentation
    segment1 = segmentation.watershed(elevation_map, markers)

    segment = nd.binary_fill_holes(segment1-1) #segment-1

    # imshow([elevation_map, segment1], ["after sobel","segmentation watershed"])

    if ret=='markers':
        mask =  markers #segment markers
    elif ret=='segment':
        mask = segment

    if not negation:
        return np.where(mask, 1, 0)
    else:
        return np.where(mask, 0, 1)

def get_colorized_depth(depth):
    return cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.09), cv2.COLORMAP_JET)

def cv2imshow(image, depth=False):
    if depth:
        image = get_colorized_depth(image)
    cv2.imshow('',image)
    while True:
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
