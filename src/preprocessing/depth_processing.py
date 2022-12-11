import numpy as np
import cv2
from skimage.feature import canny
from skimage import data,morphology, segmentation
from skimage.color import rgb2gray, label2rgb
from skimage.filters import sobel
from skimage.exposure import histogram
import scipy.ndimage as nd

def depth_filter( depth_image, depth_scale, min_d, max_d):
    max_d = max_d / depth_scale
    min_d = min_d /depth_scale
    return np.where((depth_image > max_d) | (depth_image <= min_d), 0, depth_image)

def bg_remove_color(depth_image, color_image):
    grey_color = 0
    #depth image is 1 channel, color is 3 channels
    depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
    return np.where(depth_image_3d, color_image, grey_color)

def green_color_mask(color):
    #cv2.COLOR_RGB2BGR
    #r, g, b = cv2.split(color)

    #convert the BGR image to HSV colour space
    hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)

    #set the lower and upper bounds for the green hue
    lower_green = np.array([5,35,0])  #[50,100,50]
    upper_green = np.array([75,255,200])  #[70,255,255]

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


def region_and_edge_based_segmentation_mask(color_image, negation=False, m1=10, m2=150):
    img_wh = rgb2gray(color_image)

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

    mask =  markers #segment markers

    if not negation:
        return np.where(mask, 1, 0)
    else:
        return np.where(mask, 0, 1)
