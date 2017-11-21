"""

This module loads dental xrays and manipulates them.

You can

Load Dental Radiographs

Enahnce them

Resize them

Build Guassian Pyramids

Sobelize them

"""

import cv2
from scipy.ndimage import morphology

import numpy as np
import sys
import plotting_code

def load_images(specific=None,exclude=None):
    if specific:
        if specific < 15:
            specific_file = './Data/Radiographs/%02d.tif' % specific
        else:
            specific_file = './Data/Radiographs/extra/%02d.tif' % specific
        images = cv2.imread(specific_file)
    elif exclude:
        files = ['./Data/Radiographs/%02d.tif' % i for i in range(1, 15) if i != exclude]
        images = [cv2.imread(f) for f in files]
    else:
        files = ['./Data/Radiographs/%02d.tif' % i for i in range(1, 15)]
        images = [cv2.imread(f) for f in files]
    return images

def enhance(image):
    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bilateralFilter(image, 9, 175, 175)
    image_top = morphology.white_tophat(image, size=400)
    image_bottom = morphology.black_tophat(image, size=80)
    image = cv2.add(image, image_top)
    image = cv2.subtract(image, image_bottom)
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    image = clahe_obj.apply(image)
    return image

def resize(image, width, height):
    scale = min(float(width) / image.shape[1], float(height) / image.shape[0])
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale))), scale

def build_gauss_pyramid(image, levels):
    pyramids = []
    pyramids.append(image)
    temp = image
    for i in range(0, levels):
        temp = cv2.pyrDown(temp)
        pyramids.append(temp)
    return pyramids

def sobelize(image):
    image = cv2.GaussianBlur(image,(3,3),0)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


    print pixel_count, "pixel(s) filtered out of", xlength*ylength
    return image_array

