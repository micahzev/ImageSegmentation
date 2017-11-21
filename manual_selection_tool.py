"""

lets you manually click and drag the ASM to the best location

"""

import cv2
import numpy as np
from landmark_code import Landmark

tooth = []
tmpTooth = []
dragging = False
start_point = (0, 0)


def manual_selection(landmarks, img):

    global tooth

    oimgh = img.shape[0]
    img, scale = resize(img, 1200, 800)
    imgh = img.shape[0]
    canvasimg = np.array(img)

    # transform model points to image coord
    points = landmarks.points
    min_x = abs(points[:, 0].min())
    min_y = abs(points[:, 1].min())
    points = [((point[0]+min_x)*scale, (point[1]+min_y)*scale) for point in points]
    tooth = points
    pimg = np.array([(int(p[0]*imgh), int(p[1]*imgh)) for p in points])
    cv2.polylines(img, [pimg], True, (255, 100, 80),2)

    # show gui
    cv2.imshow('MANUAL SELECT TOOTH', img)
    cv2.setMouseCallback('MANUAL SELECT TOOTH', mouse, canvasimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return Landmark(np.array([[point[0]*oimgh, point[1]*oimgh] for point in tooth]))


def mouse(ev, x, y, flags, img):

    global tooth
    global dragging
    global start_point

    if ev == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_point = (x, y)
    elif ev == cv2.EVENT_LBUTTONUP:
        tooth = tmpTooth
        dragging = False
    elif ev == cv2.EVENT_MOUSEMOVE:
        if dragging and tooth != []:
            move(x, y, img)


def move(x, y, img):

    global tmpTooth
    imgh = img.shape[0]
    tmp = np.array(img)
    dx = (x-start_point[0])/float(imgh)
    dy = (y-start_point[1])/float(imgh)

    points = [(p[0]+dx, p[1]+dy) for p in tooth]
    tmpTooth = points

    pimg = np.array([(int(p[0]*imgh), int(p[1]*imgh)) for p in points])
    cv2.polylines(tmp, [pimg], True, (0, 255, 0))
    cv2.imshow('MANUAL SELECT TOOTH', tmp)


def resize(image, width, height):

    scale = min(float(width) / image.shape[1], float(height) / image.shape[0])
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale))), scale