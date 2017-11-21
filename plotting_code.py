
"""

Any code for plotting xrays and landmarks together goes here.

"""



import colorsys
import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from landmark_code import Landmark

def plot_landmarks(lms):

    if not isinstance(lms, list):
        lms = [lms]

    max_x, min_x, max_y, min_y = [], [], [], []
    for lm in lms:
        points = lm.points
        max_x.append(points[:, 0].max())
        min_x.append(points[:, 0].min())
        max_y.append(points[:, 1].max())
        min_y.append(points[:, 1].min())
    max_x, min_x, max_y, min_y = max(max_x), min(min_x), max(max_y), min(min_y)

    img = np.zeros((int((max_y - min_y) + 20), int((max_x - min_x) + 20)))

    for lm in lms:
        points = lm.points
        for i in range(len(points)):
            img[int(points[i, 1] - min_y) + 10, int(points[i, 0] - min_x) + 10] = 1

    cv2.imshow('Landmarks', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_landmarks_on_image(lms_list, img, show=True, save=False, wait=True, index=0,title='Landmarks'):

    img = img.copy()

    colors = __get_colors(len(lms_list))
    for ind, lms in enumerate(lms_list):
        points = lms.points
        for i in range(len(points)):
            cv2.line(img, (int(points[i, 0]), int(points[i, 1])),
                     (int(points[(i + 1)%40, 0]), int(points[(i + 1)%40, 1])),
                     colors[ind],2)

    if show:
        img = __fit_on_screen(img)
        cv2.imshow(title, img)
        if wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.waitKey(1)
            time.sleep(0.025)
    if save:
        cv2.imwrite('./Data/FinalOutput/predictions/'+title+'-%d.png' % index, img)


def plot_evaluations(lms_list, img, show=True, save=False, wait=True, index=0,title='Evaluation'):

    image = img.copy()

    colors = [ (81, 2, 159), (99,137,77)]

    for lms_pair in lms_list:
        for ind, lms in enumerate(lms_pair):
                points = np.array(lms.points)



                for i in range(len(points)):
                    cv2.line(image, (int(points[i, 0]), int(points[i, 1])),
                             (int(points[(i + 1)%40, 0]), int(points[(i + 1)%40, 1])),
                             colors[ind],2)




    if show:
        image = __fit_on_screen(image)
        cv2.imshow(title, image)
        if wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.waitKey(1)
            time.sleep(0.025)
    if save:
        cv2.imwrite('./Data/FinalOutput/evals/'+title+'-%d.png' % index, image)


def plot_image(img, save=False, title="Radiograph"):

    img = __fit_on_screen(img)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    if save:
        cv2.imwrite('Plot/'+title+'.png', img)
    cv2.destroyAllWindows()


def plot_procrustes(mean_shape, aligned_shapes, incisor_nr=0, save=False):

    # white background
    img = np.ones((1000, 600, 3), np.uint8) * 255

    # plot mean shape
    mean_shape = mean_shape.scale(1500).translate([300, 500])
    points = mean_shape.points
    for i in range(len(points)):
        cv2.line(img, (int(points[i, 0]), int(points[i, 1])),
                 (int(points[(i + 1) % 40, 0]), int(points[(i + 1) % 40, 1])),
                 (0, 0, 0), 2)
    ## center of mean shape
    cv2.circle(img, (300, 500), 10, (255, 255, 255))

    # plot aligned shapes
    colors = __get_colors(len(aligned_shapes))
    for ind, aligned_shape in enumerate(aligned_shapes):
        aligned_shape = aligned_shape.scale(1500).translate([300, 500])
        points = aligned_shape.points
        for i in range(len(points)):
            cv2.line(img, (int(points[i, 0]), int(points[i, 1])),
                     (int(points[(i + 1) % 40, 0]), int(points[(i + 1) % 40, 1])),
                     colors[ind])

    # show
    img = __fit_on_screen(img)
    cv2.imshow('Procrustes result for incisor ' + str(incisor_nr), img)
    cv2.waitKey(0)
    if save:
        cv2.imwrite('Plot/Procrustes/'+str(incisor_nr)+'.png', img)
    cv2.destroyAllWindows()


def plot_asm(asm, incisor_nr=0, save=False):

    __plot_mode(asm.mean_shape.as_vector(), asm.pc_modes[:, 0], title="PCA/incisor"+str(incisor_nr)+"mode1", save=save)
    __plot_mode(asm.mean_shape.as_vector(), asm.pc_modes[:, 1], title="PCA/incisor"+str(incisor_nr)+"mode2", save=save)
    __plot_mode(asm.mean_shape.as_vector(), asm.pc_modes[:, 2], title="PCA/incisor"+str(incisor_nr)+"mode3", save=save)
    __plot_mode(asm.mean_shape.as_vector(), asm.pc_modes[:, 3], title="PCA/incisor"+str(incisor_nr)+"mode4", save=save)


def __plot_mode(mu, pc, title="Active Shape Model", save=False):

    colors = [
        (147, 156, 253),
        (60, 76, 252),
        (27, 39, 176),
        (7, 11, 50),
        (27, 39, 176),
        (60, 76, 252),
        (147, 156, 253), ]

    shapes = [Landmark(mu-3*pc),
              Landmark(mu-2*pc),
              Landmark(mu-1*pc),
              Landmark(mu),
              Landmark(mu+1*pc),
              Landmark(mu+2*pc),
              Landmark(mu+3*pc)
             ]
    plot_shapes(shapes, colors, title, save)


def plot_shapes(shapes, colors, title="Shape Model", save=False):

    cv2.namedWindow(title)

    shapes = [shape.scale_to_unit().scale(1000) for shape in shapes]

    max_x = int(max([shape.points[:, 0].max() for shape in shapes]))
    max_y = int(max([shape.points[:, 1].max() for shape in shapes]))
    min_x = int(min([shape.points[:, 0].min() for shape in shapes]))
    min_y = int(min([shape.points[:, 1].min() for shape in shapes]))

    img = np.ones((max_y-min_y+20, max_x-min_x+20, 3), np.uint8)*255
    for shape_num, shape in enumerate(shapes):
        points = shape.points
        for i in range(len(points)):
            cv2.line(img, (int(points[i, 0]-min_x+10), int(points[i, 1]-min_y+10)),
                     (int(points[(i + 1) % 40, 0]-min_x+10), int(points[(i + 1) % 40, 1]-min_y+10)),
                     colors[shape_num], thickness=1)

    cv2.imshow(title, img)
    cv2.waitKey()
    if save:
        cv2.imwrite('Plot/'+title+'.png', img)
    cv2.destroyAllWindows()



def plot_profile(img, profile, save=False, title="profile"):

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    points = profile.points
    center = profile.model_point
    for i in range(len(points) - 1):
        cv2.line(img, (int(points[i, 0]), int(points[i, 1])),
                 (int(points[i + 1, 0]), int(points[i + 1, 1])),
                 (0, 255, 0))
    cv2.circle(img, (int(center[0]), int(center[1])), 2, (0, 255, 0))
    roi = img[int(center[1]-100):int(center[1]+100), int(center[0]-100):int(center[0]+100)]

    fig = plt.figure()
    fig.suptitle(title, fontsize=20)
    plt.subplot(1, 2, 1)
    plt.imshow(roi)
    plt.subplot(1, 2, 2)
    plt.plot(profile.samples)
    plt.ylabel('some numbers')
    plt.show()
    if save:
        fig.savefig('Plot/profile/'+title+'.png')


def plot_grey_level_model(model, imgs, save=False, title='Grey-level model'):

    fig = plt.figure()

    cols = 7
    rows = int(math.ceil(len(imgs) / cols)) + 1
    gs = gridspec.GridSpec(rows, cols)
    gs.update(wspace=0.025, hspace=0.05)
    ax1 = plt.subplot(gs[0, :4])
    ax2 = plt.subplot(gs[0, 5:])

    ax1.plot(model.mean_profile)
    ax2.pcolor(model.covariance)

    for ind, img in enumerate(imgs):
        points = model.profiles[ind].points
        center = model.profiles[ind].model_point
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.circle(img, (int(points[0, 0]), int(points[0, 1])), 2, (255, 0, 0))
        for i in range(len(points) - 1):
            cv2.line(img, (int(points[i, 0]), int(points[i, 1])),
                     (int(points[i + 1, 0]), int(points[i + 1, 1])),
                     (0, 255, 0))
        cv2.circle(img, (int(center[0]), int(center[1])), 2, (0, 255, 0))
        roi = img[int(center[1]-10):int(center[1]+10), int(center[0]-10):int(center[0]+10)]
        ax = fig.add_subplot(gs[ind+cols])
        ax.imshow(roi)
        ax.axis('off')

    plt.show()
    if save:
        fig.savefig('Plot/glm/'+title+'.png')


def plot_fits(img, profile, glm, fits, best, k, m):

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    points = profile.points
    center = profile.model_point

    maxfit = max(fits)


    cv2.circle(img, (int(points[0, 0]), int(points[0, 1])), 2, (255, 0, 0))
    cv2.circle(img, (int(points[-1, 0]), int(points[-1, 1])), 2, (255, 0, 0))
    for i in range(k, k+2*(m-k)):
        cv2.line(img, (int(points[i, 0]), int(points[i, 1])),
                 (int(points[i + 1, 0]), int(points[i + 1, 1])),
                 tuple(255*x for x in plt.cm.jet(fits[i-k]/maxfit)))
    cv2.circle(img, (int(best[0]), int(best[1])), 2, (0, 255, 0))
    cv2.circle(img, (int(center[0]), int(center[1])), 2, (0, 0, 255))
    roi = img[int(center[1]-100):int(center[1]+100), int(center[0]-100):int(center[0]+100)]

    plt.subplot(221)
    plt.imshow(roi)
    plt.subplot(222)
    plt.plot(fits)
    plt.title('Fit quality in candidate points')
    plt.subplot(212)
    plt.plot(glm.mean_profile)
    plt.title('Grey-level model')
    plt.show()

def plot_autoinit(img, window, quality, jaw_split, search_region=None,
                  best_score_bbox=None, title="autoinit", wait=False):

    img = img.copy()
    # convert search_region to OpenCV format

    # draw search_region on image
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if search_region:
        cv2.rectangle(img, search_region[1], search_region[0], (0, 206, 255), 3)

    # draw best score on image
    if best_score_bbox:
        cv2.rectangle(img, best_score_bbox[1], best_score_bbox[0], (0, 255, 177), 2)

    # draw current score on image
    cv2.rectangle(img, window[1], window[0], (255, 0, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img, "{0:.2f}".format(quality), window[0], font, 2, (0, 78, 255), 2)

    # draw jaw split on image
    __draw_path(img, jaw_split, color=(255, 177, 0))

    # show image
    img = __fit_on_screen(img)
    cv2.imshow(title, img)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.waitKey(1)
        time.sleep(0.025)


def plot_jaw_split(img, minimal_points, paths, best_path):

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    map(lambda x: __draw_path(img, x, color=(255, 177, 0)), paths)
    # map(lambda x: cv2.putText(img, str(int(path_intensity(radiograph, x)/(path_length(x)))),
                              # x[0], cv.CV_FONT_HERSHEY_PLAIN, 5, 255), paths)
    __draw_path(img, best_path, color=(206, 255, 0))

    for _, x, y in minimal_points:
        cv2.circle(img, (x, y), 1, (177, 0, 255), 12)
    img = __fit_on_screen(img)
    cv2.imshow('jaw split', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def __draw_path(radiograph, path, color=255):
    for i in range(0, len(path.edges)-1):
        cv2.line(radiograph, path.edges[i], path.edges[i+1], color, 5)

def __fit_on_screen(image):

    # find minimum scale to fit image on screen
    scale = min(float(1200) / image.shape[1], float(800) / image.shape[0])
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))


def __get_colors(num_colors):

    colors = []
    for i in np.arange(0., 200., 200. / num_colors):
        hue = i/360.
        lightness = (40 + np.random.rand() * 10)/100.
        saturation = (85 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return [(int(r*255), int(g*255), int(b*255)) for (r, g, b) in colors]
