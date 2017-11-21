"""

Implements automatic finding of the tooth location


"""

import math
import plotting_code
import cv2
import numpy as np
import xray_code as xray
from split_radiograph import split
from time import sleep
from grey_level_model import Profile

bbox = None
rect_endpoint_tmp = []
rect_bbox = []
drawing = False

def create_database(radiographs):
    def draw_rect_roi(event, x, y, flags, param):
        global rect_bbox, rect_endpoint_tmp, drawing, bbox
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_endpoint_tmp = []
            rect_bbox = [(x, y)]
            drawing = True

        elif event == cv2.EVENT_LBUTTONUP:
            rect_bbox.append((x, y))
            drawing = False

            p_1, p_2 = rect_bbox
            p_1x, p_1y = p_1
            p_2x, p_2y = p_2

            lx = min(p_1x, p_2x)
            ty = min(p_1y, p_2y)
            rx = max(p_1x, p_2x)
            by = max(p_1y, p_2y)

            if (lx, ty) != (rx, by):
                bbox = [(lx, ty), (rx, by)]

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect_endpoint_tmp = [(x, y)]

    for is_lower in range(0, 2):
        if is_lower:
            print 'Select the region of the four lower incisors for each radiograph\n', \
                    'and press the a key when done or the q key to ignore the example.'
        else:
            print 'Select the region of the four upper incisors for each radiograph\n', \
                    'and press the a key when done or the q key to ignore the example.'

        bbox_list = []
        for ind, img in enumerate(radiographs):
            if is_lower:
                windowtitle = "Lower incisors [%d/%d]" % (ind+1, len(radiographs),)
            else:
                windowtitle = "Upper incisors [%d/%d]" % (ind+1, len(radiographs),)

            canvasimg = img.copy()

            canvasimg, scale = xray.resize(canvasimg, 1200, 800)
            cv2.namedWindow(windowtitle)
            cv2.setMouseCallback(windowtitle, draw_rect_roi)

            while True:
                rect_cpy = canvasimg.copy()
                if not drawing:
                    if bbox:
                        start_point = bbox[0]
                        end_point_tmp = bbox[1]
                        cv2.rectangle(rect_cpy, start_point, end_point_tmp, (0, 255, 0), 1)
                        cv2.imshow(windowtitle, rect_cpy)
                    else:
                        cv2.imshow(windowtitle, canvasimg)
                elif drawing and rect_endpoint_tmp:
                    start_point = rect_bbox[0]
                    end_point_tmp = rect_endpoint_tmp[0]
                    cv2.rectangle(rect_cpy, start_point, end_point_tmp, (0, 255, 0), 1)
                    cv2.imshow(windowtitle, rect_cpy)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('a'):
                    bbox_list.append(bbox)
                    break

                if key == ord('q'):
                    break

            cv2.destroyAllWindows()

        bbox_list = np.array([[(int(p[0]/scale), int(p[1]/scale))
                               for p in bb]
                              for bb in bbox_list])

        # print results summary
        bbs = [bb[1] - bb[0] for bb in bbox_list]
        avg_width, avg_height = np.mean(bbs, axis=0)
        print 'Avg. height: ' + str(avg_height)
        print 'Avg. width: ' + str(avg_width)

        if is_lower:
            np.save('./Data/EigenIncisorSets/uppers', bbox_list)
        else:
            np.save('./Data/EigenIncisorSets/lowers.npy', bbox_list)


def load_database(radiographs, is_upper, rewidth=500, reheight=500):

    smallImages = np.zeros((14, rewidth * reheight))
    try:
        if is_upper:
            four_incisor_bbox = np.load('./Data/EigenIncisorSets/uppers.npy')
        else:
            four_incisor_bbox = np.load('./Data/EigenIncisorSets/lowers.npy')
    except IOError:
        create_database(radiographs)
        sleep(5)
        if is_upper:
            four_incisor_bbox = np.load('./Data/EigenIncisorSets/uppers.npy')
        else:
            four_incisor_bbox = np.load('./Data/EigenIncisorSets/lowers.npy')

    radiographs = [xray.enhance(radiograph) for radiograph in radiographs]
    for ind, radiograph in enumerate(radiographs):
        [(x1, y1), (x2, y2)] = four_incisor_bbox[ind-1]
        cutImage = radiograph[y1:y2, x1:x2]
        result = cv2.resize(cutImage, (rewidth, reheight), interpolation=cv2.INTER_NEAREST)
        smallImages[ind-1] = result.flatten()

    return smallImages


def project(W, X, mu):
    return np.dot(X - mu.T, W)


def reconstruct(W, Y, mu):
    return np.dot(Y, W.T) + mu.T


def pca(X, nb_components=0):
    n = X.shape[0]
    if (nb_components <= 0) or (nb_components > n):
        nb_components = n

    mu = np.average(X, axis=0)
    X -= mu.transpose()

    eigenvalues, eigenvectors = np.linalg.eig(np.dot(X, np.transpose(X)))
    eigenvectors = np.dot(np.transpose(X), eigenvectors)

    eig = zip(eigenvalues, np.transpose(eigenvectors))
    eig = map(lambda x: (x[0] * np.linalg.norm(x[1]),
                         x[1] / np.linalg.norm(x[1])), eig)

    eig = sorted(eig, reverse=True, key=lambda x: abs(x[0]))
    eig = eig[:nb_components]

    eigenvalues, eigenvectors = map(np.array, zip(*eig))

    return eigenvalues, np.transpose(eigenvectors), mu


def normalize(img):
    return (img*(255./(np.max(img)-np.min(img)))+np.min(img)).astype(np.uint8)


def find_bbox(mean, evecs, image, width, height, is_upper, jaw_split, show=False):
    h, w = image.shape

    if is_upper:
        b1 = int(w/2 - w/15)
        b2 = int(w/2 + w/15)
        a1 = int(np.mean(jaw_split.get_part(b1, b2), axis=0)[1]) - 350
        a2 = int(np.mean(jaw_split.get_part(b1, b2), axis=0)[1])
    else:
        b1 = int(w/2 - w/20)
        b2 = int(w/2 + w/20)
        a1 = int(np.min(jaw_split.get_part(b1, b2), axis=0)[1])
        a2 = int(np.min(jaw_split.get_part(b1, b2), axis=0)[1]) + 350

    search_region = [(b1, a1), (b2, a2)]

    best_score = float("inf")
    best_score_bbox = [(-1, -1), (-1, -1)]
    best_score_img = np.zeros((500, 400))
    for wscale in np.arange(0.8, 1.3, 0.1):
        for hscale in np.arange(0.7, 1.3, 0.1):
            winW = int(width * wscale)
            winH = int(height * hscale)
            for (x, y, window) in sliding_window(image, search_region, step_size=36, window_size=(winW, winH)):

                if window.shape[0] != winH or window.shape[1] != winW:
                    continue

                reCut = cv2.resize(window, (width, height))

                X = reCut.flatten()
                Y = project(evecs, X, mean)
                Xacc = reconstruct(evecs, Y, mean)

                score = np.linalg.norm(Xacc - X)
                if score < best_score:
                    best_score = score
                    best_score_bbox = [(x, y), (x + winW, y + winH)]
                    best_score_img = reCut

                if show:
                    window = [(x, y), (x + winW, y + winH)]
                    plotting_code.plot_autoinit(image, window, score, jaw_split, search_region, best_score_bbox,
                                          title="wscale="+str(wscale)+" hscale="+str(hscale))

    return (best_score_bbox, best_score_img)

def sliding_window(image, search_region, step_size, window_size):

    for y in range(search_region[0][1], search_region[1][1] - window_size[1], step_size) + \
                [search_region[1][1] - window_size[1]]:
        for x in range(search_region[0][0], search_region[1][0] - window_size[0], step_size) + \
                [search_region[1][0] - window_size[0]]:
            # yield the current window
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])



def fit_template(template, model, img):

    gimg = xray.sobelize(img)

    dmin, best = np.inf, None
    for t_x in xrange(-5, 50, 10):
        for t_y in xrange(-50, 50, 10):
            for s in np.arange(0.8, 1.2, 0.1):
                for theta in np.arange(-math.pi/16, math.pi/16, math.pi/16):
                    dists = []
                    X = template.T([t_x, t_y], s, theta)
                    for ind in list(range(15)) + list(range(25,40)):
                        profile = Profile(img, gimg, X, ind, model.k)
                        dist = model.glms[0][ind].quality_of_fit(profile.samples)
                        dists.append(dist)
                    avg_dist = np.mean(np.array(dists))
                    if avg_dist < dmin:
                        dmin = avg_dist
                        best = X

                    # plotting_code.plot_landmarks_on_image([template, best, X], img, wait=False)

    return best


def automatic_selection(model, img, incisor, show=False):

    tooth = incisor
    is_upper = tooth < 5
    if is_upper:
        width = 397
        height = 365
    else:

        width = 307
        height = 333

    radiographs = xray.load_images()

    eigen_incisor_set = load_database(radiographs, is_upper, width, height)

    [_, evecs, mean] = pca(eigen_incisor_set, 5)

    # Visualize the appearance model
    # cv2.imshow('img',np.hstack( (mean.reshape(1120,2116),
    #                              normalize(evecs[:,0].reshape(1120,2116)),
    #                              normalize(evecs[:,1].reshape(1120,2116)),
    #                              normalize(evecs[:,2].reshape(1120,2116)))
    #                            ).astype(np.uint8))
    # cv2.waitKey(0)

    # Find the jaw split
    divided_image = split(img, show=False)

    img = xray.enhance(img)
    [(a, b), (c, d)], _ = find_bbox(mean, evecs, img, width, height, is_upper, divided_image, show=show)

    ind = tooth if tooth < 5 else tooth - 4
    bbox = [(a +(ind-1)*(c-a)/4, b), (a +(ind)*(c-a)/4, d)]
    center = np.mean(bbox, axis=0)

    if show:
        plotting_code.plot_autoinit(img, bbox, 0, divided_image, wait=True)

    template = model.mean_shape.capture_bounding_box(bbox).translate(center)

    if is_upper:
        X = template
    else:
        X = template

    if show:
        plotting_code.plot_landmarks_on_image([X], img, wait=True)

    return X
