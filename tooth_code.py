
"""

Code for the model of each incisor tooth.

The model comes equipped with a fitting algorithm that iteratively fits an active shape model to the surrounding pixel
neighbourhood


"""


import math

import plotting_code
from manual_selection_tool import manual_selection
import numpy as np
import xray_code as xray
from landmark_code import Landmark
from grey_level_model import Grey_Level_Model
from grey_level_model import Profile



class Model(object):

    def __init__(self, incisor):
        self.incisor_nr = incisor

        self.pyramid_levels = 1

    def train(self, lms, imgs, as_model):

        self.k = 10

        self.asm = as_model

        # plotting_code.plot_asm(self.asm)

        pyramids = [xray.build_gauss_pyramid(image, self.pyramid_levels) for image in imgs]
        lms_pyramids = [[lm.scaleposition(1.0/2**i)
                         for i in range(0, self.pyramid_levels+1)]
                        for lm in lms]


        self.glms = []
        for i in range(0, self.pyramid_levels+1):
            pimages = [xray.enhance(image) for image in zip(*pyramids)[i]]
            gimages = [xray.sobelize(img) for img in pimages]
            lms = zip(*lms_pyramids)[i]

            glms = []
            for i in range(0, 40):
                glm = Grey_Level_Model()
                glm.build(pimages, gimages, lms, i, self.k)
                glms.append(glm)
            self.glms.append(glms)

    def fit(self, X, testimg, m=15):

        maximum_iterations = 50

        if not m > self.k:
            raise ValueError("m <= k")

        pyramid = xray.build_gauss_pyramid(testimg, self.pyramid_levels)
        X = X.scaleposition(1.0 / 2**(self.pyramid_levels+1))
        for img, glms in zip(reversed(pyramid), reversed(self.glms)):
            X = X.scaleposition(2)
            X = self.fit_one_level(X, img, glms, m, maximum_iterations)

        return X


    def fit_one_level(self, X, testimg, glms, m, max_iter):

        img = xray.enhance(testimg)
        gimg = xray.sobelize(img)

        b = np.zeros(self.asm.pc_modes.shape[1])
        X_prev = Landmark(np.zeros_like(X.points))

        nb_iter = 0
        n_close = 0
        best = np.inf
        best_Y = None
        total_s = 1
        total_theta = 0
        while (n_close < 16 and nb_iter <= max_iter):

            Y, n_close, quality = self.findfits(X, img, gimg, glms, m)
            if quality < best:
                best = quality
                best_Y = Y
            # plotting_code.plot_landmarks_on_image([X, Y], testimg, wait=False, title="Fitting incisor nr. %d" % (self.incisor_nr,))

            if nb_iter == max_iter:
                Y = best_Y

            b, t, s, theta = self.update_fit_params(X, Y, testimg)

            b = np.clip(b, -3, 3)

            s = np.clip(s, 0.999, 1.001)
            if total_s * s > 1.01 or total_s * s < 0.99:
                s = 1
            total_s *= s

            theta = np.clip(theta, -math.pi/72, math.pi/72)
            if total_theta + theta > math.pi/36 or total_theta + theta < - math.pi/36:
                theta = 0
            total_theta += theta


            X_prev = X
            X = Landmark(X.as_vector() + np.dot(self.asm.pc_modes, b)).T(t, s, theta)
            # plotting_code.plot_landmarks_on_image([X_prev, X], testimg, wait=False, title="Fitting incisor nr. %d" % (self.incisor_nr,))

            nb_iter += 1

        return X

    def findfits(self, X, img, gimg, glms, m):

        fits = []
        n_close = 0

        profiles = []
        bests = []
        qualities = []
        for ind in range(len(X.points)):

            profile = Profile(img, gimg, X, ind, m)
            profiles.append(profile)


            dmin, best = np.inf, None
            dists = []
            for i in range(self.k, self.k+2*(m-self.k)+1):
                subprofile = profile.samples[i-self.k:i+self.k+1]
                dist = glms[ind].mahalanobis_dist(subprofile)
                dists.append(dist)
                if dist < dmin:
                    dmin = dist
                    best = i

            bests.append(best)
            qualities.append(dmin)
            best_point = [int(c) for c in profile.points[best, :]]

            is_upper = True if self.incisor_nr < 5 else False
            if (((is_upper and (ind > 9 and ind < 31)) or
                 (not is_upper and (ind < 11 or ind > 29))) and
                    best > 3*m/4 and best < 5*m/4):
                n_close += 1

            # plotting_code.plot_fits(gimg, profile, glms[ind], dists, best_point, self.k, m)

        bests.extend(bests)
        bests = np.rint(medfilt(np.asarray(bests), 5)).astype(int)
        for best, profile in zip(bests, profiles):
            best_point = [int(c) for c in profile.points[best, :]]
            fits.append(best_point)

        is_upper = True if self.incisor_nr < 5 else False
        if is_upper:
            quality = np.mean(qualities[10:30])
        else:
            quality = np.mean(qualities[0:10] + qualities[30:40])

        return Landmark(np.array(fits)), n_close, quality

    def update_fit_params(self, X, Y, testimg):

        b = np.zeros(self.asm.pc_modes.shape[1])
        b_prev = np.ones(self.asm.pc_modes.shape[1])
        i = 0
        while (np.mean(np.abs(b-b_prev)) >= 1e-14):
            i += 1
            x = Landmark(X.as_vector() + np.dot(self.asm.pc_modes, b))

            is_upper = True if self.incisor_nr < 5 else False
            t, s, theta = align_params(x.get_crown(is_upper), Y.get_crown(is_upper))

            y = Y.invT(t, s, theta)

            yacc = Landmark(y.as_vector() / np.dot(y.as_vector(), X.as_vector().T))

            b_prev = b
            b = np.dot(self.asm.pc_modes.T, (yacc.as_vector()-X.as_vector()))

        return b, t, s, theta





def align_params(x1, x2):

    x1 = x1.as_vector()
    x2 = x2.as_vector()

    l1 = len(x1)/2
    l2 = len(x2)/2

    x1_centroid = np.array([np.mean(x1[:l1]), np.mean(x1[l1:])])
    x2_centroid = np.array([np.mean(x2[:l2]), np.mean(x2[l2:])])
    x1 = [x - x1_centroid[0] for x in x1[:l1]] + [y - x1_centroid[1] for y in x1[l1:]]
    x2 = [x - x2_centroid[0] for x in x2[:l2]] + [y - x2_centroid[1] for y in x2[l2:]]

    norm_x1_sq = (np.linalg.norm(x1)**2)
    a = np.dot(x1, x2) / norm_x1_sq

    b = (np.dot(x1[:l1], x2[l2:]) - np.dot(x1[l1:], x2[:l2])) / norm_x1_sq

    s = np.sqrt(a**2 + b**2)

    theta = np.arctan(b/a)

    t = x2_centroid - x1_centroid

    return t, s, theta



def medfilt(x, k):

    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i+1)] = x[j:]
        y[-j:, -(i+1)] = x[-1]
    return np.median(y, axis=1)
