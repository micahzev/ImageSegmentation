import math
import numpy as np
from scipy import linspace, asarray

class Grey_Level_Model(object):

    def __init__(self):
        self.profiles = []
        self.mean_profile = None
        self.covariance = []

    def build(self, images, gimages, models, point_ind, k):



        for ind in range(len(images)):
            self.profiles.append(Profile(images[ind], gimages[ind], models[ind], point_ind, k))


        mat = []
        for profile in self.profiles:
            mat.append(profile.samples)
        mat = np.array(mat)
        self.mean_profile = (np.mean(mat, axis=0))
        self.covariance = (np.cov(mat, rowvar=0))

    def mahalanobis_dist(self, samples):

        return (samples - self.mean_profile).T \
            .dot(self.covariance) \
            .dot(samples - self.mean_profile)


class Profile(object):


    def __init__(self, image, grad_image, model, point_ind, k):

        self.image = image
        self.grad_image = grad_image
        self.model_point = model.points[point_ind, :]
        self.k = k
        self.normal = self.__calculate_normal(model.points[(point_ind - 1) % 40, :],
                                              model.points[(point_ind + 1) % 40, :])
        self.points, self.samples = self.__sample()

    def __calculate_normal(self, p_prev, p_next):

        n1 = normal(p_prev, self.model_point)
        n2 = normal(self.model_point, p_next)
        n = (n1 + n2) / 2
        return n / np.linalg.norm(n)

    def __sample(self):

        # Take a slice of the image in pos and neg normal direction
        pos_points, pos_values, pos_grads = self.__slice_image2(-self.normal)
        neg_points, neg_values, neg_grads = self.__slice_image2(self.normal)

        # Merge the positive and negative slices in one list
        neg_values = neg_values[::-1]  # reverse
        neg_grads = neg_grads[::-1]  # reverse
        neg_points = neg_points[::-1]  # reverse
        points = np.vstack((neg_points, pos_points[1:, :]))
        values = np.append(neg_values, pos_values[1:])
        grads = np.append(neg_grads, pos_grads[1:])

        # Compute the final sample values
        div = max(sum([math.fabs(v) for v in values]), 1)
        samples = [float(g) / div for g in grads]

        return points, samples

    def __slice_image(self, direction, *arg, **kws):

        from scipy.ndimage import map_coordinates

        a = asarray(self.model_point)
        b = asarray(self.model_point + direction * self.k)
        coordinates = (a[:, np.newaxis] * linspace(1, 0, self.k + 1) +
                       b[:, np.newaxis] * linspace(0, 1, self.k + 1))
        values = map_coordinates(self.image, coordinates, order=1, *arg, **kws)
        grad_values = map_coordinates(self.grad_image, coordinates, order=1, *arg, **kws)
        return coordinates.T, values, grad_values

    def __slice_image2(self, direction):

        a = asarray(self.model_point)
        b = asarray(self.model_point + direction * self.k)
        coordinates = (a[:, np.newaxis] * linspace(1, 0, self.k + 1) +
                       b[:, np.newaxis] * linspace(0, 1, self.k + 1))
        values = self.image[coordinates[1].astype(np.int), coordinates[0].astype(np.int)]
        grad_values = self.grad_image[coordinates[1].astype(np.int), coordinates[0].astype(np.int)]
        return coordinates.T, values, grad_values

def normal(p1, p2):

    return np.array([p1[1] - p2[1], p2[0] - p1[0]])