'''

Implements procrustes anlaysis

'''

import numpy as np
from landmark_code import Landmark
import plotting_code

def procrustes(landmarks):

    aligned_shapes = list(landmarks)

    aligned_shapes = [shape.translate_to_origin() for shape in aligned_shapes]

    x0 = aligned_shapes[0].scale_to_unit()
    mean_shape = x0

    while True:

        for ind, lm in enumerate(aligned_shapes):
            aligned_shapes[ind] = shape_alignment(lm, mean_shape)

        mat = []
        for lm in aligned_shapes:
            mat.append(lm.as_vector())
        mat = np.array(mat)
        new_mean_shape = Landmark(np.mean(mat, axis=0))
        new_mean_shape = shape_alignment(new_mean_shape, x0)
        new_mean_shape = new_mean_shape.scale_to_unit().translate_to_origin()

        # plotting_code.plot_procrustes(new_mean_shape, aligned_shapes)

        if ((mean_shape.as_vector() - new_mean_shape.as_vector()) < 1e-10).all():
            break

        mean_shape = new_mean_shape

    return mean_shape, aligned_shapes


def shape_alignment(shape1, shape2):

    s, theta = alignment(shape1, shape2)

    shape1 = shape1.rotate(theta)
    shape1 = shape1.scale(s)

    xx = np.dot(shape1.as_vector(), shape2.as_vector())
    return Landmark(shape1.as_vector()*(1.0/xx))


def alignment(shape1, shape2):

    shape1 = shape1.as_vector()
    shape2 = shape2.as_vector()

    l1 = len(shape1)/2
    l2 = len(shape2)/2

    shape1_centroid = np.array([np.mean(shape1[:l1]), np.mean(shape1[l1:])])
    shape2_centroid = np.array([np.mean(shape2[:l2]), np.mean(shape2[l2:])])
    shape1 = [x - shape1_centroid[0] for x in shape1[:l1]] + [y - shape1_centroid[1] for y in shape1[l1:]]
    shape2 = [x - shape2_centroid[0] for x in shape2[:l2]] + [y - shape2_centroid[1] for y in shape2[l2:]]

    norm_shape1_sq = (np.linalg.norm(shape1)**2)
    a = np.dot(shape1, shape2) / norm_shape1_sq

    b = (np.dot(shape1[:l1], shape2[l2:]) - np.dot(shape1[l1:], shape2[:l2])) / norm_shape1_sq

    s = np.sqrt(a**2 + b**2)

    theta = np.arctan(b/a)

    return s, theta
