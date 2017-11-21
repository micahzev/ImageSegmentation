
"""

implements a very useful Landmark object that always returns a 40x2 dimensional landmark object with points

allows for basic to complex landmark manipulations and analysis

"""


import os
import re
import numpy as np

class Landmark(object):

    def __init__(self, landmark_data):
        if isinstance(landmark_data, str):
            lines = open(landmark_data).readlines()
            points = []
            for x, y in zip(lines[0::2], lines[1::2]):
                points.append(np.array([float(x), float(y)]))
            self.points = np.array(points)

        elif isinstance(landmark_data, np.ndarray) and np.atleast_2d(landmark_data).shape[0] == 1:
            self.points = np.array((landmark_data[:len(landmark_data)/2], landmark_data[len(landmark_data)/2:])).T

        elif isinstance(landmark_data, np.ndarray) and landmark_data.shape[1] == 2:
            self.points = landmark_data


    def as_vector(self):
        return np.hstack((self.points[:, 0], self.points[:, 1]))

    def get_center(self):
        return [self.points[:, 0].min() + (self.points[:, 0].max() - self.points[:, 0].min())/2,
                self.points[:, 1].min() + (self.points[:, 1].max() - self.points[:, 1].min())/2]

    def get_crown(self, is_upper):
        if is_upper:
            return Landmark(self.points[10:30, :])
        else:
            points = np.vstack((self.points[0:10, :], self.points[30:40, :]))
            return Landmark(points)

    def translate_to_origin(self):
        centroid = np.mean(self.points, axis=0)
        points = self.points - centroid
        return Landmark(points)

    def scale_to_unit(self):
        centroid = np.mean(self.points, axis=0)
        scale_factor = np.sqrt(np.power(self.points - centroid, 2).sum())
        points = self.points.dot(1. / scale_factor)
        return Landmark(points)

    def translate(self, vec):
        points = self.points + vec
        return Landmark(points)

    def scale(self, factor):
        centroid = np.mean(self.points, axis=0)
        points = (self.points - centroid).dot(factor) + centroid
        return Landmark(points)

    def capture_bounding_box(self, bbox):
        bbox_h = bbox[1][1] - bbox[0][1]
        scale_h = bbox_h / (self.points[:, 1].max() - self.points[:, 1].min())
        return self.scale(scale_h)

    def scaleposition(self, factor):
        points = self.points.dot(factor)
        return Landmark(points)

    def rotate(self, angle):
        rotmat = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])

        points = np.zeros_like(self.points)
        centroid = np.mean(self.points, axis=0)
        tmp_points = self.points - centroid
        for ind in range(len(tmp_points)):
            points[ind, :] = tmp_points[ind, :].dot(rotmat)
        points = points + centroid

        return Landmark(points)

    def T(self, t, s, theta):
        return self.rotate(theta).scale(s).translate(t)

    def invT(self, t, s, theta):
        return self.translate(-t).scale(1/s).rotate(-theta)


def load_landmarks(landmarks_dir, incisor, mirrored, exclude=None):

    search_string = "-" + str(incisor)

    if exclude:
        original_landmark_files = [landmarks_dir + 'original/' + file for file in os.listdir(landmarks_dir + 'original')
                                   if search_string in file and "s" + str(exclude)+"-" not in file]
    else:
        original_landmark_files = [landmarks_dir + 'original/' + file for file in os.listdir(landmarks_dir + 'original')
                                   if search_string in file]

    files = sorted(original_landmark_files, key=lambda x: int(re.search('[0-9]+', x).group()))

    if mirrored:
        if exclude:
            mirrored_landmark_files = [landmarks_dir + 'mirrored/' + file for file in os.listdir(landmarks_dir + 'mirrored')
                                        if search_string in file and "s" + str(exclude+14)+"-" not in file]
        else:
            mirrored_landmark_files = [landmarks_dir + 'mirrored/' + file for file in os.listdir(landmarks_dir + 'mirrored')
                                       if search_string in file]

        files2 = sorted(mirrored_landmark_files, key=lambda x: int(re.search('[0-9]+', x).group()))
        files = files + files2

    landmarks = []
    for filename in files:
        landmarks.append(Landmark(filename))
    return landmarks



def load_ground(landmarks_dir, incisor, l_o_o):

    search_string = "-" + str(incisor)

    ground = [landmarks_dir + 'original/' + file for file in os.listdir(landmarks_dir + 'original')
                                   if search_string in file and "s"+str(l_o_o)+"-" in file]

    return Landmark(ground[0])



def load_all_landmarks(landmarks_dir, exclude=None):

    search_string = "-"

    if exclude:
        original_landmark_files = [landmarks_dir + 'original/' + file for file in os.listdir(landmarks_dir + 'original')
                                   if "s" + str(exclude)+"-" not in file]
    else:
        original_landmark_files = [landmarks_dir + 'original/' + file for file in os.listdir(landmarks_dir + 'original')
                                   if search_string in file]

    files = sorted(original_landmark_files, key=lambda x: int(re.search('[0-9]+', x).group()))


    landmarks = []
    for filename in files:
        landmarks.append(Landmark(filename))
    return landmarks
