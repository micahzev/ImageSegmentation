"""

Dental xray is split in half by finding the most probably location of the split between upper and lower incisors

this is based on pixel intensity histograms

"""

import math

import cv2
import numpy as np
import scipy.fftpack
import scipy.signal
from scipy.ndimage import morphology
import plotting_code


def split(radiograph, interval=50, show=False):
    img = cv2.cvtColor(radiograph, cv2.COLOR_BGR2GRAY)
    img = morphology.white_tophat(img, size=400)

    height, width = img.shape
    mask = 255-img
    filt = gaussian_filter(450, width)
    if width % 2 == 0:
        filt = filt[:-1]
    mask = np.multiply(mask, filt)

    minimal_points = []
    for x in range(interval, width, interval):
        hist = []
        for y in range(int(height*0.4), int(height*0.7), 1):
            hist.append((np.sum(mask[y][x-interval:x+interval+1]), x, y))

        fft = scipy.fftpack.rfft([intensity for (intensity, _, _) in hist])
        fft[30:] = 0
        smoothed = scipy.fftpack.irfft(fft)

        indices = scipy.signal.argrelmax(smoothed)[0]
        minimal_points_width = []
        for idx in indices:
            minimal_points_width.append(hist[idx])
        minimal_points_width.sort(reverse=True)

        count = 0
        to_keep = []
        for min_point in minimal_points_width:
            _, _, d = min_point
            if all(abs(b-d) > 150 for _, _, b in to_keep) and count < 4:
                count += 1
                to_keep.append(min_point)
        minimal_points.extend(to_keep)

    edges = []
    for _, x, y in minimal_points:
        min_intensity = float('inf')
        min_coords = (-1, -1)
        for _, u, v in minimal_points:
            intensity = _edge_intensity(mask, (x, y), (u, v))
            if x < u and intensity < min_intensity and abs(v-y) < 0.1*height:
                min_intensity = intensity
                min_coords = (u, v)
        if min_coords != (-1, -1):
            edges.append([(x, y), min_coords])

    paths = []
    for edge in edges:
        new_path = True

        for path in paths:
            if path.edges[-1] == edge[0]:
                new_path = False
                path.extend(edge)
        if new_path:
            paths.append(Path([edge[0], edge[1]]))

    mask2 = mask * (255/mask.max())
    mask2 = mask2.astype('uint8')

    map(lambda p: p.trim(mask2), paths)
    paths = remove_short_paths(paths, width, 0.3)

    best_path = sorted([(p.intensity(img) / (p.length()), p) for p in paths])[0][1]

    if show:
        plotting_code.plot_jaw_split(img, minimal_points, paths, best_path)

    return best_path


class Path(object):

    def __init__(self, edges):
        self.edges = edges

    def get_part(self, min_bound, max_bound):
        edges = []
        for edge in self.edges:
            if edge[0] > min_bound and edge[0] < max_bound:
                edges.append(edge)

        return edges

    def extend(self, edge):
        self.edges.append(edge[1])

    def intensity(self, radiograph):
        intensity = 0
        for i in range(0, len(self.edges)-1):
            intensity += _edge_intensity(radiograph, self.edges[i], self.edges[i+1])
        return intensity

    def trim(self, radiograph):
        mean_intensity = self.intensity(radiograph) / self.length()
        while len(self.edges) > 2:
            if mean_intensity > _edge_intensity(radiograph, self.edges[0], self.edges[1]) / \
                    math.hypot(self.edges[1][0]-self.edges[0][0], self.edges[1][1]-self.edges[0][1]):
                del self.edges[0]
            else:
                break
        while len(self.edges) > 2:
            if mean_intensity > _edge_intensity(radiograph, self.edges[-1], self.edges[-2]) / \
                    math.hypot(self.edges[-1][0]-self.edges[-2][0], self.edges[-1][1]-self.edges[-2][1]):
                del self.edges[-1]
            else:
                break

    def length(self):
        return np.sum(np.sqrt(np.sum(np.power(np.diff(self.edges, axis=0), 2), axis=1)))


def remove_short_paths(paths, width, ratio):
    return filter(lambda p: p.length() >= width*ratio, paths)


def _edge_intensity(radiograph, p1, p2):
    intensities = createLineIterator(radiograph, p1, p2)
    return sum(intensities)


def gaussian_filter(sigma, filter_length=None):
    def gaussian_function(sigma, u):
        return 1/(math.sqrt(2*math.pi)*sigma)*math.e**-(u**2/(2*sigma**2))

    if filter_length is None:
        filter_length = math.ceil(sigma*5)
        filter_length = 2*(int(filter_length)/2) + 1

    sigma = float(sigma)

    result = np.asarray([gaussian_function(sigma, u) for u in range(-(filter_length/2), filter_length/2 + 1, 1)])
    result = result / result.sum()

    return result


def createLineIterator(img, P1, P2):
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
    elif P1Y == P2Y:
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else:
        steepSlope = dYa > dXa
        if steepSlope:
            slope = float(dX)/float(dY)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
        else:
            slope = float(dY)/float(dX)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer[:,2]
