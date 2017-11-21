"""

PCA done over incisor landmarks to find active shape model of tooth.

"""

import numpy as np

class Active_Shape_Model(object):

    def __init__(self, mean_shape, aligned_landmarks):

        mat = []
        for lm in aligned_landmarks:
            mat.append(lm.as_vector())
        XnewVec = np.array(mat)

        S = np.cov(XnewVec, rowvar=0)

        self.k = len(mean_shape.points)
        self.mean_shape = mean_shape
        self.covariance = S
        self.aligned_shapes = aligned_landmarks

        eigvals, eigvecs = np.linalg.eigh(S)
        idx = np.argsort(-eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        self.scores = np.dot(XnewVec, eigvecs)
        self.mean_scores = np.dot(mean_shape.as_vector(), eigvecs)
        self.variance_explained = np.cumsum(eigvals/np.sum(eigvals))

        for index,item in enumerate(self.variance_explained > 0.99):
            if item:
                npcs = index
                break

        M = []
        for i in range(0, npcs):
            M.append(np.sqrt(eigvals[i]) * eigvecs[:, i])
        self.pc_modes = np.array(M).squeeze().T
