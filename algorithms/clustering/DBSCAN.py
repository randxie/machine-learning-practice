import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict, deque

UNDEFINED = -999

class DBSCAN(object):
    """
    Follow descriptions in https://en.wikipedia.org/wiki/DBSCAN
    """
    def __init__(self, eps, minPts, dist_metric='Euclidean'):
        self.eps = eps
        self.minPts = minPts
        self.dist_metric = dist_metric
        self.initialize_dist_func()
        self.cluster_labels = None

    def initialize_dist_func(self):
        if self.dist_metric == 'Euclidean':
            self.distFunc = lambda x, y: np.sum((x-y) ** 2)
        else:
            raise NotImplementedError("Distance metric not supported")

    def fit(self, points):
        cur_label = 0
        self.cluster_labels = UNDEFINED * np.ones((points.shape[0], 1))
        for idx, pt in enumerate(points):
            if self.cluster_labels[idx] != UNDEFINED:
                continue

            neighbor_indices = self.range_query(points, pt)

            if len(neighbor_indices) < self.minPts:
                self.cluster_labels[idx] = 0
                continue

            cur_label += 1
            self.cluster_labels[idx] = cur_label

            expand_nn = deque(neighbor_indices)

            while len(expand_nn) > 0:
                n_idx = expand_nn.popleft()
                if self.cluster_labels[n_idx] == 0:
                    self.cluster_labels[n_idx] = cur_label

                if self.cluster_labels[n_idx] == UNDEFINED:
                    self.cluster_labels[n_idx] = cur_label
                    q_neighbors = self.range_query(points, points[n_idx, :])

                    if len(q_neighbors) >= self.minPts:
                        expand_nn.extend(q_neighbors)

        self.visalize_cluster(points)

    def visalize_cluster(self, points):
        plt.scatter(points[:, 0], points[:, 1], c=self.cluster_labels[:, 0])
        plt.show()

    def range_query(self, points, query_pt):
        nn_indices = []

        for idx, pt in enumerate(points):
            if self.distFunc(query_pt, pt) <= self.eps:
                nn_indices.append(idx)

        return nn_indices


if __name__ == '__main__':
    cluster_1 = np.hstack((np.random.normal(0, 1, size=(200, 1)), np.random.normal(0, 1, size=(200, 1))))
    cluster_2 = np.hstack((np.random.normal(0, 1, size=(200, 1)), np.random.normal(0, 1, size=(200, 1))))

    cluster_1 /= np.sqrt(np.sum(cluster_1**2, axis=1)[:,  np.newaxis])
    cluster_2 /= 0.25 * np.sqrt(np.sum(cluster_2**2, axis=1)[:,  np.newaxis])

    points = np.vstack((cluster_1, cluster_2))

    mdl = DBSCAN(1, 5)
    mdl.fit(points)
