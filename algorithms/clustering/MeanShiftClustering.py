from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

class MeanShiftClustering(object):
    """
    Follow descriptions in http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/TUZEL1/MeanShift.pdf
    """
    def __init__(self, k, bandwidth, kernel='Gaussian', max_iter=100, eps=0.1):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.k = k
        self.max_iter = max_iter
        self.eps = eps
        self.center = None
        self.kernel_func = None
        self.initialize_kernel()

    def initialize_kernel(self):
        if self.kernel == 'Gaussian':
            self.kernel_func = lambda x:  norm(0, self.bandwidth).pdf(np.sum(x ** 2, axis=1))
        else:
            raise NotImplementedError("Kernel not supported")

    def shift_mean(self, points, xt):
        """
        Shift mean to find mode
        """
        dist = self.kernel_func(points - xt)[:, None]
        xt = np.matmul(dist.T, points) / np.sum(dist)
        return xt

    def fit(self, points):
        # select k points as initial values
        np.random.shuffle(points)
        self.center = points[:self.k, :]

        # shift distance to control convergence
        d_shift = 100 * self.eps
        i = 0
        while i < self.max_iter and d_shift > self.eps:
            prev_center = np.copy(self.center)
            for idx, c in enumerate(self.center):
                self.center[idx, :] = self.shift_mean(points, c)

            d_shift = np.sum((prev_center - self.center) ** 2)
            i += 1

        self.visalize_cluster(points)

    def visalize_cluster(self, points):
        plt.scatter(points[:, 0], points[:, 1], s=2)
        plt.scatter(self.center[:, 0], self.center[:, 1], s=10)
        plt.show()


if __name__ == '__main__':
    cluster_1 = np.hstack((np.random.normal(0, 1, size=(200, 1)), np.random.normal(0, 1, size=(200, 1))))
    cluster_2 = np.hstack((np.random.normal(5, 1, size=(200, 1)), np.random.normal(5, 1, size=(200, 1))))

    points = np.vstack((cluster_1, cluster_2))

    mdl = MeanShiftClustering(10, 5, max_iter=100, eps=0.01)
    mdl.fit(points)
