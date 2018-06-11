import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

class GaussianMixtureModel(object):
    """
    Follow descriptions in
        [1] http://statweb.stanford.edu/~tibs/stat315a/LECTURES/em.pdf
        [2] http://www.cse.iitm.ac.in/~vplab/courses/DVP/PDF/gmm.pdf
    """
    def __init__(self, k, max_iter=50):
        self.k = k
        self.max_iter = max_iter
        self.center = None

    def fit(self, points):
        self.initialize(points)

        for i in range(self.max_iter):
            # expectation
            self.compute_responsibility(points)

            # maximization
            self.update_parameter(points)

        self.visalize_cluster(points)

    def initialize(self, points):
        num_samples = points.shape[0]
        num_features = points.shape[1]

        indices = np.random.randint(low=0, high=self.k, size=num_samples)
        self.responsibility = np.zeros((num_samples, self.k))
        self.mixing_coef = np.ones((self.k, 1)) / self.k
        self.mu = np.zeros((num_features, self.k))
        self.sigma = np.zeros((num_features, num_features, self.k))

        for i in range(self.k):
            indexing = (indices == i)
            self.mu[:, i] = np.mean(points[indexing, :], axis=0)
            self.sigma[:, :, i] = np.cov(points[indexing, :].T)

    def compute_responsibility(self, data):
        for i in range(self.responsibility.shape[0]):
            for j in range(self.responsibility.shape[1]):
                self.responsibility[i, j] = self.mixing_coef[j] * multivariate_normal.pdf(data[i, :], self.mu[:, j], self.sigma[:, :, j])

        # row-wise normalization
        self.responsibility = self.responsibility / (self.responsibility.sum(axis=1)[:, None])

    def update_parameter(self, data):
        for j in range(self.k):
            self.mu[:, j] = np.matmul(data.T, self.responsibility[:, j]) / np.sum(self.responsibility[:, j])
            self.sigma[:, :, j] = np.matmul((data - self.mu[:, j]).T,
                                            (data-self.mu[:, j]) * np.matmul(self.responsibility[:, j][:, None], np.ones((1, data.shape[1]))))\
                                  / np.sum(self.responsibility[:, j])
            self.mixing_coef[j] = np.mean(self.responsibility[:, j])

    def plot_cov_ellipse(self, cov_mtx, pos, nstd=2, ax=None, **kwargs):
        """
        From https://github.com/SJinping/Gaussian-ellipse/blob/master/gaussian_%20ellipse.py
        """

        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:, order]

        if ax is None:
            ax = plt.gca()

        vals, vecs = eigsorted(cov_mtx)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(vals)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

        ax.add_artist(ellip)
        return ellip


    def visalize_cluster(self, points):
        plt.scatter(points[:, 0], points[:, 1], s=2)

        for j in range(self.k):
            self.plot_cov_ellipse(self.sigma[:, :, j], self.mu[:, j], alpha=0.2)

        plt.show()


if __name__ == '__main__':
    cluster_1 = np.hstack((np.random.normal(0, 1, size=(100, 1)), np.random.normal(0, 1, size=(100, 1))))
    cluster_2 = np.hstack((np.random.normal(5, 1, size=(100, 1)), np.random.normal(5, 1, size=(100, 1))))
    cluster_3 = np.hstack((np.random.normal(0, 1, size=(100, 1)), np.random.normal(5, 1, size=(100, 1))))

    points = np.vstack((cluster_1, cluster_2, cluster_3))

    mdl = GaussianMixtureModel(k=3)
    mdl.fit(points)