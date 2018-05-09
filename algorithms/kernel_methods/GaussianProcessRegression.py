import numpy as np
import matplotlib.pyplot as plt

class GaussianProcessRegression(object):
    def __init__(self, kernel='Gaussian'):
        self.points = None
        self.kernel_func = None
        self.kernel = kernel
        self.gamma = 1
        self.initialize_kernel()

    def initialize_kernel(self):
        if self.kernel == 'Gaussian':
            self.kernel_func = lambda d: np.exp(-d ** 2 / (2*self.gamma ** 2))
        else:
            raise NotImplementedError("Kernel function not implemented")

    def compute_kernel_matrix(self, xs, xo):
        ret = np.zeros((xs.shape[0], xo.shape[0]))
        for i in range(xs.shape[0]):
            ret[i, :] = self.kernel_func(xo - xs[i, :]).T
        return ret

    def fit(self, points, y):
        self.points = points
        self.y = y

    def predict(self, xs):
        Kxx = self.compute_kernel_matrix(self.points, self.points)
        Kxs_x = self.compute_kernel_matrix(xs, self.points).T
        y_pred = np.matmul(np.matmul(Kxs_x.T, np.linalg.pinv(Kxx)), self.y)

        self.visual_process(xs, y_pred)

    def visual_process(self, xs, y_pred):
        plt.scatter(self.points, self.y)
        plt.plot(xs, y_pred)
        plt.show()

if __name__ == '__main__':
    t = np.arange(0, 10, 1)[:, None]
    y = np.sin(t)
    ts = np.arange(0, 10, 0.2)[:, None]
    mdl = GaussianProcessRegression()
    mdl.fit(t, y)
    mdl.predict(ts)
