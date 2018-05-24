import numpy as np
import matplotlib.pyplot as plt

class KernelDensityEstimation(object):
    def __init__(self, wlen=None, num_points=100, kernel='Gaussian'):
        self.wlen = wlen
        self.num_points = num_points
        self.kernel = kernel
        self.initialize_kernel()

    def initialize_kernel(self):
        if self.kernel == 'Gaussian':
            self.kernel_func = lambda d: np.exp(-d ** 2 / (2*self.wlen ** 2)) / (2 * np.pi)
        else:
            raise NotImplementedError("Kernel function not implemented")

    def estimate_window(self, data):
        n = len(data)
        sigma = np.std(data)
        self.wlen = (4 * (sigma ** 5) / (3 * n)) ** (1/5)

    def apply_kernel(self, x, data):
        y = np.sum(self.kernel_func(x - data)) / len(data) / self.wlen
        return y

    def kde(self, data):
        if self.wlen is None:
            self.estimate_window(data)

        x = np.linspace(min(data), max(data), num=self.num_points)
        y = np.zeros(len(x))

        for i in range(len(y)):
            y[i] = self.apply_kernel(x[i], data)

        return x, y


if __name__ == '__main__':
    data = np.hstack((np.random.randn(200), np.random.rand(1000)*10))

    mdl = KernelDensityEstimation()
    x, y = mdl.kde(data)

    plt.scatter(data, np.zeros(len(data)))
    plt.plot(x, y)
    plt.show()
