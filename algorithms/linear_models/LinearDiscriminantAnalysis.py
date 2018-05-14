import numpy as np
import matplotlib.pyplot as plt

class LDA(object):
    """
    Follow discription in http://www.sci.utah.edu/~shireen/pdfs/tutorials/Elhabian_LDA09.pdf
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        # compute between-class scatter matrix and with-class scatter matrix
        mu0 = np.mean(X[y == 0, :], axis=0)[:, None]
        mu1 = np.mean(X[y == 1, :], axis=0)[:, None]
        Sb = np.matmul((mu0 - mu1), (mu0 - mu1).T)
        Sw = np.cov(X[y == 0, :].T) + np.cov(X[y == 1, :].T)
        ret = np.linalg.eig(np.matmul(np.linalg.inv(Sw), Sb))

        # find eigenvector correponding to maximum eigen value
        idx = np.argmax(ret[0])
        w = ret[1][:, idx]
        self.w = w

    def visalize(self, points):
        plt.scatter(points[:, 0], points[:, 1], s=2)
        plt.show()


if __name__ == '__main__':
    # data set 1:
    """
    p1 = np.array([[4, 2],
                   [2, 4],
                   [2, 3],
                   [3, 6],
                   [4, 4]])

    p2 = np.array([[9, 10],
                   [6, 8],
                   [9, 5],
                   [8, 7],
                   [10, 8]])
    X = np.vstack((p1, p2))
    y = np.vstack((np.zeros((5, 1)), np.ones((5, 1))))[:, 0]
    """

    # data set 2:
    p1 = np.random.multivariate_normal([0, 0], [[1,0], [0, 1]], 500)
    p2 = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 500)
    X = np.vstack((p1, p2))
    y = np.vstack((np.zeros((500, 1)), np.ones((500, 1))))[:, 0]

    mdl = LDA()
    mdl.fit(X, y)

    print(mdl.w)