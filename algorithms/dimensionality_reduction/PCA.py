'''
Created on Nov 7, 2014
Design for simple PCA algorithm
@author: Yuyang
'''

import numpy as np


class PCAalgor:
    def __init__(self, x):
        print('Initialization')
        self.data = x

    def Normalization(self):
        Average = np.zeros((self.data.shape[1], 1))
        Variance = np.zeros((self.data.shape[1], 1))
        for i in range(0, self.data.shape[1]):
            Average[i] = np.average(self.data[:, i])
            Variance[i] = np.var(self.data[:, i])
        for i in range(0, self.data.shape[1]):
            self.data[:, i] = (self.data[:, i] - Average[i]) / Variance[i]

    def CalCoVar(self):
        self.CoVar = np.dot(np.transpose(self.data), self.data) / self.data.shape[0]
        # return self.CoVar

    def PCA_SVD(self):
        U, s, V = np.linalg.svd(self.CoVar, full_matrices=True)
        return U

    def ShowCentroid(self):
        return self.Centroid