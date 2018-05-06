'''
Created on Nov 6, 2014
Design for simple n-variable K mean algorithm
@author: Yuyang
'''
import numpy as np


class KmeanNet:
    def __init__(self, CenPoint, nVar, DataRange):
        print('Initialization')
        self.nVar = nVar
        self.CenPoint = CenPoint
        self.Centroid = (np.random.rand(nVar, CenPoint)) * DataRange

    def CalDistance(self, a, b):
        return np.sum((a - b) * (a - b))

    def Update(self, x):
        NewCentroid = np.zeros((self.nVar + 1, self.CenPoint))
        for j in range(0, x.shape[1]):
            itr = 0;
            TmpDis = self.CalDistance(self.Centroid[:, itr], x[:, j])
            for i in range(0, self.CenPoint):
                CurDis = self.CalDistance(self.Centroid[:, i], x[:, j])
                if (CurDis < TmpDis):
                    itr = i;
                    TmpDis = CurDis
            NewCentroid[0:self.nVar, itr] = NewCentroid[0:self.nVar, itr] + x[:, j];
            NewCentroid[self.nVar, itr] = NewCentroid[self.nVar, itr] + 1;

        for k in range(0, self.CenPoint):
            if (NewCentroid[self.nVar, k] != 0):
                self.Centroid[:, k] = NewCentroid[0:self.nVar, k] / NewCentroid[self.nVar, k]
                print
                NewCentroid[self.nVar, k]

    def ShowCentroid(self):
        return self.Centroid