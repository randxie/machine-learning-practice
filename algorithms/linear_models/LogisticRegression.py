# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 20:06:31 2016
@author: randxie
"""
import numpy as np
from scipy.misc import logsumexp

class LrSolver():
    def __init__(self, data, num_feature, num_class):
        # general parameters
        self.max_iter = 200
        self.lreg_factor = 1
        self.N = data.shape[0]
        self.lreg = 1/self.N
        self.exp_thres = 1000
        self.tol = 0.0001
        self.eta = 0.01
        # parameter specific to GD,SGD and SVRG (same initial point)
        # GD
        self.GD_grad_eval = self.max_iter
        self.b_GD = np.random.random_sample((num_feature, num_class))

        # SGD
        self.SGD_grad_eval = self.max_iter*self.N
        self.b_SGD = self.b_GD

        # SVRG
        self.out_in_ratio = 100
        self.SVRG_grad_eval = self.max_iter * self.N
        self.SVRG_outloop = int(np.ceil(np.sqrt(self.SVRG_grad_eval/(self.out_in_ratio))))
        self.SVRG_inloop = int(self.out_in_ratio*self.SVRG_outloop)
        self.b_SVRG = self.b_GD

    def fi_beta_full(self,b,data):
        GD_full = np.zeros(b.shape)
        for i in range(self.N):
            GD_full = GD_full + self.fi_beta(b,data[[i],0:-1].T, data[i,-1])
        return GD_full/self.N

    def fi_beta(self,b,xi,yi):
        GD_i = np.zeros(b.shape)
        GD_i = np.dot(-xi,(np.exp(-b.T.dot(xi))).T) / self.cal_sum(b,xi)
        GD_i[:,[yi]] = GD_i[:,[yi]] + xi
        # add regularization
        GD_i = GD_i + self.lreg*self.lreg_factor*b
        return GD_i

    def cal_sum(self,b,xi):
        val = logsumexp(-b.T.dot(xi))
        if val < self.exp_thres:
            return np.exp(val)
        else:
            return np.exp(self.exp_thres)

    def SGD(self,data, data_test):
        self.SGD_perm = np.zeros((3,self.max_iter))
        data_idx = np.random.randint(low=0, high=self.N, size=self.SGD_grad_eval)
        count = 0
        for itrs in range(self.SGD_grad_eval):
            if (itrs%self.N==0):
                if count<self.SGD_perm.shape[1]:
                    self.SGD_perm[0,count] = itrs
                    self.SGD_perm[1,count] = 1 - self.cal_TR(self.b_SGD,data)
                    self.SGD_perm[2,count] = 1 - self.cal_TR(self.b_SGD,data_test)
                    count = count+1
            i = data_idx[itrs]
            self.b_SGD = self.b_SGD - self.eta*self.fi_beta(self.b_SGD,data[[i],0:-1].T, data[i,-1])

    def GD(self,data,data_test):
        self.GD_perm = np.zeros((3,self.GD_grad_eval))
        count = 0
        for itrs in range(self.GD_grad_eval):
            if count<self.GD_perm.shape[1]:
                self.GD_perm[0,count] = itrs*self.N
                self.GD_perm[1,count] = 1 - self.cal_TR(self.b_GD,data)
                self.GD_perm[2,count] = 1 - self.cal_TR(self.b_GD,data_test)
                count = count+1
            self.b_GD = self.b_GD - self.eta*self.fi_beta_full(self.b_GD,data)

    def SVRG(self,data,data_test):
        self.SVRG_perm = np.zeros((3,self.SVRG_outloop))
        count = 0
        for s in range(self.SVRG_outloop):
            # first calculate full gradient
            us = self.fi_beta_full(self.b_SVRG,data)
            if count<self.SVRG_perm.shape[1]:
                self.SVRG_perm[0,count] = s*self.SVRG_inloop
                self.SVRG_perm[1,count] = 1-self.cal_TR(self.b_SVRG,data)
                self.SVRG_perm[2,count] = 1-self.cal_TR(self.b_SVRG,data_test)
                count = count+1
            b_old = self.b_SVRG
            data_idx = np.random.randint(low=0, high=self.N, size=self.SVRG_inloop)
            for k in range(self.SVRG_inloop):
                i = data_idx[k]
                dxk = self.fi_beta(self.b_SVRG,data[[i],0:-1].T, data[i,-1])
                dyk = self.fi_beta(b_old,data[[i],0:-1].T, data[i,-1])
                self.b_SVRG = self.b_SVRG - self.eta*(dxk - dyk + us)

    def cal_TR(self,b,data):
        y_hat = np.argmax(np.exp(-np.dot(data[:,0:-1],b)), axis=1)
        dy = (y_hat == data[:,-1])
        return sum(dy)*1.0/y_hat.shape[0]