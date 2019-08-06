"""
Author : Jiwoo Park
Date : 2019-08-04
Desc : Valuing Option Price under Transaction Cost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import standard_normal, seed
from scipy.stats import norm

import sys

import datetime
import time
import bspline
import bspline.splinelab as splinelab


class TrQBlackScholes():
    def __init__(self,
                 mu,
                 vol,
                 s0,
                 T,
                 r,
                 num_steps,
                 num_paths,
                 risk_lambda,
                 tr_alpha):
        self.mu = mu
        self.vol = vol
        self.s0 = s0
        self.T = T
        self.r = r
        self.num_steps = num_steps
        self.num_paths = num_paths
        self.risk_lambda = risk_lambda

        self.dt = T / num_steps
        self.gamma = np.exp(-r * self.dt)
        self.tr_alpha = tr_alpha

        self.s_values = np.zeros((self.num_paths, self.num_steps + 1), 'float')
        self.s_values[:, 0] = s0 * np.ones(self.num_paths, 'float')
        self.option_values = np.zeros((self.num_paths, self.num_steps + 1), 'float')
        self.intrinsic_values = np.zeros((self.num_paths, self.num_steps + 1), 'float')

        self.b_values = np.zeros((self.num_paths, self.num_steps + 1), 'float')
        self.opt_hedge = np.zeros((self.num_paths, self.num_steps + 1), 'float')

        self.X = None
        self.data = None
        self.delta_S_hat = None
        coef = 1.0 / (2 * self.gamma * self.risk_lambda)
        self.coef = coef

    def gen_path(self):
        # Path Generator (Black Scholes )
        np.seed(42)

        for i in range(1, self.num_steps + 1):
            std_norm = standard_normal(self.num_paths)
            exp_pow = (self.mu - self.vol ** 2 / 2) * self.dt \
                      + self.vol * np.sqrt(self.dt) * std_norm
            self.s_values[:, i] = self.s_values[:, i - 1] + np.exp(exp_pow)

        delta_S = (1 - self.tr_alpha) * self.s_values[:, 1:] - 1 / self.gamma * self.s_values[:, :self.num_steps]
        self.delta_S_hat = np.apply_along_axis(lambda x: x - np.mean(x), axis=0, arr=delta_S)
        self.X = - (self.mu - 0.5 * self.vol ** 2) * np.arange(self.num_steps + 1) * self.dt + np.log(self.s_values)

        X_min = np.min(np.min(self.X))
        X_max = np.max(np.max(self.X))

        print("Shape of X : {} \n Max : {} \n Min {}".format(self.X.shape, X_max, X_min))

        p = 4
        ncolloc = 12
        tau = np.linspace(X_min, X_max, ncolloc)

        k = splinelab.aptknt(tau, p)
        basis = bspline.Bspline(k, p)

        num_basis = ncolloc
        self.data = np.zeros((self.num_steps + 1, self.num_paths, num_basis))

        t0 = time.time()
        for ix in np.arange(self.num_steps + 1):
            x = self.X[:, ix]
            self.data[ix, :, :] = np.array([basis(el) for el in x])
        t1 = time.time()
        print("\n Time for basis expansion {}".format(t1 - t0))


    def function_A_vec(self, t, reg_param=1e-3):
        pass

    def function_B_vec(self, t, pi_hat):
        pass

    def roll_backward(self):
        """
        Roll the price and optimal hedge
        :return:
        """
        pass



