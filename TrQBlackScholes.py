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

REG_PARAM = 1e-3

class TrQBlackScholes():
    def __init__(self,
                 mu: float,
                 vol: float,
                 s0: float,
                 T: int,
                 K : float,
                 r: float,
                 num_steps: int,
                 num_paths: int,
                 risk_lambda: float,
                 tr_alpha: float):
        self.mu = mu
        self.vol = vol
        self.s0 = s0
        self.T = T
        self.K = K
        self.r = r
        self.num_steps = num_steps
        self.num_paths = num_paths
        self.risk_lambda = risk_lambda

        self.dt = T / num_steps
        self.gamma = np.exp(-r * self.dt)
        self.tr_alpha = tr_alpha

        self.s_values = np.zeros((self.num_paths, self.num_steps + 1), 'float')
        self.delta_S = None
        self.s_values[:, 0] = s0 * np.ones(self.num_paths, 'float')

        self.opt_hedge = np.zeros((self.num_paths, self.num_steps + 1), 'float')

        self.X = None
        self.data = None
        self.delta_S_hat = None
        coef = 1.0 / (2 * self.gamma * self.risk_lambda)
        self.coef = coef

        self.pi = np.zeros((self.num_paths, self.num_steps + 1), 'float')
        self.pi_hat = np.zeros((self.num_paths, self.num_steps + 1), 'float')

        self.r = np.zeros((self.num_paths, self.num_steps + 1), 'float')

    def gen_path(self):
        # Path Generator (Black Scholes )
        seed(42)

        for i in range(1, self.num_steps + 1):
            std_norm = standard_normal(self.num_paths)
            exp_pow = (self.mu - self.vol ** 2 / 2) * self.dt \
                      + self.vol * np.sqrt(self.dt) * std_norm
            self.s_values[:, i] = self.s_values[:, i - 1] * np.exp(exp_pow)

        delta_S = (1 - self.tr_alpha) * self.s_values[:, 1:] - 1 / self.gamma * self.s_values[:, :self.num_steps]
        self.delta_S = delta_S
        self.delta_S_hat = np.apply_along_axis(lambda x: x - np.mean(x), axis=0, arr=delta_S)
        self.X = - (self.mu - 0.5 * self.vol ** 2) * np.arange(self.num_steps + 1) * self.dt + np.log(self.s_values)

        X_min = np.min(np.min(self.X))
        X_max = np.max(np.max(self.X))

        print("Shape of X : {} \n Max : {} \n Min : {}".format(self.X.shape, X_max, X_min))

        self.pi[:, -1] = np.maximum(self.s_values[:, -1] - self.K, 0)
        self.pi_hat[:, -1] = self.pi[:, -1] - np.mean(self.pi[:, -1])

        self.r[:, -1] = -self.risk_lambda * np.var(self.pi[:, -1])

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
        print("\nTime for basis expansion {}".format(t1 - t0))

    def function_A_vec(self, t, reg_param=1e-3):
        """ Equation for matrix A
        :param t:
        :param reg_param:
        :return:
        """

        x_data = self.data[t, :, :]
        num_basis_funcs = x_data.shape[1]
        self_dS = self.delta_S_hat[:, t]
        x_data = x_data.T @  self_dS
        mat_A = x_data.T @ x_data

        return mat_A + reg_param * np.eye(num_basis_funcs)

    def function_B_vec(self, t, pi_hat):
        x_data = self.data[t, :, :]
        this_dS = self.delta_S_hat[:, t]
        coef = 1 / (2 * self.gamma * self.risk_lambda)
        temp = pi_hat * this_dS + coef * self.delta_S[:, t]
        mat_B = x_data.T @ (pi_hat * this_dS + coef * self.delta_S[:, t])

        return mat_B

    def roll_backward(self):
        """
        Roll backward and get the price and optimal hedge vals
        :return:
        """
        for t in range(self.num_steps - 1, -1, -1):
            pi_next = self.pi[:, t + 1]
            pi_next_prime = pi_next + self.tr_alpha * self.opt_hedge[:, t + 1] * self.s_values[:, t + 1]
            self.pi[:, t] = self.gamma * (pi_next_prime - self.opt_hedge[:, t] * self.delta_S[:, t])
            pi_prime_hat = pi_next_prime - np.mean(pi_next_prime)

            mat_A = self.function_A_vec(t, REG_PARAM)
            vec_B = self.function_B_vec(t, pi_prime_hat)

            phi = np.linalg.inv(mat_A) @ vec_B
            self.opt_hedge[:, t] = np.dot(self.data[t, :, :], phi)
            self.r[:, t] = self.gamma * self.opt_hedge[:, t] * self.delta_S[:, t] \
                           - self.risk_lambda * np.var(self.pi[:, t])


    def function_C_vec(self, t, reg_param):
        this_data = self.data[t, :, :]
        mat_C = this_data.T @ this_data
        return mat_C + reg_param * np.eye(this_data.shape[1])


    def function_D_vec(self, t, q_values, r_values):
        this_q = q_values[:, t + 1]
        this_r = r_values[:, t]
        mat_D = this_q @ (this_r + self.opt_hedge[:, t + 1] * self.s_values[:, t + 1])
        return mat_D


if __name__ == "__main__":
    trMC = TrQBlackScholes(0.02, 0.2, 100, 1, 0.04, 252, 1000, 0.001, 0.001)
    trMC.gen_path()
    trMC.roll_backward()
