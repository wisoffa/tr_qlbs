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


