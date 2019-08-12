"""
Author : Jiwoo Park
Date   : 2019. 8. 5
Desc   : Reference for Black Scholes
"""
import numpy as np

from scipy.stats import norm


def bs_put(t, S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 1/2 * sigma ** 2) * (T-t)) / sigma / np.sqrt(T-t)
    d2 = (np.log(S0/K) + (r - 1/2 * sigma ** 2) * (T-t)) / sigma / np.sqrt(T-t)
    price = K * np.exp(-r * (T-t)) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return price


def bs_call(t, s0, K, r, sigma, T):
    d1 = (np.log(s0/K) + (r + 1/2 * sigma ** 2) * (T-t)) / (sigma * np.sqrt(T-t))
    d2 = (np.log(s0/K) + (r - 1/2 * sigma ** 2) * (T-t)) / (sigma * np.sqrt(T-t))
    price = s0 * norm.cdf(d1) - K * np.exp(-r * (T-t)) * norm.cdf(d2)
    return price


if __name__ == "__main__":
    print(bs_call(0, 100, 120, 0.03, 0.1, 1))
