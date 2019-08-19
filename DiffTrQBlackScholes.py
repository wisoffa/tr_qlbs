"""
Author : Jiwoo Park
Date   : 2019-08-14
Desc   : Valuing Option Price under transaction cost. Re-hedging using difference method.
"""
from .RoundTrQBlackScholes import RoundTrQBlackScholes, REG_PARAM
import numpy as np


class DiffTrQBlackScholes(RoundTrQBlackScholes):
    """
    Transaction Cost occurs in consecutive difference of optimal hedge
    """
    def roll_backward_hedge(self):
        for t in range(self.num_steps - 1, -1, -1):
            pi_next = self.pi[:, t + 1]
            pi_next_prime = pi_next + self.tr_alpha * self.opt_hedge[:, t + 1] * self.s_values[:, t + 1]
            pi_prime_hat = pi_next_prime - np.mean(pi_next_prime)

            mat_A = self.function_A_vec(t, REG_PARAM)
            vec_B = self.function_B_vec(t, pi_prime_hat)

            phi = np.linalg.inv(mat_A) @ vec_B
            # TODO
            # 1. calculate 2 different hedge
            # 2. compare 2 with Q values
            # 3. pick hedge value to make lower Q mean values
            # 4. update pi, q and r using optimal hedge
            #
            self.opt_hedge[:, t] = np.dot(self.data[t, :, :], phi)
            self.pi[:, t] = self.gamma * (pi_next_prime - self.opt_hedge[:, t] * self.delta_S[:, t])
            self.r[:, t] = self.gamma * self.opt_hedge[:, t] * self.delta_S[:, t] \
                           - self.risk_lambda * np.var(self.pi[:, t])


    def roll_backward_q(self):
        pass

