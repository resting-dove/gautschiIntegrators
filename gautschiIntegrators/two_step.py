import abc

import numpy as np
import scipy.sparse


class TwoStepIntegratorF():
    """
    Two-step trigonometric integrator for second order differential equations of the form
        x'' = - \Omega^2 @ x + g(x).

    Source: Eq. (2.5) employing method F from Table 1 of
        E. Hairer and C. Lubich, “Long-Time Energy Conservation of Numerical Methods for Oscillatory Differential
        Equations,” SIAM J. Numer. Anal., vol. 38, no. 2, pp. 414–441, Jul. 2000, doi: 10.1137/S0036142999353594.
    """

    def __init__(self, h: float, x0: np.array, *, cosm: callable, sincm: callable, g: callable):
        self.prev_x = x0
        self.h = h
        self.cosm = cosm  # cosm(h, A, b) = cos(h * sqrt(A)) @ b
        self.sincm = sincm  # sincm(h, A, b) = sinc(h * sqrt(A)) @ b
        self.g = g

    def set_h(self, h):
        self.h = h

    def step(self, omega2: scipy.sparse.sparray, x: np.array):
        g = self.g(x)
        cosm = self.cosm(self.h, omega2, x)
        sincm = self.sincm(self.h, omega2, g)
        sincm2 = self.sincm(self.h, omega2, sincm)

        result = 2 * cosm - self.prev_x + self.h ** 2 * sincm2
        self.prev_x = x
        return result
