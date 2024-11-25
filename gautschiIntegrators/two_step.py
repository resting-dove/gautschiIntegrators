import numpy as np
import scipy.sparse
from gautschiIntegrators.base import Solver


class TwoStepIntegratorF(Solver):
    """
    Two-step trigonometric integrator for second order differential equations of the form
        x'' = - \Omega^2 @ x + g(x).

    Source: Eq. (2.5) employing method F from Table 1 of
        E. Hairer and C. Lubich, “Long-Time Energy Conservation of Numerical Methods for Oscillatory Differential
        Equations,” SIAM J. Numer. Anal., vol. 38, no. 2, pp. 414–441, Jul. 2000, doi: 10.1137/S0036142999353594.
    """

    def __init__(self, h: float, t_end: float, x0: np.array, v0: np.array, g: callable, cosm: callable, sincm: callable,
                 *args, **kwargs):
        super().__init__(h, t_end, x0, v0, g)
        self.prev_x = None
        self.cosm = cosm  # cosm(h, A, b) = cos(h * sqrt(A)) @ b
        self.sincm = sincm  # sincm(h, A, b) = sinc(h * sqrt(A)) @ b

    def first_step_impl(self, omega2: scipy.sparse.spmatrix):
        x, n_matvecs = get_scipy_result(omega2, np.concatenate([self.x, self.v]), self.g, self.h)
        self.prev_x = self.x
        self.x = x
        self.t += self.h
        self.iterations += 1
        self.n_matvecs[self.n] += n_matvecs
        return True, None

    def _step_impl(self, omega2: scipy.sparse.sparray):
        if self.t == 0:
            return self.first_step_impl(omega2)
        g_x = self.g(self.x)
        cosm = self.cosm(self.h, omega2, self.x)
        sincm = self.sincm(self.h, omega2, g_x)
        sincm2 = self.sincm(self.h, omega2, sincm)

        res = 2 * cosm - self.prev_x + self.h ** 2 * sincm2
        self.prev_x = self.x
        self.x = res
        self.t += self.h
        self.iterations += 1
        self.n_matvecs[self.n] += np.inf  # TODO: Make the function evaluator interface return this
        return True, None


class TwoStepIntegrator2_16(Solver):
    """
    Two-step trigonometric integrator for second order differential equations of the form
        x'' = - \Omega^2 @ x + g(x).

    Source: Eq. (2.16) of
        E. Hairer and C. Lubich, “Long-Time Energy Conservation of Numerical Methods for Oscillatory Differential
        Equations,” SIAM J. Numer. Anal., vol. 38, no. 2, pp. 414–441, Jul. 2000, doi: 10.1137/S0036142999353594.
    """

    def __init__(self, h: float, t_end: float, x0: np.array, v0: np.array, g: callable, cosm: callable, sincm: callable,
                 *args, **kwargs):
        super().__init__(h, t_end, x0, v0, g)
        self.prev_x = None
        self.cosm = cosm  # cosm(h, A, b) = cos(h * sqrt(A)) @ b
        self.sincm = sincm  # sincm(h, A, b) = sinc(h * sqrt(A)) @ b

    def first_step_impl(self, omega2: scipy.sparse.spmatrix):
        x, n_matvecs = get_scipy_result(omega2, np.concatenate([self.x, self.v]), self.g, self.h)
        self.prev_x = self.x
        self.x = x
        self.t += self.h
        self.iterations += 1
        self.n_matvecs[self.n] += n_matvecs
        return True, None

    def _step_impl(self, omega2: scipy.sparse.sparray):
        if self.t == 0:
            return self.first_step_impl(omega2)
        g_x = self.g(self.x)
        cosm_xn = self.cosm(self.h, omega2, self.x)
        sincm_xn = self.sincm(self.h, omega2, self.x)
        sincm_gn = self.sincm(self.h, omega2, g_x)
        sincm2_gn = self.sincm(self.h, omega2, sincm_gn)
        g_sincm_xn = self.g(sincm_xn)
        sincm_g_sincm_xn = self.sincm(self.h, omega2, g_sincm_xn)

        res = 2 * cosm_xn - self.prev_x + self.h ** 2 * (sincm2_gn + sincm_gn + sincm_g_sincm_xn)
        self.prev_x = self.x
        self.x = res
        self.t += self.h
        self.iterations += 1
        self.n_matvecs[self.n] += np.inf  # TODO: Make the function evaluator interface return this
        return True, None


def get_scipy_result(A, X, g, t_end):
    # TODO: Replace this with a one step method
    n = A.shape[0]

    def deriv(t, y):
        return np.concatenate((y[n:], -1 * A @ y[:n] + g(y[:n])))

    scipy_result = scipy.integrate.solve_ivp(deriv, [0, t_end], X)
    return scipy_result["y"][:n, -1], scipy_result["nfev"]
