import numpy as np
import scipy
from gautschiIntegrators.base import Solver


class VelocityVerlet:
    def __init__(self, h: float):
        self.prev_force = None
        self.h = h

    def advance_positions(self, x: np.array, v: np.array, force: np.ndarray) -> np.array:
        """Part one of the Velocity - Verlet scheme.

        Parameters
        ----------
        x, v : array_like
            Positions and velocities of the particles. Length is 3n.
        force : array_like
            Forces acting on the particles, already divided by the mass term. Length is 3n.
        """
        next_x = x + self.h * v + 0.5 * self.h ** 2 * force
        self.prev_force = force
        return next_x

    def advance_velocities(self, v: np.array, force: np.ndarray) -> np.array:
        """Part two of the Velocity - Verlet scheme.

        Parameters
        ----------
        v : array_like
            Positions and velocities of the particles. Length is 3n.
        force : array_like
            Forces acting on the particles, already divided by the mass term. Length is 3n.
        """
        next_v = v + 0.5 * self.h * (self.prev_force + force)
        return next_v

    def step(self, x: np.array, v: np.array, force: np.array) -> (np.array, np.array):
        """Perform one step of the Velocity - Verlet scheme.

        Parameters
        ----------
        x, v : array_like
            Positions and velocities of the particles. Length is 3n.
        force : array_like
            Forces acting on the particles, already divided by the mass term. Length is 3n.
        """
        next_x = self.advance_positions(x, v, force)
        next_v = self.advance_velocities(v, force)
        return next_x, next_v


class ExplicitEuler(Solver):
    """
    Solve second order differential equation $x'' = - \Omega^2 x + g(x)$ by transforming into first order ODE and
    applying the explicit Euler method.
    """

    def __init__(self, h: float, t_end: float, x0: np.array, v0: np.array, g: callable, *args, **kwargs):
        super().__init__(h, t_end, x0, v0, g)

    def _step_impl(self, omega2: scipy.sparse.sparray) -> (np.array, np.array):
        X = np.concatenate([self.x, self.v])
        mat = scipy.sparse.block_array([[None, scipy.sparse.eye_array(*self.v.shape)], [-1 * omega2, None]])
        G = np.concatenate([np.zeros_like(self.x), self.g(self.x)])
        next_X = X + self.h * (mat @ X + G)
        self.x, self.v = next_X[:len(self.x)], next_X[len(self.x):]
        self.t += self.h
        self.iterations += 1
        self.n_matvecs[2 * self.n] += 1
        return True, None


class OneStepF(Solver):
    """
       One-step trigonometric integrator for second order differential equations of the form
           x'' = - \Omega^2 @ x + g(x).

       Source: Eq. (2.2) and configuration F of Table 1 of
           E. Hairer and C. Lubich, “Long-Time Energy Conservation of Numerical Methods for Oscillatory Differential
           Equations,” SIAM J. Numer. Anal., vol. 38, no. 2, pp. 414–441, Jul. 2000, doi: 10.1137/S0036142999353594.
       """

    def __init__(self, h: float, t_end: float, x0: np.array, v0: np.array, g: callable, *args, cosm: callable,
                 sincm: callable,
                 msinm=None, **kwargs):
        super().__init__(h, t_end, x0, v0, g)
        self.cosm = cosm  # cosm(h, A, b) = cos(h * sqrt(A)) @ b
        self.sincm = sincm  # sincm(h, A, b) = sinc(h * sqrt(A)) @ b
        # sqrt(A) @ sin(h * sqrt(A)) @ b = h * A @ sinc(h * sqrt(A))
        if msinm is None:
            self.msinm = lambda h, A, x: h * A @ self.sincm(h, A, x)
        else:
            self.msinm = msinm

    def _step_impl(self, omega2: scipy.sparse.sparray):
        gn = self.g(self.x)
        cosm_xn = self.cosm(self.h, omega2, self.x)
        msinm_xn = self.msinm(self.h, omega2, self.x)
        sincm_vn = self.sincm(self.h, omega2, self.v)
        cosm_vn = self.cosm(self.h, omega2, self.v)

        sincm_gn = self.sincm(self.h, omega2, gn)
        sincm2_gn = self.sincm(self.h, omega2, sincm_gn)
        cosm_sincm_gn = self.cosm(self.h, omega2, sincm_gn)

        x_1 = cosm_xn + self.h * sincm_vn + 0.5 * self.h ** 2 * (sincm2_gn)

        gn_1 = self.g(x_1)
        sincm_gn1 = self.sincm(self.h, omega2, gn_1)

        v_1 = - msinm_xn + cosm_vn + 0.5 * self.h * (cosm_sincm_gn + sincm_gn1)

        self.x, self.v = x_1, v_1
        self.t += self.h
        self.iterations += 1
        self.n_matvecs[self.n] += np.inf  # TODO: Make the function evaluator interface return this
        return True, None
