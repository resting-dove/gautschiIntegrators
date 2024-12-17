import numpy as np
import scipy
from .base import Solver
from .matrix_functions import MatrixFunctionEvaluator, WkmEvaluator


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
        self.t += self.h
        if self.t >= self.t_end:
            overshoot = self.t - self.t_end
            self.t -= overshoot
            self.h -= overshoot

        X = np.concatenate([self.x, self.v])
        mat = scipy.sparse.block_array([[None, scipy.sparse.eye_array(*self.v.shape)], [-1 * omega2, None]])
        G = np.concatenate([np.zeros_like(self.x), self.g(self.x)])
        next_X = X + self.h * (mat @ X + G)
        self.x, self.v = next_X[:len(self.x)], next_X[len(self.x):]
        self.iterations += 1
        self.work[2 * self.n] += 1
        return True, None


class OneStepF(Solver):
    """
       One-step trigonometric integrator for second order differential equations of the form
           x'' = - \Omega^2 @ x + g(x).

       Source: Eq. (2.2) and configuration F of Table 1 of
           E. Hairer and C. Lubich, “Long-Time Energy Conservation of Numerical Methods for Oscillatory Differential
           Equations,” SIAM J. Numer. Anal., vol. 38, no. 2, pp. 414–441, Jul. 2000, doi: 10.1137/S0036142999353594.
       """

    def __init__(self, h: float, t_end: float, x0: np.array, v0: np.array, g: callable,
                 evaluator: MatrixFunctionEvaluator = WkmEvaluator(), **kwargs):
        super().__init__(h, t_end, x0, v0, g)
        self.evaluator = evaluator

    def _step_impl(self, omega2: scipy.sparse.sparray):
        self.t += self.h
        if self.t >= self.t_end:
            overshoot = self.t - self.t_end
            self.t -= overshoot
            self.h -= overshoot

        gn = self.g(self.x)
        cosm_xn = self.evaluator.wave_kernel_c(self.h, omega2, self.x)
        msinm_xn = self.evaluator.wave_kernel_msinm(self.h, omega2, self.x)
        self.work.add(self.evaluator.reset())

        cosm_vn, sincm_vn = self.evaluator.wave_kernels(self.h, omega2, self.v)
        self.work.add(self.evaluator.reset())

        sincm_gn = self.evaluator.wave_kernel_s(self.h, omega2, gn)
        self.work.add(self.evaluator.reset())

        cosm_sincm_gn, sincm2_gn = self.evaluator.wave_kernels(self.h, omega2, sincm_gn)
        self.work.add(self.evaluator.reset())

        x_1 = cosm_xn + self.h * sincm_vn + 0.5 * self.h ** 2 * (sincm2_gn)

        gn_1 = self.g(x_1)
        sincm_gn1 = self.evaluator.wave_kernel_s(self.h, omega2, gn_1)
        self.work.add(self.evaluator.reset())

        v_1 = - msinm_xn + cosm_vn + 0.5 * self.h * (cosm_sincm_gn + sincm_gn1)

        self.x, self.v = x_1, v_1
        self.iterations += 1
        return True, None


class OneStepGS99(Solver):
    """
       One-step trigonometric integrator for second order differential equations of the form
           x'' = - \Omega^2 @ x + g(x).

       Source:
            Originally proposed by
                B. García-Archilla, J. M. Sanz-Serna, and R. D. Skeel, “Long-Time-Step Methods for Oscillatory
                Differential Equations,” SIAM J. Sci. Comput., vol. 20, no. 3, pp. 930–963, Oct. 1998,
                doi: 10.1137/S1064827596313851.
            Also found in Eq. (2.2) and configuration E of Table 1 of
                E. Hairer and C. Lubich, “Long-Time Energy Conservation of Numerical Methods for Oscillatory
                Differential Equations,” SIAM J. Numer. Anal., vol. 38, no. 2, pp. 414–441, Jul. 2000,
                doi: 10.1137/S0036142999353594.
       """

    def __init__(self, h: float, t_end: float, x0: np.array, v0: np.array, g: callable,
                 evaluator: MatrixFunctionEvaluator = WkmEvaluator(), **kwargs):
        super().__init__(h, t_end, x0, v0, g)
        self.evaluator = evaluator

    def _step_impl(self, omega2: scipy.sparse.sparray):
        self.t += self.h
        if self.t >= self.t_end:
            overshoot = self.t - self.t_end
            self.t -= overshoot
            self.h -= overshoot

        cosm_xn, sincm_xn = self.evaluator.wave_kernels(self.h, omega2, self.x)
        msinm_xn = self.evaluator.wave_kernel_msinm(self.h, omega2, self.x)
        self.work.add(self.evaluator.reset())

        gn = self.g(sincm_xn)

        cosm_vn, sincm_vn = self.evaluator.wave_kernels(self.h, omega2, self.v)
        self.work.add(self.evaluator.reset())

        sincm_gn = self.evaluator.wave_kernel_s(self.h, omega2, gn)
        self.work.add(self.evaluator.reset())

        cosm_sincm_gn, sincm2_gn = self.evaluator.wave_kernels(self.h, omega2, sincm_gn)
        self.work.add(self.evaluator.reset())

        x_1 = cosm_xn + self.h * sincm_vn + 0.5 * self.h ** 2 * (sincm2_gn)

        sincm_x1 = self.evaluator.wave_kernel_s(self.h, omega2, x_1)
        self.work.add(self.evaluator.reset())
        g_sincm_x1 = self.g(sincm_x1)
        sincm_g_sincm_x1 = self.evaluator.wave_kernel_s(self.h, omega2, g_sincm_x1)
        self.work.add(self.evaluator.reset())

        v_1 = -1 * msinm_xn + cosm_vn + 0.5 * self.h * (cosm_sincm_gn + sincm_g_sincm_x1)

        self.x, self.v = x_1, v_1
        self.iterations += 1
        return True, None


class OneStep217(Solver):
    """
       One-step trigonometric integrator for second order differential equations of the form
           x'' = - \Omega^2 @ x + g(x).

       Source: Eq. (2.17) of
           E. Hairer and C. Lubich, “Long-Time Energy Conservation of Numerical Methods for Oscillatory Differential
           Equations,” SIAM J. Numer. Anal., vol. 38, no. 2, pp. 414–441, Jul. 2000, doi: 10.1137/S0036142999353594.
       """

    def __init__(self, h: float, t_end: float, x0: np.array, v0: np.array, g: callable,
                 evaluator: MatrixFunctionEvaluator = WkmEvaluator(), **kwargs):
        super().__init__(h, t_end, x0, v0, g)
        self.evaluator = evaluator

    def _step_impl(self, omega2: scipy.sparse.sparray):
        self.t += self.h
        if self.t >= self.t_end:
            overshoot = self.t - self.t_end
            self.t -= overshoot
            self.h -= overshoot

        gn = self.g(self.x)
        cosm_xn, sincm_xn = self.evaluator.wave_kernels(self.h, omega2, self.x)
        msinm_xn = self.evaluator.wave_kernel_msinm(self.h, omega2, self.x)
        self.work.add(self.evaluator.reset())

        cosm_vn, sincm_vn = self.evaluator.wave_kernels(self.h, omega2, self.v)
        self.work.add(self.evaluator.reset())

        sincm_gn = self.evaluator.wave_kernel_s(self.h, omega2, gn)
        self.work.add(self.evaluator.reset())

        g_sincm_xn = self.g(sincm_xn)
        gntilde = gn + sincm_gn - g_sincm_xn
        cosm_gntilde, sincm_gntilde = self.evaluator.wave_kernels(self.h, omega2, gntilde)
        self.work.add(self.evaluator.reset())

        x_1 = cosm_xn + self.h * sincm_vn + 0.5 * self.h ** 2 * sincm_gntilde

        gn_1 = self.g(x_1)
        sincm_gn1 = self.evaluator.wave_kernel_s(self.h, omega2, gn_1)
        self.work.add(self.evaluator.reset())

        sincm_x_1 = self.evaluator.wave_kernel_s(self.h, omega2, x_1)
        self.work.add(self.evaluator.reset())
        g_sincm_x_1 = self.g(sincm_x_1)
        g_1tilde = gn_1 + sincm_gn1 - g_sincm_x_1

        v_1 = - msinm_xn + cosm_vn + 0.5 * self.h * (cosm_gntilde + g_1tilde)

        self.x, self.v = x_1, v_1
        self.iterations += 1
        return True, None
