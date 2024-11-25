from collections import defaultdict

import numpy as np


class Solver:
    """Base Class for integrators for second order differential equations of the form
           x'' = - \Omega^2 @ x + g(x).

    This class is based on Scipy's OdeSolver base class.

    """

    def __init__(self, h: float, t_end: float, x0: np.array, v0: np.array, g: callable, *args, **kwargs):
        self.h = h
        self.t_end = t_end
        self.x = x0
        if v0 is None:
            self.v = np.zeros_like(self.x)
        else:
            self.v = v0
        if g is None:
            g = zero_g
        self.g = g

        self.t = 0
        self.status = 'running'
        self.n = self.x.size

        self.iterations = 0
        self.n_matvecs = defaultdict(lambda: 0)

    def step(self, omega2):
        """Perform one integration step.

        Returns
        -------
        message : string or None
            Report from the solver. Typically a reason for a failure if
            `self.status` is 'failed' after the step was taken or None
            otherwise.
        """
        if self.status != 'running':
            raise RuntimeError("Attempt to step on a failed or finished "
                               "solver.")

        if self.n == 0 or self.t == self.t_end:
            # Handle corner cases of empty solver or no integration.
            self.t = self.t_end
            message = None
            self.status = 'finished'
        else:
            t = self.t
            success, message = self._step_impl(omega2)

            if not success:
                self.status = 'failed'
            else:
                if self.t >= self.t_end:
                    self.status = 'finished'

        return message

    def _step_impl(self, omega2):
        raise NotImplementedError


def zero_g(x):
    return 0 * x
