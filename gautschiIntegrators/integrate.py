import inspect

import numpy as np
import scipy.sparse

from gautschiIntegrators.base import Solver
from gautschiIntegrators.one_step import OneStepF, ExplicitEuler
from gautschiIntegrators.two_step import TwoStepIntegratorF, TwoStepIntegrator2_16
from gautschiIntegrators.matrix_functions import sym_sincm_sqrt, sym_cosm_sqrt, sym_msinm_sqrt


def integrate_two_step_f_sym(omega2: scipy.sparse.sparray, g: callable, x0: np.array, h: float, t: float, v0=None, *,
                             cosm=sym_cosm_sqrt, sincm=sym_sincm_sqrt):
    """
    Integrate second order differential equation of the form $x'' = -\Omega^2 x + g(x)$ with a two-step method.

    The integration is discretized in time and occurs in steps of h up to the final time t. The initial velocities are
    set to 0 if not specified explicitly.

    Parameters
    ----------
    omega2 : scipy.sparse.sparse.spmatrix
        Matrix $\Omega^2$
    g : callable
    x0 : array_like
    h, t : float
        The integration step size and final desired time.
    v0 : float, optional
        The initial velocities. Set to 0 if not specified.
    cosm : callable, default=sym_cosm_sqrt
    sincm : callable, default=sym_sincm_sqrt

    Returns
    -------
    array_like
        x values at final time t.

    Notes:
    Source: Eq. (2.5) employing method F from Table 1 of
        E. Hairer and C. Lubich, “Long-Time Energy Conservation of Numerical Methods for Oscillatory Differential
        Equations,” SIAM J. Numer. Anal., vol. 38, no. 2, pp. 414–441, Jul. 2000, doi: 10.1137/S0036142999353594.
    """
    if v0 is None:
        v0 = np.zeros_like(x0)
    assert t > 2 * h
    # x, v = ExplicitEuler(h / 2).step(omega2, x0, v0, g)
    # x, v = ExplicitEuler(h / 2).step(omega2, x, v, g)
    x, v = get_scipy_result(omega2, np.concatenate([x0, v0]), g, h)
    t -= h
    integrator = TwoStepIntegratorF(h, x0, cosm=cosm, sincm=sincm, g=g)
    while t > 0:  # TODO: This doesn't just work when it's not a clean division
        h = min(h, t)
        integrator.set_h(h)
        x = integrator.step(omega2, x)
        t -= h
    return x


def integrate_two_step_216_sym(omega2: scipy.sparse.sparray, g: callable, x0: np.array, h: float, t: float, v0=None, *,
                               cosm=sym_cosm_sqrt, sincm=sym_sincm_sqrt):
    """
    Integrate second order differential equation of the form $x'' = -\Omega^2 x + g(x)$ with a two-step method.

    The integration is discretized in time and occurs in steps of h up to the final time t. The initial velocities are
    set to 0 if not specified explicitly.

    Parameters
    ----------
    omega2 : scipy.sparse.sparse.spmatrix
        Matrix $\Omega^2$
    g : callable
    x0 : array_like
    h, t : float
        The integration step size and final desired time.
    v0 : float, optional
        The initial velocities. Set to 0 if not specified.
    cosm : callable, default=sym_cosm_sqrt
    sincm : callable, default=sym_sincm_sqrt

    Returns
    -------
    array_like
        x values at final time t.

    Notes:
    Source: Eq. (2.16) of
        E. Hairer and C. Lubich, “Long-Time Energy Conservation of Numerical Methods for Oscillatory Differential
        Equations,” SIAM J. Numer. Anal., vol. 38, no. 2, pp. 414–441, Jul. 2000, doi: 10.1137/S0036142999353594.
    """
    if v0 is None:
        v0 = np.zeros_like(x0)
    assert t > 2 * h
    # x, v = ExplicitEuler(h / 2).step(omega2, x0, v0, g)
    # x, v = ExplicitEuler(h / 2).step(omega2, x, v, g)
    x, v = get_scipy_result(omega2, np.concatenate([x0, v0]), g, h)
    t -= h
    integrator = TwoStepIntegrator2_16(h, x0, cosm=cosm, sincm=sincm, g=g)
    while t > 0:  # TODO: This doesn't just work when it's not a clean division
        h = min(h, t)
        integrator.set_h(h)
        x = integrator.step(omega2, x)
        t -= h
    return x


def integrate_one_step_f_sym(omega2: scipy.sparse.sparray, g: callable, x0: np.array, h: float, t: float, v0=None, *,
                             cosm=sym_cosm_sqrt, sincm=sym_sincm_sqrt, msinm=sym_msinm_sqrt):
    """
    Integrate second order differential equation of the form $x'' = -\Omega^2 x + g(x)$ with a one-step method.

    The integration is discretized in time and occurs in steps of h up to the final time t. The initial velocities are
    set to 0 if not specified explicitly.

    Parameters
    ----------
    omega2 : scipy.sparse.sparse.spmatrix
        Matrix $\Omega^2$
    g : callable
    x0 : array_like
    h, t : float
        The integration step size and final desired time.
    v0 : float, optional
        The initial velocities. Set to 0 if not specified.
    cosm : callable, default=sym_cosm_sqrt
    sincm : callable, default=sym_sincm_sqrt
    msinm : callable, default=sym_msinm_sqrt

    Returns
    -------
    array_like
        x values at final time t.
    array_like
        x' values at final time t.

    Notes:
    Source: Eq. (2.2) & (2.3) employing method F from Table 1 of
        E. Hairer and C. Lubich, “Long-Time Energy Conservation of Numerical Methods for Oscillatory Differential
        Equations,” SIAM J. Numer. Anal., vol. 38, no. 2, pp. 414–441, Jul. 2000, doi: 10.1137/S0036142999353594.
    """
    if v0 is None:
        v0 = np.zeros_like(x0)
    assert t > 2 * h
    x, v = x0, v0
    integrator = OneStepF(h, cosm=cosm, sincm=sincm, msinm=msinm, g=g)
    while t > 0:  # TODO: This doesn't just work when it's not a clean division
        h = min(h, t)
        integrator.set_h(h)
        x, v = integrator.step(omega2, x, v)
        t -= h
    return x, v


def get_scipy_result(A, X, g, t_end):
    # TODO: Replace this with a one step method
    n = A.shape[0]

    def deriv(t, y):
        return np.concatenate((y[n:], -1 * A @ y[:n] + g(y[:n])))

    scipy_result = scipy.integrate.solve_ivp(deriv, [0, t_end], X)
    return scipy_result["y"][:n, -1], scipy_result["y"][n:, -1]


METHODS = {
    "OneStepF": OneStepF,
    "TwoStepF": TwoStepIntegratorF,
    "TwoStep216": TwoStepIntegrator2_16,
    "ExplicitEuler": ExplicitEuler,
}

MESSAGES = {0: "The solver successfully reached the end of the integration interval.",
            1: "A termination event occurred."}


def solve_ivp(A, g, h: float, t_end: float, x0: np.array, v0: np.array, method: str, **options):
    """Solve an initial value problem given by a second order differential equations of the form
           x'' = - \Omega^2 @ x + g(x).

        This function is based on Scipy's solve_ivp.
        """
    if method not in METHODS and not (
            inspect.isclass(method) and issubclass(method, Solver)):
        raise ValueError(f"`method` must be one of {METHODS} or Solver class.")

    if method in METHODS:
        method: Solver = METHODS[method]

    solver = method(h, t_end, x0, v0, g, **options)

    t, x = None, None
    status = None
    while status is None:
        message = solver.step(A)

        if solver.status == 'finished':
            status = 0
        elif solver.status == 'failed':
            status = -1
            break
        t = solver.t
        x = solver.x

    message = MESSAGES.get(status, message)

    return {
        "t": t,
        "x": x,
        "v": solver.v,
        "iterations": solver.iterations,
        "n_matvecs": solver.n_matvecs,
        "status": status, "message": message, "success": status >= 0
    }
