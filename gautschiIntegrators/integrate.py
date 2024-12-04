import inspect

import numpy as np

from gautschiIntegrators.base import Solver
from gautschiIntegrators.one_step import OneStepF, ExplicitEuler, OneStepGS99
from gautschiIntegrators.two_step import TwoStepIntegratorF, TwoStepIntegrator2_16

METHODS = {
    "OneStepF": OneStepF,
    "TwoStepF": TwoStepIntegratorF,
    "TwoStep216": TwoStepIntegrator2_16,
    "ExplicitEuler": ExplicitEuler,
    "OneStepGS99": OneStepGS99,
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
    ts = [0]
    xs = []
    vs = []
    status = None
    while status is None:
        message = solver.step(A)

        if solver.status == 'finished':
            status = 0
        elif solver.status == 'failed':
            status = -1
            break

        ts.append(solver.t)
        xs.append(solver.x)
        vs.append(solver.v)

    message = MESSAGES.get(status, message)

    return {
        "t": solver.t,
        "x": solver.x,
        "v": solver.v,
        "ts": ts,
        "xs": np.stack(xs, axis=-1),
        "vs": np.stack(vs, axis=-1),
        "iterations": solver.iterations,
        "work": solver.work,
        "status": status, "message": message, "success": status >= 0
    }
