"""
Multi-Frequency numerical example Fermi-Pasta-Ulam-Tsingou example from
D. Cohen, E. Hairer, and Ch. Lubich, “Numerical Energy Conservation for Multi-Frequency Oscillatory Differential
Equations,” Bit Numer Math, vol. 45, no. 2, pp. 287–305, Jun. 2005, doi: 10.1007/s10543-005-7121-z.

The trigonometric integrators should conserve the total Oscillatory Energy I and the smooth energy K over long times
using time steps that are large compared to the smallest non-zero frequency.

Resonant frequencies exchange energy but the sum should also be conserved, e.g. I1 + I3.

"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

from gautschiIntegrators.integrate import solve_ivp
from gautschiIntegrators.matrix_functions import DenseWkmEvaluator, TridiagDiagonalizationEvaluator

n = 5
omega = 70
eps = 1 / omega
lambdas = np.array([0, 1, 1, np.sqrt(2), 2])
omegas = lambdas / eps

Omega2 = np.diag(omegas ** 2)


def calcHyx(y, x):
    assert y.shape == x.shape
    assert x.shape[0] == 1
    return 1 / 2 * calcIyx(y, x).sum() + U(x)


def calcKyx(y, x):
    return 1 / 2 * np.linalg.norm(y[0, 0]) ** 2 + U(x)


def U(x):
    assert x.shape[0] == 1
    factor = np.ones_like(x)
    factor[0, 0] = 0.001
    return np.sum(factor * x) ** 4


def dU(u):
    if not u.shape[0] == 1:
        x = u.reshape((1, -1))
    else:
        x = u
    factor = np.ones_like(x)
    factor[0, 0] = 0.001
    term = 4 * np.sum(factor * x) ** 3
    return -(factor * term).flatten()


def calcIyx(y, x):
    return 1 / 2 * (y[0, :] ** 2 + (omegas * x[0, :]) ** 2)


def get_scipy_result(y, x, t_end):
    n = y.size
    m = x.shape[1]
    X = np.concatenate([x.flatten(), y.flatten()])

    def deriv(t, y):
        return np.concatenate((y[n:], -1 * Omega2 @ y[:n]
                               + dU(y[:n].reshape((1, -1)))
                               ))

    scipy_result = scipy.integrate.solve_ivp(deriv, [0, t_end], X,
                                             # t_eval=np.linspace(0, t_end, 1000),
                                             method='BDF')
    return scipy_result


def calcconstantsscipy(scipy_result):
    X = scipy_result["y"]
    sx = X[:n, :]
    sx = sx.reshape((1, n, -1))
    sy = X[n:, :]
    sy = sy.reshape((1, n, -1))
    Is = np.array([calcIyx(sy[:, :, i], sx[:, :, i]) for i in range(sx.shape[-1])])
    Hs = np.array([calcHyx(sy[:, :, i], sx[:, :, i]) for i in range(sx.shape[-1])])
    Ks = np.array([calcKyx(sy[:, :, i], sx[:, :, i]) for i in range(sx.shape[-1])])
    return Is, Hs, Ks


def calcconstantsgautschi(res):
    sx = res["xs"]
    sx = sx.reshape((1, n, -1))
    sy = res["vs"]
    sy = sy.reshape((1, n, -1))
    Is = np.array([calcIyx(sy[:, :, i], sx[:, :, i]) for i in range(sx.shape[-1])])
    Hs = np.array([calcHyx(sy[:, :, i], sx[:, :, i]) for i in range(sx.shape[-1])])
    Ks = np.array([calcKyx(sy[:, :, i], sx[:, :, i]) for i in range(sx.shape[-1])])
    return Is, Hs, Ks


def timescale1(y, x, h=0.0025, method="TwoStepF"):
    t_end = 20000
    # Using Scipy's solve_ivp is quite slow for this particular problem.
    # scipy_result = get_scipy_result(y, x, t_end=t_end)
    # Is, Hs, Ks = calcconstantsscipy(scipy_result)
    # mu = np.array([0, 1, 1, 0, 2])
    # I13 = np.nansum((mu * Is / lambdas), axis=1)
    # plt.plot(np.linspace(0, t_end, Is.shape[0]), Is[:, 1], label=f"{(lambdas[1]):.1f}w")
    # plt.plot(np.linspace(0, t_end, Is.shape[0]), Is[:, 2], label=f"{(lambdas[2]):.1f}w")
    # plt.plot(np.linspace(0, t_end, Is.shape[0]), Is[:, 3], label=f"{(lambdas[3]):.1f}w")
    # plt.plot(np.linspace(0, t_end, Is.shape[0]), Is[:, 4], label=f"{(lambdas[4]):.1f}w")
    # plt.plot(np.linspace(0, t_end, Is.shape[0]), I13, label="I13")
    # plt.plot(np.linspace(0, t_end, Is.shape[0]), Is.sum(-1), label="I")
    # plt.plot(np.linspace(0, t_end, Is.shape[0]), Hs, label="H", linestyle=":")
    # plt.legend(loc='center right')
    # plt.show()
    m = x.shape[1]
    res = solve_ivp(Omega2, dU, h, t_end, x.flatten(), y.flatten(), method,
                    evaluator=TridiagDiagonalizationEvaluator())
    Is, Hs, Ks = calcconstantsgautschi(res)
    mu = np.array([0, 1, 1, 0, 2])
    I13 = np.nansum((mu * Is / lambdas), axis=1)
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Is[:, 1], label=f"{(lambdas[1]):.1f}w")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Is[:, 2], label=f"{(lambdas[2]):.1f}w")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Is[:, 3], label=f"{(lambdas[3]):.1f}w")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Is[:, 4], label=f"{(lambdas[4]):.1f}w")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), I13, label="I13")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Is.sum(-1), label="I")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Hs, label="H", linestyle=":")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Ks, label="K", linestyle=":")
    plt.legend(loc='center right')
    plt.show()


if __name__ == "__main__":
    h = 0.0025
    x = np.random.normal(loc=0, scale=1, size=n).reshape((1, -1))
    x[0] = np.array([1, 0.3 * eps, 0.8 * eps, -1.1 * eps, 0.7 * eps])
    y = np.random.normal(loc=0, scale=1, size=n).reshape((1, -1))
    y[0] = np.array([-0.75, 0.6, 0.7, -0.9, 0.8])

    timescale1(y, x, 8 * eps, "OneStepF")
