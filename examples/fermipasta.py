"""
A single frequency Fermi-Pasta-Ulam-Tsingou problem from
Section XIII.2.1 Time Scales in the Fermi–Pasta–Ulam Problem in
E. Hairer, G. Wanner, and C. Lubich, Geometric Numerical Integration, vol. 31. in Springer Series in Computational
Mathematics, vol. 31. Berlin/Heidelberg: Springer-Verlag, 2006. doi: 10.1007/3-540-30666-8.

This problem models the energy exchange between three stiff springs.

This script contains code to plot the dynamics of the system across three time-scales and compare the trigonometric
Gautschi-type integrators to the Scipy ODE solver for energy conservation and energy exchange between the springs.
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt

from gautschiIntegrators.integrate import solve_ivp
from gautschiIntegrators.matrix_functions import DenseWkmEvaluator, TridiagDiagonalizationEvaluator
from gautschiIntegrators.lanczos.LanczosEvaluator import RestartedLanczosWkmEvaluator, LanczosWkmEvaluator


def calcHpq(p, q, omega):
    assert p.shape == q.shape
    term1, term2 = 0, 0
    term3 = (q[0] - 0) ** 4
    for i in range(1, p.shape[0] // 2):
        term1 += p[2 * i - 2] ** 2 + p[2 * i - 1] ** 2
        term2 += (q[2 * i - 1] - q[2 * i - 2]) ** 2
        term3 += (q[2 * (i - 1)] - q[2 * (i - 1) - 1]) ** 4
    term3 += (1 - q[-1]) ** 4
    return 1 / 2 * term1 + omega ** 2 / 4 * term2 + term3


def calcHyx(y, x, omega):
    assert y.shape == x.shape
    assert x.shape[0] == 2
    term1, term2 = 0, 0
    for i in range(1, x.shape[1]):
        term1 += y[0, i - 1] ** 2 + y[1, i - 1] ** 2
        term2 += (x[1, i - 1]) ** 2
    return 1 / 2 * term1 + omega ** 2 / 2 * term2 + U(x)


def calcHmat(y, x, omega):
    # x = u, v; y = u', v'
    n = y.shape[1]
    term1 = 1 / 2 * np.dot(y.flatten(), y.flatten())
    Omega = np.zeros((x.size, x.size))
    Omega[n:, n:] = omega ** 2 * np.eye(n)
    term2 = 1 / 2 * x.flatten() @ Omega @ x.flatten()
    return term1 + term2 + U(x)


def U(x):
    assert x.shape[0] == 2
    ex = np.pad(x, ((0, 0), (1, 1)), 'constant', constant_values=(0, 0))
    return ((ex[0, 1:] - ex[0, :-1] - ex[1, 1:] - ex[1, :-1]) ** 4).sum() / 4


# A linear term g.
# def U(x):
#     assert x.shape[0] == 2
#     n = x.size
#     A = np.diag(np.arange(1, n + 1) / 3)
#     return 1 / 2 * x.flatten() @ A @ x.flatten()
#
#
# def dU(u):
#     if u.shape[0] == 2:
#         x = u.flatten()
#     else:
#         x = u
#     n = x.size
#     A = np.diag(np.arange(1, n + 1) / 3)
#     return - A @ x


def dU(u):
    if not u.shape[0] == 2:
        x = u.reshape((2, -1))
    else:
        x = u
    ex = np.pad(x, ((0, 0), (1, 1)), 'constant', constant_values=(0, 0))
    term = (ex[0, 1:] - ex[0, :-1] - ex[1, 1:] - ex[1, :-1]) ** 3
    du = term[:-1] - term[1:]
    dv = -term[:-1] - term[1:]
    return -np.array([du, dv]).flatten()


def calcTs(y, x):
    return np.linalg.norm(y[0]) ** 2 / 2, np.linalg.norm(y[1]) ** 2 / 2


def calcIxy(y, x, omega):
    return 1 / 2 * (y[1, :] ** 2 + omega ** 2 * x[1, :] ** 2)


def get_scipy_result(y, x, omega, t_end):
    n = y.size
    m = x.shape[1]
    X = np.concatenate([x.flatten(), y.flatten()])

    Omega2 = np.zeros((x.size, x.size))
    Omega2[m:, m:] = omega ** 2 * np.eye(m)

    def deriv(t, y):
        return np.concatenate((y[n:], -1 * Omega2 @ y[:n]
                               + dU(y[:n].reshape((2, -1)))
                               ))

    scipy_result = scipy.integrate.solve_ivp(deriv, [0, t_end], X,
                                             # t_eval=np.linspace(0, t_end, 1000),
                                             method='BDF')
    return scipy_result


def calcconstantsscipy(scipy_result):
    X = scipy_result["y"]
    sx = X[:6, :]
    sx = sx.reshape((2, 3, -1))
    sy = X[6:, :]
    sy = sy.reshape((2, 3, -1))
    Is = np.array([calcIxy(sy[:, :, i], sx[:, :, i], omega) for i in range(sx.shape[-1])])
    Ts = np.array([calcTs(sy[:, :, i], sx[:, :, i]) for i in range(sx.shape[-1])])
    Hs = np.array([calcHyx(sy[:, :, i], sx[:, :, i], omega) for i in range(sx.shape[-1])]) - 0.8
    return Is, Ts, Hs


def calcconstantsgautschi(res):
    sx = res["xs"]
    sx = sx.reshape((2, 3, -1))
    sy = res["vs"]
    sy = sy.reshape((2, 3, -1))
    Is = np.array([calcIxy(sy[:, :, i], sx[:, :, i], omega) for i in range(sx.shape[-1])])
    Ts = np.array([calcTs(sy[:, :, i], sx[:, :, i]) for i in range(sx.shape[-1])])
    Hs = np.array([calcHyx(sy[:, :, i], sx[:, :, i], omega) for i in range(sx.shape[-1])]) - 0.8
    return Is, Ts, Hs


def timescale1(y, x, omega, h=0.0025, method="TwoStepF"):
    t_end = 1
    scipy_result = get_scipy_result(y, x, omega, t_end=t_end)
    Is, Ts, Hs = calcconstantsscipy(scipy_result)
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Is.sum(-1), label="I")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Ts[:, 1], label="T1")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Hs, label="H")
    plt.legend()
    plt.show()
    m = x.shape[1]
    Omega2 = np.zeros((x.size, x.size))
    Omega2[m:, m:] = omega ** 2 * np.eye(m)
    res = solve_ivp(Omega2, dU, h, t_end, x.flatten(), y.flatten(), method,
                    evaluator=LanczosWkmEvaluator(krylov_size=2))
    Is, Ts, Hs = calcconstantsgautschi(res)
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Is.sum(-1), label="I")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Ts[:, 0], label="T0")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Ts[:, 1], label="T1")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Hs, label="H")
    plt.legend()
    plt.show()


def timescale2(y, x, omega, h=0.0025, method="TwoStepF"):
    t_end = 3
    scipy_result = get_scipy_result(y, x, omega, t_end=t_end)
    Is, Ts, Hs = calcconstantsscipy(scipy_result)
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Is.sum(-1), label="I")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Ts[:, 0], label="T0")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Ts[:, 1], label="T1")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Hs, label="H")
    plt.legend()
    plt.show()

    m = x.shape[1]
    Omega2 = np.zeros((x.size, x.size))
    Omega2[m:, m:] = omega ** 2 * np.eye(m)
    res = solve_ivp(Omega2, dU, h, t_end, x.flatten(), y.flatten(), method,
                    evaluator=DenseWkmEvaluator())
    Is, Ts, Hs = calcconstantsgautschi(res)
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Is.sum(-1), label="I")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Ts[:, 0], label="T0")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Ts[:, 1], label="T1")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Hs, label="H")
    plt.legend()
    plt.show()


def timescale3(y, x, omega, h=0.0025, method="TwoStepF"):
    t_end = 200
    scipy_result = get_scipy_result(y, x, omega, t_end=t_end)
    Is, Ts, Hs = calcconstantsscipy(scipy_result)
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Is.sum(-1), label="I")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Is[:, 0], label="I1")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Is[:, 1], label="I2")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Is[:, 2], label="I3")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Hs, label="H")
    plt.legend()
    plt.show()

    m = x.shape[1]
    Omega2 = np.zeros((x.size, x.size))
    Omega2[m:, m:] = omega ** 2 * np.eye(m)
    res = solve_ivp(Omega2, dU, h, t_end, x.flatten(), y.flatten(), method,
                    evaluator=DenseWkmEvaluator())
    Is, Ts, Hs = calcconstantsgautschi(res)
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Is.sum(-1), label="I")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Is[:, 0], label="I1")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Is[:, 1], label="I2")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Is[:, 2], label="I3")
    plt.plot(np.linspace(0, t_end, Is.shape[0]), Hs, label="H")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    h = 0.0025
    omega = 50
    q = np.random.normal(loc=0, scale=1, size=6)
    p = np.zeros_like(q)
    eq = np.pad(q, ((1, 1)), 'constant', constant_values=(0, 0))
    ep = np.pad(p, ((1, 1)), 'constant', constant_values=(0, 0))

    x = np.array([(eq[2::2] + eq[1:-1:2]) / np.sqrt(2),
                  (eq[2::2] - eq[1:-1:2]) / np.sqrt(2)])
    y = np.array([(ep[2::2] + ep[1:-1:2]) / np.sqrt(2),
                  (ep[2::2] - ep[1:-1:2]) / np.sqrt(2)])

    x = np.zeros_like(x)
    y = np.zeros_like(y)
    x[:, 0] = np.array([1, 1 / omega])
    y[:, 0] = np.array([1, 1])

    ex = np.pad(x, ((0, 0), (1, 1)), 'constant', constant_values=(0, 0))

    timescale1(y, x, omega, h, "OneStepF")
    # timescale2(y, x, omega, h*10)
    # timescale3(y, x, omega, 2 / omega, "OneStepF")
