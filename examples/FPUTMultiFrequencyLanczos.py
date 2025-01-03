import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
from gautschiIntegrators.integrate import solve_ivp
from gautschiIntegrators.matrix_functions import DenseWkmEvaluator, TridiagDiagonalizationEvaluator
from gautschiIntegrators.lanczos.LanczosEvaluator import LanczosWkmEvaluator, AdaptiveRestartedLanczosWkmEvaluator

n = 200
omega = 70
eps = 1 / omega
lambdas = np.array([0] + [1] + list(np.linspace(1, 3, n - 4)) + [np.sqrt(2), 2 * np.sqrt(2)])
omegas = lambdas / eps

Omega2 = np.diag(omegas ** 2)


def calcHyx(y, x):
    assert y.shape == x.shape
    assert x.shape[0] == 1
    return calcIyx(y, x).sum() + U(x)


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


def calcconstantsgautschi(res):
    sx = res["xs"]
    sx = sx.reshape((1, n, -1))
    sy = res["vs"]
    sy = sy.reshape((1, n, -1))
    Is = np.array([calcIyx(sy[:, :, i], sx[:, :, i]) for i in range(sx.shape[-1])])
    Hs = np.array([calcHyx(sy[:, :, i], sx[:, :, i]) for i in range(sx.shape[-1])])
    Ks = np.array([calcKyx(sy[:, :, i], sx[:, :, i]) for i in range(sx.shape[-1])])
    return Is, Hs, Ks


def timescale1(y, x, evaluator, h=0.0025, method="TwoStepF", filename="fputmultifreqlanczos"):
    t_end = 1000
    t_curr = 0
    Isums, Hs, Ks = [], [], []
    I11, I12 = [], []
    Isqrt2, I2sqrt2 = [], []
    xx, yy = x, y
    while t_curr < t_end:
        print(t_curr)
        start = time.time()
        res = solve_ivp(Omega2, dU, h, 200, xx.flatten(), yy.flatten(), method,
                        evaluator=evaluator)
        print(f"{len(res['ts'])} steps took {time.time() - start}")
        xx, yy = res["x"], res["v"]
        t_curr += res["t"]
        Is_, Hs_, Ks_ = calcconstantsgautschi(res)
        Hs.extend(Hs_)
        Ks.extend(Ks_)
        I11.extend(Is_[:, 1])
        I12.extend(Is_[:, 2])
        Isqrt2.extend(Is_[:, -2])
        I2sqrt2.extend(Is_[:, -1])
        Isums.extend(Is_.sum(-1))
    plot_store = {"ts": res["ts"],
                  "xs": res["xs"],
                  "vs": res["vs"],
                  "work": str(res["work"]),
                  "Isums": Isums, "Hs": Hs, "Ks": Ks, "lambdas": lambdas, "eps": eps,
                  "I11": I11, "I12": I12, "Isqrt2": Isqrt2, "I2sqrt2": I2sqrt2, }
    np.savez(filename, **plot_store)
    Isqrt2s = (1 / lambdas[-2] * np.array(Isqrt2)) + (2 / lambdas[-1] * np.array(I2sqrt2))
    plt.plot(np.linspace(0, t_end, len(Isums)), Isums, label="I")
    plt.plot(np.linspace(0, t_end, len(I2sqrt2)), I2sqrt2, label=f"{(lambdas[-2]):.1f}w")
    plt.plot(np.linspace(0, t_end, len(Isqrt2)), Isqrt2, label=f"{(lambdas[-1]):.1f}w")
    plt.plot(np.linspace(0, t_end, len(Isqrt2s)), Isqrt2s, label="Isqrt2s")

    plt.plot(np.linspace(0, t_end, len(I11)), I11, label=f"{(lambdas[1]):.1f}w")
    plt.plot(np.linspace(0, t_end, len(I12)), I12, label=f"{(lambdas[2]):.1f}w")

    plt.plot(np.linspace(0, t_end, len(Hs[::10])), Hs[::10], label="H", linestyle=":")
    plt.plot(np.linspace(0, t_end, len(Ks[::10])), Ks[::10], label="K")
    plt.legend(loc='center right')
    plt.savefig(filename + ".png")
    plt.show()


if __name__ == "__main__":
    rng = np.random.default_rng(33)
    x = rng.normal(loc=0, scale=1, size=n).reshape((1, -1))
    x[0, 1:] *= eps
    x[0, 0] = 1
    y = rng.normal(loc=0, scale=1, size=n).reshape((1, -1))

    y = y / calcIyx(y, x)

    timescale1(y, x, TridiagDiagonalizationEvaluator(),
               2 * eps, "OneStepGS99", filename=f"OneStepGS99")
    timescale1(y, x, AdaptiveRestartedLanczosWkmEvaluator(krylov_size=4, max_restarts=10, arnoldi_acc=1e-20),
               2 * eps, "OneStepGS99", filename=f"OneStepGS99_adaptive")
