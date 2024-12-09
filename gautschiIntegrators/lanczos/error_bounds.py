import numpy as np

def hochbruck_lubich(sm_ev: float, t: float, k_max: int, desired_acc: float):
    """
    A priori error bound for the Arnoldi approximation of exp(tA),
    where the spectrum of A is entirely in the left half plane.

    Source:  Theorem 2 of  M. Hochbruck and C. Lubich,
    “On Krylov Subspace Approximations to the Matrix Exponential Operator,”
    SIAM J. Numer. Anal., vol. 34, no. 5, pp. 1911–1925, Oct. 1997, doi: 10.1137/S0036142995280572.

    Note: 2-norm."""
    rho = sm_ev / -4

    ms = np.arange(1, k_max + 1)
    out = np.empty_like(ms) * np.nan
    out = np.where(ms >= 2 * rho * t, 10 / (rho * t) / np.exp(rho * t) * (np.exp(1) * rho * t / ms) ** ms, out)
    indices = (2 * rho * t >= ms) * (ms >= np.sqrt(4 * rho * t))
    out = np.where(indices, 10 / np.exp(ms ** 2 / (5 * rho * t)), out)
    if out[-1] < desired_acc:
        k = np.argmax(out < desired_acc)
    else:
        k = k_max
    return k