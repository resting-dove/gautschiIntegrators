import numpy as np
import scipy

from .error_bounds import hochbruck_lubich
from .krylov_basis import arnoldi, extend_arnoldi


class LanczosProviderBase:
    def __init__(self):
        self.V, self.T = None, None
        self.v, self.w = None, None
        self.m = 0

    def construct(self, h, omega2, b, k, arnoldi_acc):
        # return m
        if self.V is None:
            self._construct(h, omega2, b, k, arnoldi_acc)
        return self.m

    def _construct(self, h, omega2, b, k, arnoldi_acc):
        raise NotImplementedError

    def construct_zero(self, n):
        zero = scipy.sparse.lil_array((n, 1))
        self.V = zero
        zero2 = np.array([[0]])
        self.v, self.T = zero2, zero2

    def diagonalize(self):
        # w, v = smth
        # return w, v
        raise NotImplementedError

    def reset(self):
        self.V, self.T = None, None
        self.v, self.w = None, None
        self.m = 0


class LanczosProvider(LanczosProviderBase):
    def _construct(self, h, omega2, b, k, arnoldi_acc):
        (w, V, T, breakdown) = arnoldi(A=h**2 * omega2, w=b, m=k, trunc=1, eps=arnoldi_acc)
        if breakdown:
            m = breakdown
        else:
            m = k
        # eta = T[m, m - 1]
        self.T = T[:m, :m]
        self.V = V
        self.m = m

    def diagonalize(self):
        if self.w is None:
            self.w, self.v = scipy.linalg.eigh_tridiagonal(np.diag(self.T), np.diag(self.T, -1))
            if min(self.w) < 0:
                self.w = self.w + 0j
            return 1
        else:
            return 0


class AdaptiveLanczosProviderHL(LanczosProvider):
    def _construct(self, h, omega2, b, k, arnoldi_acc):
        if k > 10:
            (w, V, T, breakdown) = arnoldi(A=h**2 * omega2, w=b, m=10, trunc=1, eps=arnoldi_acc)
            evals = scipy.linalg.eigvalsh_tridiagonal(np.diag(T), np.diag(T, -1)[:-1])
            k = hochbruck_lubich(-max(evals) / h, 1, k, arnoldi_acc)
            (w, V, T, breakdown) = extend_arnoldi(A=h**2 * omega2, V=V, w=w, H=T, s=10, m=k, trunc=-1)
        else:
            (w, V, T, breakdown) = arnoldi(A=h**2 * omega2, w=b, m=k, trunc=1, eps=arnoldi_acc)
        if breakdown:
            self.m = breakdown
        else:
            self.m = k
        print(f"m={self.m}")
        # eta = T[self.m, self.m - 1]
        self.T = T[: self.m, : self.m]
        self.V = V
