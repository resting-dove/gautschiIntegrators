import numpy as np
import scipy

from gautschiIntegrators.base import WorkLog
from gautschiIntegrators.lanczos.krylov_basis import arnoldi
from gautschiIntegrators.matrix_functions import MatrixFunctionEvaluator


class LanczosDiagonalizationEvaluator(MatrixFunctionEvaluator):
    def __init__(self, krylov_size, arnoldi_acc=1e-10):
        self.k = krylov_size
        self.arnoldi_acc = arnoldi_acc
        self.V, self.T = None, None
        self.v, self.w = None, None
        self.n = 0
        self.m = 0
        self.beta = None
        self.work = WorkLog()

    def calculate_lanczos(self, h, omega2, b):
        if self.beta is None:
            self.beta = scipy.linalg.norm(b)
        if self.beta == 0:
            zero = scipy.sparse.lil_array((omega2.shape[0], 1))
            self.V = zero
            zero2 = np.array([[0]])
            self.v, self.T = zero2, zero2
        elif self.V is None:  # or not np.all(self.V[:, 0] == b / self.beta):
            (w, V, T, breakdown) = arnoldi(A=h ** 2 * omega2, w=b / self.beta, m=self.k, trunc=1, eps=self.arnoldi_acc)
            if breakdown:
                stopping_criterion = True
                self.m = breakdown
            else:
                self.m = self.k
            eta = T[self.m, self.m - 1]
            self.T = T[:self.m, :self.m]
            self.V = V

            self.work.add({omega2.shape[0]: self.m})
        else:
            # We already calculated the Lanczos decomposition
            pass

    def calculate_fTe1(self, f):
        if self.beta != 0:
            self.diagonalize()
            self.work.add({self.m: 2})
            return self.v @ (np.diag(f(self.w)) @ self.v.T[:, 0])
        else:
            return self.v[:, 0]

    @staticmethod
    def wave_kernel_c_scalar(w):
        return np.real(np.cos(np.sqrt(w)))

    @staticmethod
    def wave_kernel_s_scalar(w):
        return np.real(np.sinc(np.sqrt(w) / np.pi))

    @staticmethod
    def wave_kernel_xsin_scalar(w):
        # Pay attention to the h!
        return np.real(np.sqrt(w) * np.sin(np.sqrt(w)))

    def diagonalize(self):
        if self.w is None:
            self.w, self.v = scipy.linalg.eigh_tridiagonal(np.diag(self.T), np.diag(self.T, -1))
            if min(self.w) < 0:
                self.w = self.w + 0j
            self.work.add({f"({self.m}, {self.m}) diagonalizations": 1})

    def wave_kernels(self, h, omega2, b):
        self.calculate_lanczos(h, omega2, b)
        cosT = self.calculate_fTe1(self.wave_kernel_c_scalar)
        sincT = self.calculate_fTe1(self.wave_kernel_s_scalar)
        # f = self.V @ self.v @ cosd @ self.v.T[:, 0]  # @ e1
        self.work.add({self.V.shape[0]: 2})
        return self.beta * self.V @ cosT, self.beta * self.V @ sincT

    def wave_kernel_s(self, h, omega2, b):
        self.calculate_lanczos(h, omega2, b)
        sincT = self.calculate_fTe1(self.wave_kernel_s_scalar)
        self.work.add({self.V.shape[0]: 2})
        return self.beta * self.V @ sincT

    def wave_kernel_c(self, h, omega2, b):
        self.calculate_lanczos(h, omega2, b)
        cosT = self.calculate_fTe1(self.wave_kernel_c_scalar)
        self.work.add({self.V.shape[0]: 2})
        return self.beta * self.V @ cosT

    def wave_kernel_msinm(self, h, omega2, b):
        self.calculate_lanczos(h, omega2, b)
        sincT = self.calculate_fTe1(self.wave_kernel_xsin_scalar) / h
        self.work.add({self.V.shape[0]: 2})
        return self.beta * self.V @ sincT

    def reset(self):
        self.V, self.T = None, None
        self.v, self.w = None, None
        self.n = 0
        self.m = 0
        self.beta = None
        return self.work.store
