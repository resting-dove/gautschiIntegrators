import numpy as np
import scipy

from gautschiIntegrators.base import WorkLog
from gautschiIntegrators.lanczos.LanczosProvider import LanczosProvider
from gautschiIntegrators.matrix_functions import MatrixFunctionEvaluator


class LanczosDiagonalizationEvaluator(MatrixFunctionEvaluator):
    def __init__(self, krylov_size, arnoldi_acc=1e-10, lanczos=LanczosProvider()):
        self.k = krylov_size
        self.arnoldi_acc = arnoldi_acc
        self.lanczos = lanczos
        self.n = 0
        self.m = 0
        self.beta = None
        self.work = WorkLog()

    def calculate_lanczos(self, h, omega2, b):
        if self.beta is None:
            self.beta = scipy.linalg.norm(b)
        if self.beta == 0:
            self.lanczos.construct_zero(omega2.shape[0])
        else:
            self.m = self.lanczos.construct(h, omega2, b / self.beta, self.k, self.arnoldi_acc)
            self.work.add({omega2.shape[0]: self.m})

    def calculate_fTe1(self, f):
        if self.beta != 0:
            self.work.add({f"({self.m}, {self.m}) diagonalizations": self.lanczos.diagonalize()})
            self.work.add({self.m: 2})
            return self.lanczos.v @ (np.diag(f(self.lanczos.w)) @ self.lanczos.v.T[:, 0])
        else:
            return self.lanczos.v[:, 0]

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

    def wave_kernels(self, h, omega2, b):
        self.calculate_lanczos(h, omega2, b)
        cosT = self.calculate_fTe1(self.wave_kernel_c_scalar)
        sincT = self.calculate_fTe1(self.wave_kernel_s_scalar)
        # f = self.V @ self.v @ cosd @ self.v.T[:, 0]  # @ e1
        self.work.add({self.lanczos.V.shape[0]: 2})
        return self.beta * self.lanczos.V @ cosT, self.beta * self.lanczos.V @ sincT

    def wave_kernel_s(self, h, omega2, b):
        self.calculate_lanczos(h, omega2, b)
        sincT = self.calculate_fTe1(self.wave_kernel_s_scalar)
        self.work.add({self.lanczos.V.shape[0]: 2})
        return self.beta * self.lanczos.V @ sincT

    def wave_kernel_c(self, h, omega2, b):
        self.calculate_lanczos(h, omega2, b)
        cosT = self.calculate_fTe1(self.wave_kernel_c_scalar)
        self.work.add({self.lanczos.V.shape[0]: 2})
        return self.beta * self.lanczos.V @ cosT

    def wave_kernel_msinm(self, h, omega2, b):
        self.calculate_lanczos(h, omega2, b)
        sincT = self.calculate_fTe1(self.wave_kernel_xsin_scalar) / h
        self.work.add({self.lanczos.V.shape[0]: 2})
        return self.beta * self.lanczos.V @ sincT

    def reset(self):
        self.lanczos.reset()
        self.n = 0
        self.m = 0
        self.beta = None
        return self.work.store
