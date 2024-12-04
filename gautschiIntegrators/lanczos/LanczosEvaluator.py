import numpy as np
import scipy

from gautschiIntegrators.base import WorkLog
from gautschiIntegrators.lanczos.LanczosProvider import LanczosProvider
from pywkm.wkm import wkm


class LanczosEvaluatorBase:
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

    def reset(self):
        self.lanczos.reset()
        self.n = 0
        self.m = 0
        self.beta = None
        ret = self.work.store
        self.work = WorkLog()
        return ret


class LanczosDiagonalizationEvaluator(LanczosEvaluatorBase):

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


class LanczosWkmEvaluator(LanczosEvaluatorBase):
    def __init__(self, krylov_size, arnoldi_acc=1e-10, lanczos=LanczosProvider()):
        super().__init__(krylov_size, arnoldi_acc, lanczos)
        self.wkms = 0

    def wave_kernels(self, h, omega2, b):
        self.calculate_lanczos(h, omega2, b)
        self.C, self.S = wkm(-1 * self.lanczos.T, return_sinhc=True)
        self.wkms += 1
        self.n = self.C.shape[0]
        cosT = self.C[:, 0]
        sincT = self.S[:, 0]
        self.work.add({self.lanczos.V.shape[0]: 2})
        return self.beta * self.lanczos.V @ cosT, self.beta * self.lanczos.V @ sincT

    def wave_kernel_s(self, h, omega2, b):
        self.calculate_lanczos(h, omega2, b)
        self.C, self.S = wkm(-1 * self.lanczos.T, return_sinhc=True)
        self.wkms += 1
        self.n = self.C.shape[0]
        sincT = self.S[:, 0]
        self.work.add({self.lanczos.V.shape[0]: 1})
        return self.beta * self.lanczos.V @ sincT

    def wave_kernel_c(self, h, omega2, b):
        self.calculate_lanczos(h, omega2, b)
        self.C = wkm(-1 * self.lanczos.T, return_sinhc=False)
        self.wkms += 1
        self.n = self.C.shape[0]
        cosT = self.C[:, 0]
        self.work.add({self.lanczos.V.shape[0]: 1})
        return self.beta * self.lanczos.V @ cosT

    def wave_kernel_msinm(self, h, omega2, b):
        self.calculate_lanczos(h, omega2, b)
        self.C, self.S = wkm(-1 * self.lanczos.T, return_sinhc=True)
        self.wkms += 1
        self.n = self.C.shape[0]
        sincT = self.S[:, 0]
        self.work.add({self.lanczos.V.shape[0]: 2})
        return self.beta * h * omega2 @ (self.lanczos.V @ sincT)

    def reset(self):
        self.lanczos.reset()
        self.n = 0
        self.m = 0
        self.beta = None
        self.C, self.S = None, None
        self.work.add({f"({self.n}, {self.n}) wkms": self.wkms})
        ret = self.work.store
        self.wkms = 0
        self.work = WorkLog()
        return ret
