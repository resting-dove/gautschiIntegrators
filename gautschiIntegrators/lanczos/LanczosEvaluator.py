import numpy as np
import scipy

from ..base import WorkLog
from .ArnoldiProvider import RestartedLanczosProvider, DenseRestartedLanczosProvider
from .LanczosProvider import LanczosProvider
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


class RestartedLanczosWkmEvaluator:

    def __init__(self, krylov_size, arnoldi_acc=1e-10, max_restarts=10):
        self.k = krylov_size
        self.arnoldi_acc = arnoldi_acc
        self.rlanczos = DenseRestartedLanczosProvider()
        self.n = 0
        self.m = 0
        self.max_restarts = max_restarts
        self.beta = None
        self.work = WorkLog()
        self.cosO, self.sincO = None, None

    def calculate_lanczos(self, h, omega2, b):
        if self.beta is None:
            self.beta = scipy.linalg.norm(b)
        if self.beta == 0:
            self.rlanczos.construct_zero(omega2.shape[0])
        else:
            self.m = self.rlanczos.construct(h, omega2, b / self.beta, self.k, self.arnoldi_acc)
            self.work.add({omega2.shape[0]: self.m})

    def reset(self):
        self.rlanczos.reset()
        self.n = 0
        self.m = 0
        self.beta = None
        ret = self.work.store
        self.work = WorkLog()
        self.cosO, self.sincO = None, None
        return ret

    def wave_kernels(self, h, omega2, b):
        self.calculate_lanczos(h, omega2, b)
        C, S = wkm(-1 * self.rlanczos.T, return_sinhc=True)
        self.n = C.shape[0]
        self.work.add({f"({self.n}, {self.n}) wkms": 1})
        cosT = C[:, [0]]
        sincT = S[:, [0]]
        self.work.add({self.rlanczos.V.shape[0]: 2})
        self.cosO, self.sincO = self.beta * self.rlanczos.V @ cosT, self.beta * self.rlanczos.V @ sincT
        if self.beta != 0:
            for k in range(1, self.max_restarts + 1):
                self.rlanczos.add_restart(h, omega2, b / self.beta, self.k, self.arnoldi_acc)
                C, S = wkm(-1 * self.rlanczos.T, return_sinhc=True)
                self.work.add({f"({self.rlanczos.km}, {self.rlanczos.km}) wkms": 1})
                cosT = C[-self.rlanczos.m:, [0]]
                sincT = S[-self.rlanczos.m:, [0]]
                self.work.add({self.rlanczos.V.shape[0]: 2})
                cUpdate, sUpdate = self.beta * self.rlanczos.V @ cosT, self.beta * self.rlanczos.V @ sincT
                self.cosO, self.sincO = self.cosO + cUpdate, self.sincO + sUpdate

        return self.cosO.flatten(), self.sincO.flatten()

    def wave_kernel_s(self, h, omega2, b):
        return self.wave_kernels(h, omega2, b)[1]

    def wave_kernel_c(self, h, omega2, b):
        return self.wave_kernels(h, omega2, b)[0]

    def wave_kernel_msinm(self, h, omega2, b):
        if self.sincO is None:
            self.wave_kernels(h, omega2, b)
        self.work.add({self.rlanczos.V.shape[0]: 2})
        return h * omega2 @ self.sincO.flatten()


class RestartedLanczosDiagonalizationEvaluator:

    def __init__(self, krylov_size, arnoldi_acc=1e-10, max_restarts=10):
        self.k = krylov_size
        self.arnoldi_acc = arnoldi_acc
        self.rlanczos = DenseRestartedLanczosProvider()
        self.n = 0
        self.m = 0
        self.max_restarts = max_restarts
        self.beta = None
        self.work = WorkLog()
        self.sincO, self.cosO = None, None

    def calculate_lanczos(self, h, omega2, b):
        if self.beta is None:
            self.beta = scipy.linalg.norm(b)
        if self.beta == 0:
            self.rlanczos.construct_zero(omega2.shape[0])
        else:
            self.m = self.rlanczos.construct(h, omega2, b / self.beta, self.k, self.arnoldi_acc)
            self.work.add({omega2.shape[0]: self.m})

    def reset(self):
        self.rlanczos.reset()
        self.n = 0
        self.m = 0
        self.beta = None
        ret = self.work.store
        self.work = WorkLog()
        self.sincO, self.cosO = None, None
        return ret

    def wave_kernels(self, h, omega2, b):
        self.calculate_lanczos(h, omega2, b)
        cosT = self.calculate_fTe1(self.wave_kernel_c_scalar)
        sincT = self.calculate_fTe1(self.wave_kernel_s_scalar)
        self.n = cosT.shape[0]
        self.work.add({self.rlanczos.V.shape[0]: 2})
        self.cosO, self.sincO = self.beta * self.rlanczos.V @ cosT, self.beta * self.rlanczos.V @ sincT
        if self.beta != 0:
            for k in range(1, self.max_restarts + 1):
                self.rlanczos.add_restart(h, omega2, b / self.beta, self.k, self.arnoldi_acc)
                cosT = self.calculate_fTe1(self.wave_kernel_c_scalar)[-self.rlanczos.m:]
                sincT = self.calculate_fTe1(self.wave_kernel_s_scalar)[-self.rlanczos.m:]
                self.work.add({self.rlanczos.V.shape[0]: 2})
                cUpdate, sUpdate = self.beta * self.rlanczos.V @ cosT, self.beta * self.rlanczos.V @ sincT
                self.cosO, self.sincO = self.cosO + cUpdate, self.sincO + sUpdate
        return self.cosO.flatten(), self.sincO.flatten()

    def wave_kernel_s(self, h, omega2, b):
        if self.sincO is not None:
            return self.sincO
        else:
            return self.wave_kernels(h, omega2, b)[1]

    def wave_kernel_c(self, h, omega2, b):
        if self.cosO is not None:
            return self.cosO
        else:
            return self.wave_kernels(h, omega2, b)[0]

    def wave_kernel_msinm(self, h, omega2, b):
        self.calculate_lanczos(h, omega2, b)
        tsinT = self.calculate_fTe1(self.wave_kernel_xsin_scalar) / h
        self.n = tsinT.shape[0]
        self.work.add({self.rlanczos.V.shape[0]: 2})
        self.OsinO = self.beta * self.rlanczos.V @ tsinT
        if self.beta != 0:
            for k in range(1, self.max_restarts + 1):
                self.rlanczos.add_restart(h, omega2, b / self.beta, self.k, self.arnoldi_acc)
                tsinT = self.calculate_fTe1(self.wave_kernel_xsin_scalar) / h
                self.work.add({self.rlanczos.V.shape[0]: 2})
                OsUpdate = self.beta * self.rlanczos.V @ tsinT
                self.OsinO = self.OsinO + OsUpdate
        return self.OsinO.flatten()

    def calculate_fTe1(self, f):
        if self.beta != 0:
            self.work.add({f"({self.m}, {self.m}) diagonalizations": self.rlanczos.diagonalize()})
            self.work.add({self.m: 2})
            return self.rlanczos.v @ (np.diag(f(self.rlanczos.w)) @ self.rlanczos.v.T[:, [0]]).flatten()
        else:
            return self.rlanczos.v[:, 0]

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
def get_eigcond(A):
    _, vl, vr = scipy.linalg.eig(A, left=True)
    return np.reciprocal(np.sum(vl * vr, axis=0))