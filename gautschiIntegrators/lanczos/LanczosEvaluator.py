import numpy as np
import scipy

from pywkm.EvalMatPolyPS import EvalMatPolyPS
from pywkm.get_PadeOrder_Scaling import get_PadeOrder_Scaling
from pywkm.get_WaveKernels_diag_Pade_coeffs import get_WaveKernels_diag_Pade_coeffs
from ..base import WorkLog
from .ArnoldiProvider import RestartedLanczosProvider, DenseRestartedLanczosProvider
from .LanczosProvider import LanczosProvider
from pywkm.wkm import wkm, calculate_C_S_dense, calculate_C_dense


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
        ret = self.work.store
        self.lanczos.reset()
        self.n = 0
        self.m = 0
        self.beta = None
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
        self.work.add({f"({self.n}, {self.n}) wkms": self.wkms})
        ret = self.work.store
        self.lanczos.reset()
        self.n = 0
        self.m = 0
        self.beta = None
        self.C, self.S = None, None
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
        self.P, self.Q, self.dPxQ_PxdQ = None, None, None
        self.wkm_m, self.wkm_s = None, None

    def calculate_lanczos(self, h, omega2, b):
        if self.beta is None:
            self.beta = scipy.linalg.norm(b)
        if self.beta == 0:
            self.rlanczos.construct_zero(omega2.shape[0])
        else:
            self.m = self.rlanczos.construct(h, omega2, b / self.beta, self.k, self.arnoldi_acc)
            self.work.add({omega2.shape[0]: self.rlanczos.km})  # Matvecs for the Lanczos basis

    def reset(self):
        ret = self.work.store
        self.rlanczos.reset()
        self.n = 0
        self.m = 0
        self.beta = None
        self.work = WorkLog()
        self.cosO, self.sincO = None, None
        # TODO: Check if this is valid or maybe introduce a whole new abstraction like reset and reset_new_b, where the
        #  latter only causes changes for Lanczos based methods.
        # self.P, self.Q, self.dPxQ_PxdQ = None, None, None
        # self.wkm_m, self.wkm_s = None, None
        return ret

    def wave_kernels(self, h, omega2, b):
        self.calculate_lanczos(h, omega2, b)
        C, S = wkm(-1 * self.rlanczos.T, return_sinhc=True)
        self.n = C.shape[0]
        self.work.add({f"{C.shape} wkms": 1})
        cosT = C[:, [0]]
        sincT = S[:, [0]]
        self.work.add({self.rlanczos.V.shape[0]: 2})
        self.cosO, self.sincO = self.beta * self.rlanczos.V @ cosT, self.beta * self.rlanczos.V @ sincT
        if self.beta != 0:
            for k in range(1, self.max_restarts + 1):
                self.rlanczos.add_restart(h, omega2, b / self.beta, self.k, self.arnoldi_acc)
                C, S = wkm(-1 * self.rlanczos.T, return_sinhc=True)
                self.work.add({f"{C.shape} wkms": 1})
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


class AdaptiveRestartedLanczosWkmEvaluator(RestartedLanczosWkmEvaluator):
    def wkm(self, A, return_sinhc=True):
        if self.wkm_m is None:
            self.wkm_m, self.wkm_s, As = get_PadeOrder_Scaling(A)
            P, Q, dPxQ_PxdQ = get_WaveKernels_diag_Pade_coeffs(self.wkm_m)
            self.P = np.array([1] + list(P))
            self.Q = np.array([1] + list(Q))
            self.dPxQ_PxdQ = np.array([1] + list(dPxQ_PxdQ.flatten()))
            for ii in range(self.wkm_m - 1):
                As[ii] = As[ii] / (4 ** ((ii + 1) * self.wkm_s))
        else:
            As = [A]
            for ii in range(self.wkm_m - 1):
                As.append(A @ As[-1])
                As[ii] = As[ii] / (4 ** ((ii + 1) * self.wkm_s))
            As[self.wkm_m - 1] = As[self.wkm_m - 1] / (4 ** ((self.wkm_m) * self.wkm_s))
        pA = EvalMatPolyPS(self.P, As)
        qA = EvalMatPolyPS(self.Q, As)
        self.work.add({f"{A.shape} wkms": 1})
        if return_sinhc:
            return calculate_C_S_dense(As, self.dPxQ_PxdQ, qA, pA, self.wkm_s)
        else:
            return calculate_C_dense(As, qA, pA, self.wkm_s)
    def wave_kernels(self, h, omega2, b):
        self.calculate_lanczos(h, omega2, b)
        C, S = self.wkm(-1 * self.rlanczos.T, return_sinhc=True)
        self.n = C.shape[0]
        cosT = C[:, [0]]
        sincT = S[:, [0]]
        self.work.add({self.rlanczos.V.shape[0]: 2})
        self.cosO, self.sincO = self.beta * self.rlanczos.V @ cosT, self.beta * self.rlanczos.V @ sincT
        stopping_criterion = False
        k = 1
        if self.beta != 0:
            while k <= self.max_restarts and not stopping_criterion:
                self.rlanczos.add_restart(h, omega2, b / self.beta, self.k, self.arnoldi_acc)
                C, S = self.wkm(-1 * self.rlanczos.T, return_sinhc=True)
                cosT = C[-self.rlanczos.m:, [0]]
                sincT = S[-self.rlanczos.m:, [0]]
                self.work.add({self.rlanczos.V.shape[0]: 2})
                cUpdate, sUpdate = self.beta * self.rlanczos.V @ cosT, self.beta * self.rlanczos.V @ sincT
                self.cosO, self.sincO = self.cosO + cUpdate, self.sincO + sUpdate
                if scipy.linalg.norm(cUpdate) < self.arnoldi_acc and scipy.linalg.norm(sUpdate) < self.arnoldi_acc:
                    stopping_criterion = True
                k += 1
        return self.cosO.flatten(), self.sincO.flatten()


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
        ret = self.work.store
        self.rlanczos.reset()
        self.n = 0
        self.m = 0
        self.beta = None
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
