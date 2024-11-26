import numpy as np
import scipy.sparse
from pywkm.wkm import wkm


class MatrixFunctionEvaluator:

    def wave_kernels(self, h, omega2, b):
        raise NotImplemented

    def wave_kernel_s(self, h, omega2, b):
        raise NotImplemented

    def wave_kernel_c(self, h, omega2, b):
        raise NotImplemented

    def wave_kernel_msinm(self, h, omega2, b):
        raise NotImplemented

    def reset(self) -> int:
        """Return information about the work having been done."""
        raise NotImplemented


class SymDiagonalizationEvaluator(MatrixFunctionEvaluator):
    def __init__(self):
        self.w, self.v = None, None
        self.diagonalizations = 0
        self.n = 0
        self.matvecs = 0

    @staticmethod
    def wave_kernel_c_scalar(h, w):
        return np.real(np.cos(h * np.sqrt(w)))

    @staticmethod
    def wave_kernel_s_scalar(h, w):
        return np.real(np.sinc(h * np.sqrt(w) / np.pi))

    @staticmethod
    def wave_kernel_xsin_scalar(h, w):
        return np.real(np.sqrt(w) * np.sin(h * np.sqrt(w)))

    def diagonalize(self, omega2):
        if self.w is None:
            self.w, self.v = scipy.linalg.eigh(omega2.todense())
            if min(self.w) < 0:
                self.w += 0j
            self.diagonalizations += 1
            self.n = omega2.shape[0]

    def wave_kernels(self, h, omega2, b):
        self.diagonalize(omega2)
        cosd = np.diag(self.wave_kernel_c_scalar(h, self.w))
        sincd = np.diag(self.wave_kernel_s_scalar(h, self.w))
        r = self.v.T @ b
        self.matvecs += 5
        return self.v @ (cosd @ r), self.v @ (sincd @ r)

    def wave_kernel_s(self, h, omega2, b):
        self.diagonalize(omega2)
        sincd = np.diag(self.wave_kernel_s_scalar(h, self.w))
        r = self.v.T @ b
        self.matvecs += 3
        return self.v @ (sincd @ r)

    def wave_kernel_c(self, h, omega2, b):
        self.diagonalize(omega2)
        cosd = np.diag(self.wave_kernel_c_scalar(h, self.w))
        r = self.v.T @ b
        self.matvecs += 3
        return self.v @ (cosd @ r)

    def wave_kernel_msinm(self, h, omega2, b):
        self.diagonalize(omega2)
        msind = np.diag(self.wave_kernel_xsin_scalar(h, self.w))
        r = self.v.T @ b
        self.matvecs += 3
        return self.v @ (msind @ r)

    def reset(self):
        self.w, self.v = None, None
        work = {f"({self.n}, {self.n}) diagonalizations": self.diagonalizations,
                self.n: self.matvecs}
        self.diagonalizations, self.n, self.matvecs = 0, 0, 0
        return work


class TridiagDiagonalizationEvaluator(SymDiagonalizationEvaluator):
    def diagonalize(self, omega2: scipy.sparse.csr_array):
        if self.w is None:
            self.w, self.v = scipy.linalg.eigh_tridiagonal(omega2.diagonal(0), omega2.diagonal(-1))
            if min(self.w) < 0:
                self.w += 0j


class WkmEvaluator(MatrixFunctionEvaluator):
    def __init__(self):
        self.C, self.S = None, None
        self.wkms = 0
        self.n = None
        self.matvecs = 0

    def wave_kernels(self, h, omega2, b):
        self.C, self.S = wkm(-1 * h ** 2 * omega2, return_sinhc=True)
        self.wkms += 1
        self.n = self.C.shape[0]
        return self.C @ b, self.S @ b

    def wave_kernel_s(self, h, omega2, b):
        self.C, self.S = wkm(-1 * h ** 2 * omega2, return_sinhc=True)
        self.wkms += 1
        self.n = self.C.shape[0]
        return self.S @ b

    def wave_kernel_c(self, h, omega2, b):
        self.C = wkm(-1 * h ** 2 * omega2, return_sinhc=False)
        self.wkms += 1
        self.n = self.C.shape[0]
        return self.C @ b

    def wave_kernel_msinm(self, h, omega2, b):
        if self.S is None:
            self.C, self.S = wkm(-1 * h ** 2 * omega2, return_sinhc=True)
            self.wkms += 1
            self.n = self.C.shape[0]
        self.matvecs += 2
        return h * omega2 @ (self.S @ b)

    def reset(self):
        self.C, self.S = None, None
        self.wkms = 0
        self.matvecs = 0
        return {f"({self.n}, {self.n}) wkms": self.wkms,  # TODO: Add this information to be returned by PyWKM
                self.n: self.matvecs}


class DenseWkmEvaluator(WkmEvaluator):
    """Dense evaluation of matrix functions.

    Dense evaluation is much faster. Our test cases are sparse by type only, thus this implementation saves us a good
    amount of time.
    """

    @staticmethod
    def densify(omega2):
        if scipy.sparse.issparse(omega2):
            return omega2.todense()
        else:
            return omega2

    def wave_kernels(self, h, omega2, b):
        self.C, self.S = wkm(-1 * h ** 2 * self.densify(omega2), return_sinhc=True)
        self.wkms += 1
        self.n = self.C.shape[0]
        return self.C @ b, self.S @ b

    def wave_kernel_s(self, h, omega2, b):
        self.C, self.S = wkm(-1 * h ** 2 * self.densify(omega2), return_sinhc=True)
        self.wkms += 1
        self.n = self.C.shape[0]
        return self.S @ b

    def wave_kernel_c(self, h, omega2, b):
        self.C = wkm(-1 * h ** 2 * self.densify(omega2), return_sinhc=False)
        self.wkms += 1
        self.n = self.C.shape[0]
        return self.C @ b

    def wave_kernel_msinm(self, h, omega2, b):
        if self.S is None:
            self.C, self.S = wkm(-1 * h ** 2 * self.densify(omega2), return_sinhc=True)
            self.wkms += 1
            self.n = self.C.shape[0]
        self.matvecs += 2
        return h * omega2 @ (self.S @ b)

    def reset(self):
        self.C, self.S = None, None
        self.wkms = 0
        self.matvecs = 0
        return {f"({self.n}, {self.n}) wkms": self.wkms,
                self.n: self.matvecs}
