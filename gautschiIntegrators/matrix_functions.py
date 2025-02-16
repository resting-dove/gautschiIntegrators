import numpy as np
import scipy.sparse
from pywkm import wkm


class MatrixFunctionEvaluator:

    def wave_kernels(self, h, omega2, b):
        raise NotImplementedError

    def wave_kernel_s(self, h, omega2, b):
        raise NotImplementedError

    def wave_kernel_c(self, h, omega2, b):
        raise NotImplementedError

    def wave_kernel_msinm(self, h, omega2, b):
        raise NotImplementedError

    def reset(self) -> int:
        """Return information about the work having been done."""
        raise NotImplementedError


class SymDiagonalizationEvaluator(MatrixFunctionEvaluator):
    """Symmetric diagonalization-based evaluation of matrix functions."""

    def __init__(self):
        self.w, self.v = None, None
        self.diagonalizations = 0
        self.n = 0
        self.matvecs = 0
        self.h = None
        self.cosd, self.sincd, self.msind = None, None, None

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
                self.w = self.w + 0j
            self.diagonalizations += 1
            self.n = omega2.shape[0]

    def wave_kernels(self, h, omega2, b):
        if self.sincd is None or self.h != h:
            self.diagonalize(omega2)
            self.cosd = np.diag(self.wave_kernel_c_scalar(h, self.w))
            self.sincd = np.diag(self.wave_kernel_s_scalar(h, self.w))
            self.h = h
        r = self.v.T @ b
        self.matvecs += 5
        return self.v @ (self.cosd @ r), self.v @ (self.sincd @ r)

    def wave_kernel_s(self, h, omega2, b):
        if self.sincd is None or self.h != h:
            self.diagonalize(omega2)
            self.sincd = np.diag(self.wave_kernel_s_scalar(h, self.w))
            self.h = h
        r = self.v.T @ b
        self.matvecs += 3
        return self.v @ (self.sincd @ r)

    def wave_kernel_c(self, h, omega2, b):
        if self.cosd is None or self.h != h:
            self.diagonalize(omega2)
            self.cosd = np.diag(self.wave_kernel_c_scalar(h, self.w))
            self.h = h
        r = self.v.T @ b
        self.matvecs += 3
        return self.v @ (self.cosd @ r)

    def wave_kernel_msinm(self, h, omega2, b):
        if self.msind is None or self.h != h:
            self.diagonalize(omega2)
            self.msind = np.diag(self.wave_kernel_xsin_scalar(h, self.w))
            self.h = h
        r = self.v.T @ b
        self.matvecs += 3
        return self.v @ (self.msind @ r)

    def reset(self):
        self.w, self.v = None, None
        self.cosd, self.sincd, self.msind = None, None, None
        work = {f"({self.n}, {self.n}) diagonalizations": self.diagonalizations, self.n: self.matvecs}
        self.diagonalizations, self.n, self.matvecs = 0, 0, 0
        return work


class TridiagDiagonalizationEvaluator(SymDiagonalizationEvaluator):
    """Diagonalization-based evaluation of matrix functions, assuming the matrix is symmetric tridiagonal."""

    def diagonalize(self, omega2: scipy.sparse.csr_array):
        if self.w is None:
            self.w, self.v = scipy.linalg.eigh_tridiagonal(omega2.diagonal(0), omega2.diagonal(-1))
            if min(self.w) < 0:
                self.w = self.w + 0j
            self.diagonalizations += 1
            self.n = omega2.shape[0]

    def reset(self):
        self.w, self.v = None, None
        self.cosd, self.sincd, self.msind = None, None, None
        work = {f"({self.n}, {self.n}) diagonalizations": self.diagonalizations, self.n: self.matvecs}
        self.diagonalizations, self.n, self.matvecs = 0, 0, 0
        return work


class WkmEvaluator(MatrixFunctionEvaluator):
    """Evaluation of matrix functions using the evaluation of the Wave-Kernel matrix functions.

    The Wave-Kernel matrix functions are evaluated using the PyWKM package, which is not currently publicly available.
    """

    def __init__(self):
        self.C, self.S = None, None
        self.wkms = 0
        self.n = None
        self.h = None
        self.matvecs = 0

    def wave_kernels(self, h, omega2, b):
        if self.S is None or self.h != h:
            self.C, self.S = wkm(-1 * h**2 * omega2, return_sinhc=True)
            self.wkms += 1
            self.n = self.C.shape[0]
            self.h = h
        return self.C @ b, self.S @ b

    def wave_kernel_s(self, h, omega2, b):
        if self.S is None or self.h != h:
            self.C, self.S = wkm(-1 * h**2 * omega2, return_sinhc=True)
            self.wkms += 1
            self.n = self.C.shape[0]
            self.h = h
        return self.S @ b

    def wave_kernel_c(self, h, omega2, b):
        if self.C is None or self.h != h:
            self.C = wkm(-1 * h**2 * omega2, return_sinhc=False)
            self.wkms += 1
            self.n = self.C.shape[0]
            self.h = h
        return self.C @ b

    def wave_kernel_msinm(self, h, omega2, b):
        if self.S is None or self.h != h:
            self.C, self.S = wkm(-1 * h**2 * omega2, return_sinhc=True)
            self.wkms += 1
            self.n = self.C.shape[0]
            self.h = h
        self.matvecs += 2
        return h * omega2 @ (self.S @ b)

    def reset(self):
        self.C, self.S = None, None
        ret = {
            f"({self.n}, {self.n}) wkms": self.wkms,  # TODO: Add this information to be returned by PyWKM
            self.n: self.matvecs,
        }
        self.wkms = 0
        self.matvecs = 0
        return ret


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
        if self.S is None or h != self.h:
            self.C, self.S = wkm(-1 * h**2 * self.densify(omega2), return_sinhc=True)
            self.h = h
        self.wkms += 1
        self.n = self.C.shape[0]
        return self.C @ b, self.S @ b

    def wave_kernel_s(self, h, omega2, b):
        if self.S is None or h != self.h:
            self.C, self.S = wkm(-1 * h**2 * self.densify(omega2), return_sinhc=True)
            self.h = h
        self.wkms += 1
        self.n = self.C.shape[0]
        return self.S @ b

    def wave_kernel_c(self, h, omega2, b):
        if self.C is None or h != self.h:
            self.C = wkm(-1 * h**2 * self.densify(omega2), return_sinhc=False)
            self.h = h
        self.wkms += 1
        self.n = self.C.shape[0]
        return self.C @ b

    def wave_kernel_msinm(self, h, omega2, b):
        if self.S is None or h != self.h:
            self.C, self.S = wkm(-1 * h**2 * self.densify(omega2), return_sinhc=True)
            self.wkms += 1
            self.n = self.C.shape[0]
            self.h = h
        self.matvecs += 2
        return h * omega2 @ (self.S @ b)

    def reset(self):
        self.C, self.S = None, None
        ret = {f"({self.n}, {self.n}) wkms": self.wkms, self.n: self.matvecs}
        self.wkms = 0
        self.matvecs = 0
        return ret
