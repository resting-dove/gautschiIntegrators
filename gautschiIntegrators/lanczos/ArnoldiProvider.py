import numpy as np
import scipy

from gautschiIntegrators.lanczos.krylov_basis import arnoldi


class ArnoldiProviderBase:
    def __init__(self):
        self.V = None
        self.T = None  # T instead of H for symmetry with the Lanczos Provider
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
        raise NotImplementedError

    def reset(self):
        self.V, self.T = None, None
        self.v, self.w = None, None
        self.m = 0


class RestartedLanczosProvider(ArnoldiProviderBase):

    def __init__(self):
        super().__init__()
        self.subdiag_array = None
        self.km = None
        self.w = None

    def _construct(self, h, omega2, b, krylov_size, arnoldi_acc):
        current_size = 0
        (v_next, V, H, breakdown) = arnoldi(A=h ** 2 * omega2, w=b, m=krylov_size, trunc=1,
                                            eps=arnoldi_acc)
        if breakdown:
            print("Breakdown")
            stopping_criterion = True
            m = breakdown
        else:
            m = krylov_size
        self.T = scipy.sparse.csc_array(H[:m, :m])
        eta = H[m, m - 1]
        self.subdiag_array = fill_block_in_top_right(eta, rows=m, cols=self.T.shape[1])
        current_size += m
        self.V = V
        self.v_next = v_next
        self.m = m
        self.km = current_size

    def add_restart(self, h, omega2, b, krylov_size, arnoldi_acc):
        (v_next, V, H, breakdown) = arnoldi(A=h ** 2 * omega2, w=self.v_next, m=krylov_size, trunc=1,
                                            eps=arnoldi_acc)
        if breakdown:
            print("Breakdown")
            m = breakdown
        else:
            m = krylov_size
        self.T = scipy.sparse.block_array(([self.T, None], [self.subdiag_array, scipy.sparse.csc_array(H[:m, :m])]),
                                          format="csc")
        eta = H[m, m - 1]
        self.subdiag_array = fill_block_in_top_right(eta, rows=m, cols=self.T.shape[1])
        self.km += m
        self.V = V
        self.v_next = v_next
        self.m = m

    def diagonalize(self):
        if self.w is None:
            self.w, self.v = scipy.linalg.eig(self.T.todense())
            if min(self.w) < 0:
                self.w = self.w + 0j
            return 1
        else:
            return 0

    def reset(self):
        self.V, self.T = None, None
        self.v, self.w = None, None
        self.m = 0
        self.km = None
        self.v_next = None


class DenseRestartedLanczosProvider(RestartedLanczosProvider):

    def _construct(self, h, omega2, b, krylov_size, arnoldi_acc):
        current_size = 0
        (v_next, V, H, breakdown) = arnoldi(A=h ** 2 * omega2, w=b, m=krylov_size, trunc=1,
                                            eps=arnoldi_acc)
        if breakdown:
            print("Breakdown")
            stopping_criterion = True
            m = breakdown
        else:
            m = krylov_size
        self.T = H[:m, :m]
        eta = H[m, m - 1]
        self.subdiag_array = fill_block_in_top_right(eta, rows=m, cols=self.T.shape[1])
        current_size += m
        self.V = V
        self.v_next = v_next
        self.m = m
        self.km = current_size

    def add_restart(self, h, omega2, b, krylov_size, arnoldi_acc):
        (v_next, V, H, breakdown) = arnoldi(A=h ** 2 * omega2, w=self.v_next, m=krylov_size, trunc=1,
                                            eps=arnoldi_acc)
        if breakdown:
            print("Breakdown")
            m = breakdown
        else:
            m = krylov_size
        self.T = scipy.sparse.block_array(([self.T, None], [self.subdiag_array, scipy.sparse.csc_array(H[:m, :m])]),
                                          format="csc").todense()
        eta = H[m, m - 1]
        self.subdiag_array = fill_block_in_top_right(eta, rows=m, cols=self.T.shape[1])
        self.km += m
        self.V = V
        self.v_next = v_next
        self.m = m
        self.w, self.v = None, None

    def diagonalize(self):
        if self.w is None:
            self.w, self.v = scipy.linalg.eig(self.T)
            if min(self.w) < 0:
                self.w = self.w + 0j
            return 1
        else:
            return 0


def fill_block_in_top_right(eta, rows, cols):
    return scipy.sparse.coo_array(([eta], ([0], [cols - 1])), shape=(rows, cols))
