import unittest

import numpy as np
import scipy.sparse

from gautschiIntegrators.matrix_functions import sym_cosm_sqrt, sym_sincm_sqrt
from gautschiIntegrators.one_step import ExplicitEuler, VelocityVerlet
from gautschiIntegrators.integrate import solve_ivp


class LinearODE(unittest.TestCase):
    def get_matrix(self):
        raise NotImplemented

    def setUp(self):
        self.rng = np.random.default_rng(44)
        self.n = 20
        self.get_matrix()

        self.rtol = 1e-5
        self.atol = 1e-5
        self.x0 = self.rng.random(self.n)
        self.t_end = 0.1
        self.x_true = self.get_scipy_result()
        self.v0 = self.rng.normal(0, 1, self.n)
        self.x_true2 = self.get_scipy_result(self.v0)

    def get_scipy_result(self, v0=None):
        if v0 is None:
            v0 = 0 * self.x0
        X = np.concatenate([self.x0, v0])

        def deriv(t, y):
            return np.concatenate((y[self.n:], -1 * self.A @ y[:self.n]))

        scipy_result = scipy.integrate.solve_ivp(deriv, [0, self.t_end], X)
        return scipy_result["y"][:self.n, -1]

    def gautschiEvaluation(self, methodName):
        res = solve_ivp(self.A, None, self.t_end / 200, self.t_end, self.x0, None, methodName, cosm=sym_cosm_sqrt,
                        sincm=sym_sincm_sqrt)
        self.assertTrue(np.isclose(res["t"], self.t_end))
        self.assertTrue(res["success"])
        e = compute_error(res["x"], self.x_true, self.rtol, self.atol)
        self.assertTrue(np.all(e < 5))

        with self.subTest("with starting velocities"):
            res = solve_ivp(self.A, None, self.t_end / 200, self.t_end, self.x0, self.v0, methodName,
                            cosm=sym_cosm_sqrt,
                            sincm=sym_sincm_sqrt)
            e = compute_error(res["x"], self.x_true2, self.rtol, self.atol)
            self.assertTrue(np.all(e < 5))


class SymmetricSparseArray(LinearODE):
    def get_matrix(self):
        A = self.rng.random((self.n, self.n))
        A = A + A.transpose()
        self.A = scipy.sparse.csr_array(A)

    # def test_OneStepF(self):
    #     res = solve_ivp(self.A, None, self.t_end / 200, self.t_end, self.x0, None, "OneStepF", cosm=sym_cosm_sqrt,
    #                     sincm=sym_sincm_sqrt)
    #     self.assertTrue(np.isclose(res["t"], self.t_end))
    #     self.assertTrue(res["success"])
    #     e = compute_error(res["x"], self.x_true, self.rtol, self.atol)
    #     self.assertTrue(np.all(e < 5))

    def test_ExplicitEuler(self):
        res = solve_ivp(self.A, None, self.t_end / 1000, self.t_end, self.x0, None, "ExplicitEuler", cosm=sym_cosm_sqrt,
                        sincm=sym_sincm_sqrt)
        self.assertTrue(np.isclose(res["t"], self.t_end))
        self.assertTrue(res["success"])
        e = compute_error(res["x"], self.x_true, self.rtol, self.atol)
        self.assertTrue(np.all(e < 5))
        with self.subTest("With starting velocities"):
            res = solve_ivp(self.A, None, self.t_end / 1000, self.t_end, self.x0, self.v0, "ExplicitEuler",
                            cosm=sym_cosm_sqrt,
                            sincm=sym_sincm_sqrt)
            e = compute_error(res["x"], self.x_true2, self.rtol, self.atol)
            self.assertTrue(np.all(e < 5))


class PositiveDiagonal(LinearODE):
    def get_matrix(self):
        self.A = scipy.sparse.spdiags(self.rng.random(self.n), 0, self.n, self.n)

    def test_OneStepF(self):
        self.gautschiEvaluation("OneStepF")

    def test_TwoStepF(self):
        self.gautschiEvaluation("TwoStepF")

    def test_TwoStep216(self):
        self.gautschiEvaluation("TwoStep216")

    def test_ExplicitEuler(self):
        res = solve_ivp(self.A, None, self.t_end / 1000, self.t_end, self.x0, None, "ExplicitEuler", cosm=sym_cosm_sqrt,
                        sincm=sym_sincm_sqrt)
        self.assertTrue(np.isclose(res["t"], self.t_end))
        self.assertTrue(res["success"])
        e = compute_error(res["x"], self.x_true, self.rtol, self.atol)
        self.assertTrue(np.all(e < 5))


class SymmetricPositiveDefinite(LinearODE):
    def get_matrix(self):
        L = np.tril(self.rng.uniform(50, 51, (self.n, self.n)))
        A = L @ L.T
        self.A = scipy.sparse.csr_array(A / 50 ** 3)

    def test_OneStepF(self):
        self.gautschiEvaluation("OneStepF")

    def test_TwoStepF(self):
        self.gautschiEvaluation("TwoStepF")

    def test_TwoStep216(self):
        self.gautschiEvaluation("TwoStep216")

    def test_ExplicitEuler(self):
        res = solve_ivp(self.A, None, self.t_end / 1000, self.t_end, self.x0, None, "ExplicitEuler", cosm=sym_cosm_sqrt,
                        sincm=sym_sincm_sqrt)
        self.assertTrue(np.isclose(res["t"], self.t_end))
        self.assertTrue(res["success"])
        e = compute_error(res["x"], self.x_true, self.rtol, self.atol)
        self.assertTrue(np.all(e < 5))


def compute_error(x, x_true, rtol, atol):
    e = (x - x_true) / (atol + rtol * np.abs(x_true))
    return np.linalg.norm(e, axis=0) / np.sqrt(e.shape[0])


if __name__ == '__main__':
    unittest.main()
