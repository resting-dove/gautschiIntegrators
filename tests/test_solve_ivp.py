import copy
import unittest

import numpy as np
import scipy.sparse

from gautschiIntegrators.matrix_functions import WkmEvaluator, SymDiagonalizationEvaluator, \
    TridiagDiagonalizationEvaluator, DenseWkmEvaluator
from gautschiIntegrators.one_step import ExplicitEuler, VelocityVerlet
from gautschiIntegrators.integrate import solve_ivp


class Ivp(unittest.TestCase):
    def get_matrix(self):
        raise NotImplemented

    def setUp(self):
        self.rng = np.random.default_rng(44)
        self.n = 20
        self.get_matrix()

        self.rtol = 1e-5
        self.atol = 1e-5
        self.x0 = self.rng.random(self.n)
        self.t_end = 0.13
        self.h = 0.1 / 200
        self.x_true = self.get_scipy_result()

        self.v0 = self.rng.normal(0, 1, self.n)
        self.x_true2 = self.get_scipy_result(self.v0)

        self.g = self.get_g(self.x0)
        self.x_true3 = self.get_scipy_result(self.v0, self.g)

    def get_scipy_result(self, v0=None, g=None):
        if v0 is None:
            v0 = 0 * self.x0
        if g is None:
            g = lambda x: 0
        X = np.concatenate([self.x0, v0])

        def deriv(t, y):
            return np.concatenate((y[self.n:], -1 * self.A @ y[:self.n] + g(y[:self.n])))

        scipy_result = scipy.integrate.solve_ivp(deriv, [0, self.t_end], X)
        return scipy_result["y"][:self.n, -1]

    def get_g(self, x0):
        x0 = copy.deepcopy(x0)

        def g(x):
            return (x - x0) ** 2

        return g

    def gautschiEvaluation(self, methodName, evaluator=None):
        with self.subTest("Linear ODE"):
            res = solve_ivp(self.A, None, self.h, self.t_end, self.x0, None, methodName,
                            evaluator=evaluator)
            self.assertTrue(np.isclose(res["t"], self.t_end))
            self.assertTrue(res["success"])
            e = compute_error(res["x"], self.x_true, self.rtol, self.atol)
            self.assertTrue(np.all(e < 5))

        with self.subTest("Linear ODE with starting velocities"):
            res = solve_ivp(self.A, None, self.h, self.t_end, self.x0, self.v0, methodName,
                            evaluator=evaluator)
            e = compute_error(res["x"], self.x_true2, self.rtol, self.atol)
            self.assertTrue(np.all(e < 5))

        with self.subTest("Nonlinear Diff Eq"):
            res = solve_ivp(self.A, self.g, self.h, self.t_end, self.x0, self.v0, methodName,
                            evaluator=evaluator)
            e = compute_error(res["x"], self.x_true3, self.rtol, self.atol)
            self.assertTrue(np.all(e < 5))


class SymmetricSparseArray(Ivp):
    def get_matrix(self):
        A = self.rng.random((self.n, self.n))
        A = A + A.transpose()
        self.A = scipy.sparse.csr_array(A)

    def test_OneStepF(self):
        self.gautschiEvaluation("OneStepF", WkmEvaluator())

    def test_TwoStepF(self):
        self.gautschiEvaluation("TwoStepF", DenseWkmEvaluator())

    def test_TwoStep216(self):
        self.gautschiEvaluation("TwoStep216", DenseWkmEvaluator())

    def test_ExplicitEuler(self):
        with self.subTest("Linear ODE"):
            res = solve_ivp(self.A, None, self.h / 5, self.t_end, self.x0, None, "ExplicitEuler")
            self.assertTrue(np.isclose(res["t"], self.t_end))
            self.assertTrue(res["success"])
            e = compute_error(res["x"], self.x_true, self.rtol, self.atol)
            self.assertTrue(np.all(e < 5))
        with self.subTest("With starting velocities"):
            res = solve_ivp(self.A, None, self.h / 5, self.t_end, self.x0, self.v0, "ExplicitEuler")
            e = compute_error(res["x"], self.x_true2, self.rtol, self.atol)
            self.assertTrue(np.all(e < 5))
        with self.subTest("Nonlinear Diff Eq"):
            res = solve_ivp(self.A, self.g, self.h / 5, self.t_end, self.x0, self.v0, "ExplicitEuler")
            e = compute_error(res["x"], self.x_true3, self.rtol, self.atol)
            self.assertTrue(np.all(e < 5))


class PositiveDiagonal(Ivp):
    def get_matrix(self):
        self.A = scipy.sparse.diags_array(self.rng.random(self.n), offsets=0, shape=(self.n, self.n))

    def test_OneStepF_SymDiagonalization(self):
        self.gautschiEvaluation("OneStepF", SymDiagonalizationEvaluator())

    def test_OneStepF_SymTriDiagDiagonalization(self):
        self.gautschiEvaluation("OneStepF", TridiagDiagonalizationEvaluator())

    def test_OneStepF_Wkm(self):
        self.gautschiEvaluation("OneStepF", DenseWkmEvaluator())

    def test_TwoStepF_SymDiagonalization(self):
        self.gautschiEvaluation("TwoStepF", SymDiagonalizationEvaluator())

    def test_TwoStepF_SymTriDiagDiagonalization(self):
        self.gautschiEvaluation("TwoStepF", TridiagDiagonalizationEvaluator())

    def test_TwoStepF_Wkm(self):
        self.gautschiEvaluation("TwoStepF", DenseWkmEvaluator())

    def test_TwoStep216_SymDiagonalization(self):
        self.gautschiEvaluation("TwoStep216", SymDiagonalizationEvaluator())

    def test_TwoStep216_SymTriDiagDiagonalization(self):
        self.gautschiEvaluation("TwoStep216", TridiagDiagonalizationEvaluator())

    def test_TwoStep216_Wkm(self):
        self.gautschiEvaluation("TwoStep216", DenseWkmEvaluator())

    def test_ExplicitEuler(self):
        with self.subTest("Linear ODE"):
            res = solve_ivp(self.A, None, self.h / 5, self.t_end, self.x0, None, "ExplicitEuler")
            self.assertTrue(np.isclose(res["t"], self.t_end))
            self.assertTrue(res["success"])
            e = compute_error(res["x"], self.x_true, self.rtol, self.atol)
            self.assertTrue(np.all(e < 5))
        with self.subTest("With starting velocities"):
            res = solve_ivp(self.A, None, self.h / 5, self.t_end, self.x0, self.v0, "ExplicitEuler")
            e = compute_error(res["x"], self.x_true2, self.rtol, self.atol)
            self.assertTrue(np.all(e < 5))
        with self.subTest("Nonlinear Diff Eq"):
            res = solve_ivp(self.A, self.g, self.h / 5, self.t_end, self.x0, self.v0, "ExplicitEuler")
            e = compute_error(res["x"], self.x_true3, self.rtol, self.atol)
            self.assertTrue(np.all(e < 5))


class SymmetricPositiveDefinite(Ivp):
    def get_matrix(self):
        L = np.tril(self.rng.uniform(50, 51, (self.n, self.n)))
        A = L @ L.T
        self.A = scipy.sparse.csr_array(A / 50 ** 3)

    def test_OneStepF_SymDiagonalization(self):
        self.gautschiEvaluation("OneStepF", SymDiagonalizationEvaluator())

    def test_OneStepF_Wkm(self):
        self.gautschiEvaluation("OneStepF", DenseWkmEvaluator())

    def test_TwoStepF_SymDiagonalization(self):
        self.gautschiEvaluation("TwoStepF", SymDiagonalizationEvaluator())

    def test_TwoStepF_Wkm(self):
        self.gautschiEvaluation("TwoStepF", DenseWkmEvaluator())

    def test_TwoStep216_SymDiagonalization(self):
        self.gautschiEvaluation("TwoStep216", SymDiagonalizationEvaluator())

    def test_TwoStep216_Wkm(self):
        self.gautschiEvaluation("TwoStep216", DenseWkmEvaluator())

    def test_ExplicitEuler(self):
        with self.subTest("Linear ODE"):
            res = solve_ivp(self.A, None, self.h / 5, self.t_end, self.x0, None, "ExplicitEuler")
            self.assertTrue(np.isclose(res["t"], self.t_end))
            self.assertTrue(res["success"])
            e = compute_error(res["x"], self.x_true, self.rtol, self.atol)
            self.assertTrue(np.all(e < 5))
        with self.subTest("With starting velocities"):
            res = solve_ivp(self.A, None, self.h / 5, self.t_end, self.x0, self.v0, "ExplicitEuler")
            e = compute_error(res["x"], self.x_true2, self.rtol, self.atol)
            self.assertTrue(np.all(e < 5))
        with self.subTest("Nonlinear Diff Eq"):
            res = solve_ivp(self.A, self.g, self.h / 5, self.t_end, self.x0, self.v0, "ExplicitEuler")
            e = compute_error(res["x"], self.x_true3, self.rtol, self.atol)
            self.assertTrue(np.all(e < 5))


def compute_error(x, x_true, rtol, atol):
    e = (x - x_true) / (atol + rtol * np.abs(x_true))
    return np.linalg.norm(e, axis=0) / np.sqrt(e.shape[0])


if __name__ == '__main__':
    unittest.main()
