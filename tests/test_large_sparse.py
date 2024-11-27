import unittest

import numpy as np
import scipy.sparse

from gautschiIntegrators.integrate import solve_ivp
from gautschiIntegrators.lanczos.LanczosDiagonalizationEvaluator import LanczosDiagonalizationEvaluator
from gautschiIntegrators.matrix_functions import SymDiagonalizationEvaluator, \
    TridiagDiagonalizationEvaluator, DenseWkmEvaluator
from tests.test_solve_ivp import Ivp
from tests.utils import compute_error


class PositiveTriDiagonal(Ivp):
    def setUp(self):
        self.rng = np.random.default_rng(44)
        self.n = 1000
        self.get_matrix()

        self.rtol = 1e-5
        self.atol = 1e-5
        self.x0 = self.rng.random(self.n)
        self.t_end = 0.15
        self.h = 0.1 / 200
        self.x_true = self.get_scipy_result()

        self.v0 = self.rng.normal(0, 1, self.n)
        self.x_true2 = self.get_scipy_result(self.v0)

        self.g = self.get_g(self.x0)
        self.x_true3 = self.get_scipy_result(self.v0, self.g)

    def get_matrix(self):
        self.A = scipy.sparse.diags_array(self.rng.random(self.n), offsets=0, shape=(self.n, self.n)) * 10
        od = (self.A.diagonal() * self.rng.random(self.n)) / 2
        od = scipy.sparse.diags_array(od[:-1], offsets=1, shape=(self.n, self.n))
        self.A = self.A + od + od.T + scipy.sparse.eye_array(self.n) * 10

    # def test_OneStepF_SymTriDiagonalization(self):
    #     self.gautschiEvaluation("OneStepF", TridiagDiagonalizationEvaluator())

    def test_OneStepF_LanczosDiagonalization(self):
        self.gautschiEvaluation("OneStepF", LanczosDiagonalizationEvaluator(krylov_size=20))

    # def test_OneStepF_Wkm(self):
    #     self.gautschiEvaluation("OneStepF", DenseWkmEvaluator())
    #
    # def test_TwoStepF_SymDiagonalization(self):
    #     self.gautschiEvaluation("TwoStepF", SymDiagonalizationEvaluator())
    #
    # def test_TwoStepF_SymTriDiagDiagonalization(self):
    #     self.gautschiEvaluation("TwoStepF", TridiagDiagonalizationEvaluator())
    #
    # def test_TwoStepF_Wkm(self):
    #     self.gautschiEvaluation("TwoStepF", DenseWkmEvaluator())
    #
    # def test_TwoStep216_SymDiagonalization(self):
    #     self.gautschiEvaluation("TwoStep216", SymDiagonalizationEvaluator())
    #
    # def test_TwoStep216_SymTriDiagDiagonalization(self):
    #     self.gautschiEvaluation("TwoStep216", TridiagDiagonalizationEvaluator())
    #
    # def test_TwoStep216_Wkm(self):
    #     self.gautschiEvaluation("TwoStep216", DenseWkmEvaluator())

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


if __name__ == '__main__':
    unittest.main()
