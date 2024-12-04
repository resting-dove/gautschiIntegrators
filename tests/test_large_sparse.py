import unittest

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from gautschiIntegrators.integrate import solve_ivp
from gautschiIntegrators.lanczos.LanczosEvaluator import LanczosDiagonalizationEvaluator, LanczosWkmEvaluator
from gautschiIntegrators.matrix_functions import DenseWkmEvaluator, WkmEvaluator
from tests.test_solve_ivp import Ivp
from tests.utils import compute_error


class LargeIvp(Ivp):
    def setUp(self):
        self.rng = np.random.default_rng(47)
        self.n = 200
        self.get_matrix()

        self.rtol = 1e-4
        self.atol = 1e-4
        self.x0 = self.rng.random(self.n)
        self.x0 = self.x0 / scipy.linalg.norm(self.x0)
        self.t_end = 0.13
        self.h = 0.1 / 200

        self.x_true = self.get_scipy_result()

        self.get_velocities()
        self.x_true2 = self.get_scipy_result(self.v0)

        self.g = self.get_g(self.x0)
        self.x_true3 = self.get_scipy_result(self.v0, self.g)

    def get_velocities(self):
        self.v0 = self.rng.normal(0, scipy.sparse.linalg.norm(self.A) / self.n, self.n) / 100


class QuintDiagonal(LargeIvp):
    def get_matrix(self):
        d = self.rng.random(self.n - 3)
        self.A = scipy.sparse.diags_array(d, offsets=0, shape=(len(d), len(d))) * 10
        od = (self.A.diagonal() * self.rng.random(len(d))) / 2
        od = scipy.sparse.diags_array(od[:-1], offsets=1, shape=(len(d), len(d)))
        ood = (self.A.diagonal() * self.rng.random(len(d))) / 4
        ood = scipy.sparse.diags_array(ood[:-2], offsets=2, shape=self.A.shape)
        self.A = (self.A + od + od.T + ood + ood.T + scipy.sparse.eye_array(len(d)) * 10) / 300
        self.A = scipy.sparse.block_diag([self.A, scipy.sparse.coo_array([[0, 0, 0], [0, 80, 0], [0, 0, 8]])],
                                         format='csr')

        print("evals", min(np.linalg.eigvals(self.A.todense())), max(np.linalg.eigvals(self.A.todense())))

    def test_OneStepF_LanczosDiagonalization(self):
        self.gautschiEvaluation("OneStepF", LanczosDiagonalizationEvaluator(krylov_size=180))

    def test_OneStepF_LanczosWkm(self):
        self.gautschiEvaluation("OneStepF", LanczosWkmEvaluator(krylov_size=80))

    def test_OneStepF_Wkm(self):
        self.gautschiEvaluation("OneStepF", WkmEvaluator())

    def test_OneStepF_DenseWkmUsingLanczos(self):
        self.gautschiEvaluation("OneStepF", LanczosWkmEvaluator(krylov_size=200))

    def test_OneStepF_DenseWkm(self):
        self.gautschiEvaluation("OneStepF", DenseWkmEvaluator())

    def test_ExplicitEuler(self):
        with self.subTest("Linear ODE"):
            res = solve_ivp(self.A, None, self.h / 5, self.t_end, self.x0, None, "ExplicitEuler")
            self.assertTrue(np.isclose(res["t"], self.t_end))
            self.assertTrue(res["success"])
            e = compute_error(res["x"], self.x_true, self.rtol, self.atol)
            self.assertLess(e, 5)
        with self.subTest("With starting velocities"):
            res = solve_ivp(self.A, None, self.h / 5, self.t_end, self.x0, self.v0, "ExplicitEuler")
            e = compute_error(res["x"], self.x_true2, self.rtol, self.atol)
            self.assertLess(e, 5)
        with self.subTest("Nonlinear Diff Eq"):
            res = solve_ivp(self.A, self.g, self.h / 5, self.t_end, self.x0, self.v0, "ExplicitEuler")
            e = compute_error(res["x"], self.x_true3, self.rtol, self.atol)
            self.assertLess(e, 5)


if __name__ == '__main__':
    unittest.main()
