import unittest

import numpy as np
import scipy.sparse

from gautschiIntegrators.one_step import ExplicitEuler, VelocityVerlet
from gautschiIntegrators.integrate import integrate_two_step_f_sym, integrate_two_step_216_sym, integrate_one_step_f_sym


class LinearSecondOrder(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(44)
        self.n = 20
        self.t_end = 0.2
        self.bound = 1e-7  # Chosen kind of randomly
        self.steps = 20

    def get_symmetric_sparse_array(self, n: int, rng: np.random._generator):
        A = rng.random((n, n))
        A = A + A.transpose()
        return scipy.sparse.csr_array(A)

    def get_sym_pos_def(self, n: int, rng: np.random._generator):
        L = np.tril(rng.uniform(50, 51, (n, n)))
        A = L @ L.T
        return scipy.sparse.csr_array(A / 50 ** 3)

    def get_diagonal_positive_array(self, n: int, rng: np.random._generator):
        A = scipy.sparse.spdiags(rng.random(n), 0, n, n)
        return A

    def get_x0(self, n: int, rng: np.random._generator):
        b = rng.random(n)
        return b

    def get_scipy_result(self, A, X, t_end):
        n = A.shape[0]

        def deriv(t, y):
            return np.concatenate((y[n:], -1 * A @ y[:n]))

        scipy_result = scipy.integrate.solve_ivp(deriv, [0, t_end], X)
        return scipy_result["y"][:n, -1], scipy_result["y"][n:, -1]

    def performExpEulerComparison(self, A, X, t_end, steps, bound):
        n = A.shape[0]
        scipy_x, scipy_v = self.get_scipy_result(A, X, t_end)
        g = lambda x: np.zeros_like(x)
        x, v = X[:n], X[n:]
        integrator = ExplicitEuler(t_end / steps)
        for _ in range(steps):
            x, v = integrator.step(A, x, v, g)
        rel_x = scipy.linalg.norm(scipy_x - x) / scipy.linalg.norm(scipy_x)
        rel_v = scipy.linalg.norm(scipy_v - v) / scipy.linalg.norm(scipy_v)
        with self.subTest(f"{steps} steps: x"):
            self.assertLess(rel_x, bound)
        with self.subTest(f"{steps} steps: v"):
            self.assertLess(rel_v, bound)
        return rel_x, rel_v

    def performVVComparison(self, A, X, t_end, steps, bound):
        n = A.shape[0]
        scipy_x, scipy_v = self.get_scipy_result(A, X, t_end)
        x, v = X[:n], X[n:]
        integrator = VelocityVerlet(t_end / steps)
        for _ in range(steps):
            x, v = integrator.step(x, v, force=-1 * A @ x)
        rel_x = scipy.linalg.norm(scipy_x - x) / scipy.linalg.norm(scipy_x)
        rel_v = scipy.linalg.norm(scipy_v - v) / scipy.linalg.norm(scipy_v)
        with self.subTest(f"{steps} steps: x"):
            self.assertLess(rel_x, bound)
        with self.subTest(f"{steps} steps: v"):
            self.assertLess(rel_v, bound)
        return rel_x, rel_v

    def test_ExplicitEuler_should_integrate_SPD(self):
        n = self.n
        A = self.get_sym_pos_def(n=n, rng=self.rng)
        x0 = self.get_x0(n, rng=self.rng)
        v0 = np.zeros_like(x0)
        X = np.concatenate([x0, v0])
        rel_x, rel_v = self.performExpEulerComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performExpEulerComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)
        # With starting velocities
        v0 = self.rng.normal(size=n) / n
        X = np.concatenate([x0, v0])
        rel_x, rel_v = self.performExpEulerComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performExpEulerComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)

    def test_VV_should_integrate_SPD(self):
        n = self.n
        A = self.get_sym_pos_def(n=n, rng=self.rng)
        x0 = self.get_x0(n, rng=self.rng)
        v0 = np.zeros_like(x0)
        X = np.concatenate([x0, v0])
        rel_x, rel_v = self.performVVComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performVVComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)
        # With starting velocities
        v0 = self.rng.normal(size=n) / n
        X = np.concatenate([x0, v0])
        rel_x, rel_v = self.performVVComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performVVComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)

    def test_ExplicitEuler_should_integrate_DiagonalPD(self):
        n = self.n
        A = self.get_diagonal_positive_array(n=n, rng=self.rng)
        x0 = self.get_x0(n, rng=self.rng)
        v0 = np.zeros_like(x0)
        X = np.concatenate([x0, v0])
        rel_x, rel_v = self.performExpEulerComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performExpEulerComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)
        # With starting velocities
        v0 = self.rng.normal(size=n) / n
        X = np.concatenate([x0, v0])
        rel_x, rel_v = self.performExpEulerComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performExpEulerComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)

    def test_ExplicitEuler_should_integrate_Symmetric(self):
        n = self.n
        A = self.get_symmetric_sparse_array(n=n, rng=self.rng)
        x0 = self.get_x0(n, rng=self.rng)
        v0 = np.zeros_like(x0)
        X = np.concatenate([x0, v0])
        rel_x, rel_v = self.performExpEulerComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performExpEulerComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)
        # With starting velocities
        v0 = self.rng.normal(size=n) / n
        X = np.concatenate([x0, v0])
        rel_x, rel_v = self.performExpEulerComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performExpEulerComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)

    def test_VV_should_integrate_DiagonalPD(self):
        n = self.n
        A = self.get_diagonal_positive_array(n=n, rng=self.rng)
        x0 = self.get_x0(n, rng=self.rng)
        v0 = np.zeros_like(x0)
        X = np.concatenate([x0, v0])
        rel_x, rel_v = self.performVVComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performVVComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)
        # With starting velocities
        v0 = self.rng.normal(size=n) / n
        X = np.concatenate([x0, v0])
        rel_x, rel_v = self.performVVComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performVVComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)

    def test_VV_should_integrate_Symmetric(self):
        n = self.n
        A = self.get_symmetric_sparse_array(n=n, rng=self.rng)
        x0 = self.get_x0(n, rng=self.rng)
        v0 = np.zeros_like(x0)
        X = np.concatenate([x0, v0])
        rel_x, rel_v = self.performVVComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performVVComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)
        # With starting velocities
        v0 = self.rng.normal(size=n) / n
        X = np.concatenate([x0, v0])
        rel_x, rel_v = self.performVVComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performVVComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)

    def performTwoStepFComparison(self, A, X, t_end, steps, bound):
        n = A.shape[0]
        scipy_x, scipy_v = self.get_scipy_result(A, X, t_end)
        g = lambda x: np.zeros_like(x)
        x = integrate_two_step_f_sym(A, g, X[:n], t_end / steps, t_end, X[n:])
        rel_x = scipy.linalg.norm(scipy_x - x) / scipy.linalg.norm(scipy_x)
        with self.subTest(f"{steps} steps: x"):
            self.assertLess(rel_x, bound)
        return rel_x

    def test_two_step_f_should_integrate_SPD(self):
        n = self.n
        A = self.get_sym_pos_def(n=n, rng=self.rng)
        x0 = self.get_x0(n, rng=self.rng)
        v0 = np.zeros_like(x0)
        X = np.concatenate([x0, v0])
        rel_x = self.performTwoStepFComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performTwoStepFComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)
        # With starting velocities
        v0 = self.rng.normal(size=n) / n
        X = np.concatenate([x0, v0])
        rel_x = self.performTwoStepFComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performTwoStepFComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)

    def test_two_step_f_should_integrate_DiagonalPD(self):
        n = self.n
        A = self.get_diagonal_positive_array(n=n, rng=self.rng)
        x0 = self.get_x0(n, rng=self.rng)
        v0 = np.zeros_like(x0)
        X = np.concatenate([x0, v0])
        rel_x = self.performTwoStepFComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performTwoStepFComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)
        # With starting velocities
        v0 = self.rng.normal(size=n) / n
        X = np.concatenate([x0, v0])
        rel_x = self.performTwoStepFComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performTwoStepFComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)

    # Diagonalization approach only practicable for SPD matrices
    # def test_two_step_f_should_integrate_symmetric(self):
    #     n = self.n
    #     A = self.get_symmetric_sparse_array(n=n, rng=self.rng)
    #     x0 = self.get_x0(n, rng=self.rng)
    #     v0 = np.zeros_like(x0)
    #     X = np.concatenate([x0, v0])
    #     self.performTwoStepFComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)

    def performTwoStep216Comparison(self, A, X, t_end, steps, bound):
        n = A.shape[0]
        scipy_x, scipy_v = self.get_scipy_result(A, X, t_end)
        g = lambda x: np.zeros_like(x)
        x = integrate_two_step_216_sym(A, g, X[:n], t_end / steps, t_end, X[n:])
        rel_x = scipy.linalg.norm(scipy_x - x) / scipy.linalg.norm(scipy_x)
        with self.subTest(f"{steps} steps: x"):
            self.assertLess(rel_x, bound)
        return rel_x

    def test_two_step_216_should_integrate_SPD(self):
        n = self.n
        A = self.get_sym_pos_def(n=n, rng=self.rng)
        x0 = self.get_x0(n, rng=self.rng)
        v0 = np.zeros_like(x0)
        X = np.concatenate([x0, v0])
        rel_x = self.performTwoStep216Comparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performTwoStep216Comparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)
        # With starting velocities
        v0 = self.rng.normal(size=n) / n
        X = np.concatenate([x0, v0])
        rel_x = self.performTwoStep216Comparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performTwoStep216Comparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)

    def test_two_step_216_should_integrate_DiagonalPD(self):
        n = self.n
        A = self.get_diagonal_positive_array(n=n, rng=self.rng)
        x0 = self.get_x0(n, rng=self.rng)
        v0 = np.zeros_like(x0)
        X = np.concatenate([x0, v0])
        rel_x = self.performTwoStep216Comparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performTwoStep216Comparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)
        # With starting velocities
        v0 = self.rng.normal(size=n) / n
        X = np.concatenate([x0, v0])
        rel_x = self.performTwoStep216Comparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performTwoStep216Comparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)

    def performOneStepFComparison(self, A, X, t_end, steps, bound):
        n = A.shape[0]
        scipy_x, scipy_v = self.get_scipy_result(A, X, t_end)
        g = lambda x: np.zeros_like(x)
        x, v = integrate_one_step_f_sym(A, g, X[:n], t_end / steps, t_end, X[n:])
        rel_x = scipy.linalg.norm(scipy_x - x) / scipy.linalg.norm(scipy_x)
        with self.subTest(f"{steps} steps: x"):
            self.assertLess(rel_x, bound)
        rel_v = scipy.linalg.norm(scipy_v - v) / scipy.linalg.norm(scipy_v)
        with self.subTest(f"{steps} steps: v"):
            self.assertLess(rel_v, bound)
        return rel_x

    def test_one_step_F_should_integrate_SPD(self):
        n = self.n
        A = self.get_sym_pos_def(n=n, rng=self.rng)
        x0 = self.get_x0(n, rng=self.rng)
        v0 = np.zeros_like(x0)
        X = np.concatenate([x0, v0])
        rel_x = self.performOneStepFComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performOneStepFComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)
        # With starting velocities
        v0 = self.rng.normal(size=n) / n
        X = np.concatenate([x0, v0])
        rel_x = self.performOneStepFComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performOneStepFComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)

    def test_one_step_F_should_integrate_DiagonalPD(self):
        n = self.n
        A = self.get_diagonal_positive_array(n=n, rng=self.rng)
        x0 = self.get_x0(n, rng=self.rng)
        v0 = np.zeros_like(x0)
        X = np.concatenate([x0, v0])
        rel_x = self.performOneStepFComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performOneStepFComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)
        # With starting velocities
        v0 = self.rng.normal(size=n) / n
        X = np.concatenate([x0, v0])
        rel_x = self.performOneStepFComparison(A, X, t_end=self.t_end, steps=self.steps, bound=self.bound)
        self.performOneStepFComparison(A, X, t_end=self.t_end, steps=2 * self.steps, bound=rel_x)


if __name__ == '__main__':
    unittest.main()
