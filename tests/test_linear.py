import unittest

import numpy as np
import scipy.sparse

from gautschiIntegrators.matrix_functions import sym_cosm_sqrt, sym_sincm_sqrt
from gautschiIntegrators.one_step import ExplicitEuler, VelocityVerlet
from gautschiIntegrators.integrate import integrate_two_step_f_sym, integrate_two_step_216_sym, \
    integrate_one_step_f_sym, solve_ivp


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

    def get_scipy_result(self, v0=None):
        if v0 is None:
            v0 = 0 * self.x0
        X = np.concatenate([self.x0, v0])

        def deriv(t, y):
            return np.concatenate((y[self.n:], -1 * self.A @ y[:self.n]))

        scipy_result = scipy.integrate.solve_ivp(deriv, [0, self.t_end], X)
        return scipy_result["y"][:self.n, -1]


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


class PositiveDiagonal(LinearODE):
    def get_matrix(self):
        self.A = scipy.sparse.spdiags(self.rng.random(self.n), 0, self.n, self.n)

    def test_OneStepF(self):
        res = solve_ivp(self.A, None, self.t_end / 200, self.t_end, self.x0, None, "OneStepF", cosm=sym_cosm_sqrt,
                        sincm=sym_sincm_sqrt)
        self.assertTrue(np.isclose(res["t"], self.t_end))
        self.assertTrue(res["success"])
        e = compute_error(res["x"], self.x_true, self.rtol, self.atol)
        self.assertTrue(np.all(e < 5))

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
        res = solve_ivp(self.A, None, self.t_end / 200, self.t_end, self.x0, None, "OneStepF", cosm=sym_cosm_sqrt,
                        sincm=sym_sincm_sqrt)
        self.assertTrue(np.isclose(res["t"], self.t_end))
        self.assertTrue(res["success"])
        e = compute_error(res["x"], self.x_true, self.rtol, self.atol)
        self.assertTrue(np.all(e < 5))

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
