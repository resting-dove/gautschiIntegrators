import numpy as np
import scipy


class VelocityVerlet:
    def __init__(self, h: float):
        self.prev_force = None
        self.h = h

    def advance_positions(self, x: np.array, v: np.array, force: np.ndarray) -> np.array:
        """Part one of the Velocity - Verlet scheme.

        Parameters
        ----------
        x, v : array_like
            Positions and velocities of the particles. Length is 3n.
        force : array_like
            Forces acting on the particles, already divided by the mass term. Length is 3n.
        """
        next_x = x + self.h * v + 0.5 * self.h ** 2 * force
        self.prev_force = force
        return next_x

    def advance_velocities(self, v: np.array, force: np.ndarray) -> np.array:
        """Part two of the Velocity - Verlet scheme.

        Parameters
        ----------
        v : array_like
            Positions and velocities of the particles. Length is 3n.
        force : array_like
            Forces acting on the particles, already divided by the mass term. Length is 3n.
        """
        next_v = v + 0.5 * self.h * (self.prev_force + force)
        return next_v

    def step(self, x: np.array, v: np.array, force: np.array) -> (np.array, np.array):
        """Perform one step of the Velocity - Verlet scheme.

        Parameters
        ----------
        x, v : array_like
            Positions and velocities of the particles. Length is 3n.
        force : array_like
            Forces acting on the particles, already divided by the mass term. Length is 3n.
        """
        next_x = self.advance_positions(x, v, force)
        next_v = self.advance_velocities(v, force)
        return next_x, next_v


class ExplicitEuler:
    """
    Solve second order differential equation $x'' = - \Omega^2 x + g(x)$ by transforming into first order ODE and
    applying the explicit Euler method.
    """

    def __init__(self, h: float):
        self.h = h

    def step(self, omega2: scipy.sparse.sparray, x: np.array, v: np.array, g: callable) -> (np.array, np.array):
        X = np.concatenate([x, v])
        mat = scipy.sparse.block_array([[None, scipy.sparse.eye_array(*v.shape)], [-1 * omega2, None]])
        G = np.concatenate([np.zeros_like(x), g(x)])
        next_X = X + self.h * (mat @ X + G)
        nextx, next_v = next_X[:len(x)], next_X[len(x):]
        return nextx, next_v
