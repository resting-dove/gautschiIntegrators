# gautschiIntegrators

A package for integrating oscillating second order differential equations
$$x'' = -Ax + g(x).$$

The integrators are described in [[1]](#first).
They make use of trigonometric functions to take advantage of known structure of the solution, as proposed
by [[2]](#second), after whom this package is named.

The matrix functions appearing in the integrators can be evaluated by symmetric diagonalization or by
using [PyWkm](https://github.com/resting-dove/pyWKM).
Both of these methods can also be combined with the (restarted) Lanczos-method for evaluating matrix functions times a
vector.

## Usage

See `examples/` for example usage of the integrators for a Fermi-Pasta-Ulam-Tsingou problem.

The central function is `solve_ivp` in `integrate.py`.
It is modeled after the Scipy function of the same name, but has differing in- and outputs.

The matrix functions are evaluated by an instance of the `MatrixFunctionEvaluator` base class.
Subclasses implement for example use diagonalization of symmetric tri-diagonal matrices.

## Requirements

The code of PyWkm needs to be copied into a folder `pywkm` for usage in this repository.

This package does not use low-level methods from its dependencies, so I would expect it to work with any current Numpy
and Scipy versions.
`environment.yml` gives specifications for a Conda environment that this package has been used with.

## Notes

The branch titled `fermi-pasta-lanczos-experiment` contains a multi-frequency Fermi-Pasta-Ulam-Tsingou problem inspired
by [[3]](#third).

## References

> <a id="first">[1]</a> E. Hairer and C. Lubich, “Long-Time Energy Conservation of Numerical Methods for Oscillatory
> Differential Equations,” *SIAM J. Numer. Anal.*, vol. 38, no. 2, pp. 414–441, Jul. 2000, doi: 10.1137/S0036142999353594.

> <a id="second">[2]</a> W. Gautschi, “Numerical integration of ordinary differential equations based on trigonometric
> polynomials,” *Numerische Mathematik*, vol. 3, no. 1, pp. 381–397, Dec. 1961, doi: 10.1007/BF01386037.

> <a id="third">[3]</a> D. Cohen, E. Hairer, and Ch. Lubich, “Numerical Energy Conservation for Multi-Frequency
> Oscillatory Differential Equations,” *Bit Numer Math*, vol. 45, no. 2, pp. 287–305, Jun. 2005, doi:
> 10.1007/s10543-005-7121-z.
