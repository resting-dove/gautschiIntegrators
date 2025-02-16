# gautschiIntegrators

A package for integrating oscillating second order differential equations
$$x'' = -Ax + g(x).$$

The integrators are described in [[2]](#longTimeConservation).
They make use of trigonometric functions to take advantage of known structure of the solution, as proposed
by [[4]](#gautschi), after whom this package is named.

The matrix functions appearing in the integrators can be evaluated by symmetric diagonalization or by
using [PyWkm](https://github.com/resting-dove/pyWKM), which is a private package at the moment.
Both of these methods can also be combined with the (restarted) Lanczos-method for evaluating matrix functions times a
vector.

## Usage

See `examples/` for example usage of the integrators for a Fermi-Pasta-Ulam-Tsingou problem.

The central function is `solve_ivp` in `integrate.py`.
It is modeled after the Scipy function of the same name, but has differing in- and outputs.

See the `METHODS` dictionary in `integrate.py` for the available one- and two-step configurations of the gautschiIntegrators.

The matrix functions are evaluated by an instance of the `MatrixFunctionEvaluator` base class.
Subclasses need to implement the calculation of $\cos(\sqrt(A))b$, $\sinc(\sqrt(A))b$ and $\sqrt(A)\sin(\sqrt(A))b$.
This can be achieved for example by diagonalization of symmetric tri-diagonal matrices as in `TridiagDiagonalizationEvaluator`.


## Requirements

The code requires the PyWkm package to be installed.
This is a private package at the moment.
In subsequent versions, I will make this dependency optional.

This package does not use low-level methods from its dependencies, so I would expect it to work with any current Numpy
and Scipy versions.
`environment.yml` gives specifications for a Conda environment that this package has been used with.

## Examples

Gautschi-type integrators have usually been evaluated on test problems inspired by the Fermi-Pasta-Ulam-Tsingou experiment, e.g., in [[2]](#longTimeConservation).
`FermiPastaUlamTsingou.py` contains a single frequency example from Section XIII.2.1 of [[3]](#geometricIntegration).
`FPUTMultiFrequency.py` contains a multi frequency example from [[1]](#numericalConservation).
`FPUTMultiFrequencyLanczos.py` is a variation of the former with longer vectors, so that the Lanczos method can be used to approximate the matrix functions.

## Notes

The git branch titled `fermi-pasta-lanczos-experiment` contains the multi-frequency Fermi-Pasta-Ulam-Tsingou problem inspired
by [[1]](#numericalConservation) in `FPUTMultiFrequencyLanczosLonger.py`.
It is similar to `FPUTMultiFrequencyLanczos.py`, but simulates a longer time.

## References

> <a id="numericalConservation">[1]</a> D. Cohen, E. Hairer, and Ch. Lubich, “Numerical Energy Conservation for Multi-Frequency
> Oscillatory Differential Equations,” *Bit Numer Math*, vol. 45, no. 2, pp. 287–305, Jun. 2005, doi:
> 10.1007/s10543-005-7121-z.

> <a id="longTimeConservation">[2]</a> E. Hairer and C. Lubich, “Long-Time Energy Conservation of Numerical Methods for Oscillatory
> Differential Equations,” *SIAM J. Numer. Anal.*, vol. 38, no. 2, pp. 414–441, Jul. 2000, doi: 10.1137/S0036142999353594.

> <a id="geometrixIntegration">[3]</a> E. Hairer, G. Wanner, and C. Lubich, Geometric Numerical Integration, vol. 31. in Springer Series in Computational Mathematics, vol. 31. Berlin/Heidelberg: Springer-Verlag, 2006. doi: 10.1007/3-540-30666-8.

> <a id="gautschi">[4]</a> W. Gautschi, “Numerical integration of ordinary differential equations based on trigonometric
> polynomials,” *Numerische Mathematik*, vol. 3, no. 1, pp. 381–397, Dec. 1961, doi: 10.1007/BF01386037.


