"""
B-spline basis construction via Cox-de Boor recursion.

Translated from APRS.cpp (Rcpp implementation) to pure NumPy.

References
----------
- De Boor, C. (1972). On calculating with B-splines.
- Lee, D.-Y., Bak, K.-Y., & Jhong, J.-H. (2024). ANZJS.
- Lee, D.-Y., Bak, K.-Y., & Jhong, J.-H. (2025). Generalized APRS.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Single-point B-spline evaluation (Cox-de Boor recursion)
# ---------------------------------------------------------------------------

def _bsp(x, t, degree, j):
    """Evaluate j-th B-spline basis at a single point x.

    Corresponds to ``bsp()`` in APRS.cpp.
    """
    if degree == 0:
        return 1.0 if (t[j] <= x < t[j + 1]) else 0.0

    order = degree + 1
    jd, jpk, jp1 = j + degree, j + order, j + 1

    c = t[jd] - t[j]
    a = (x - t[j]) / c if c > 0 else 0.0

    d = t[jpk] - t[jp1]
    b = (t[jpk] - x) / d if d > 0 else 0.0

    return a * _bsp(x, t, degree - 1, j) + b * _bsp(x, t, degree - 1, jp1)


def _dbsp(x, t, degree, j, derivative):
    """Evaluate derivative of j-th B-spline at a single point x.

    Corresponds to ``dbsp()`` in APRS.cpp.
    """
    if derivative == 0:
        return _bsp(x, t, degree, j)

    order = degree + 1
    jp1 = j + 1

    c = t[j + degree] - t[j]
    a = degree / c if c > 0 else 0.0

    d = t[j + order] - t[jp1]
    b = degree / d if d > 0 else 0.0

    return (a * _dbsp(x, t, degree - 1, j, derivative - 1)
            - b * _dbsp(x, t, degree - 1, jp1, derivative - 1))


# ---------------------------------------------------------------------------
# Vectorized B-spline evaluation
# ---------------------------------------------------------------------------

def _bspline_vec(x, t, degree, j, derivative=0):
    """Evaluate j-th B-spline at all points in x.

    Corresponds to ``bspline()`` in APRS.cpp.
    """
    n = len(x)
    order = degree + 1
    result = np.zeros(n)

    for i in range(n):
        if t[j] <= x[i] < t[j + order]:
            if derivative == 0:
                result[i] = _bsp(x[i], t, degree, j)
            else:
                result[i] = _dbsp(x[i], t, degree, j, derivative)

    return result


def bspline_basis(x, t, degree, derivative=0):
    """Construct B-spline basis matrix.

    Corresponds to ``bsplines()`` in APRS.cpp.

    Parameters
    ----------
    x : array-like of shape (n,)
        Evaluation points (ordered).
    t : array-like
        Full knot sequence (including boundary padding).
    degree : int
        Degree of the B-spline (0=constant, 1=linear, 2=quadratic, 3=cubic).
    derivative : int, default=0
        Order of derivative to evaluate.

    Returns
    -------
    B : ndarray of shape (n, J)
        B-spline basis matrix, J = len(t) - degree - 1.

    Examples
    --------
    >>> x = np.linspace(0, 1, 100)
    >>> knots = dim2knots(x, dimension=52, degree=0)
    >>> t = knots2t(knots, degree=0)
    >>> B = bspline_basis(x, t, degree=0)
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    dimension = len(t) - degree - 1

    B = np.zeros((len(x), dimension))
    for j in range(dimension):
        B[:, j] = _bspline_vec(x, t, degree, j, derivative)

    return B


# ---------------------------------------------------------------------------
# Knot construction utilities
# ---------------------------------------------------------------------------

def dim2knots(predictor, dimension, degree):
    """Generate knots from predictor values and target dimension.

    Places knots at quantile-like positions of the predictor.
    Corresponds to ``dim2knots()`` in APRS.cpp.

    Parameters
    ----------
    predictor : array-like of shape (n,)
        Ordered predictor values.
    dimension : int
        Target number of basis functions.
    degree : int
        B-spline degree.

    Returns
    -------
    knots : ndarray
        Knot positions (interior + right boundary).
    """
    predictor = np.asarray(predictor, dtype=float)
    sample_size = len(predictor)
    knot_size = dimension - degree + 1
    knots = np.zeros(knot_size)

    for j in range(knot_size - 1):
        idx = int(sample_size / (knot_size - 1.0) * j)
        knots[j] = predictor[idx]

    knots[-1] = np.max(predictor) + 1e-5
    return knots


def knots2t(knots, degree):
    """Extend knots to full knot sequence with boundary padding.

    Corresponds to ``knots2t()`` in APRS.cpp.

    Parameters
    ----------
    knots : array-like
        Interior + boundary knots.
    degree : int
        B-spline degree.

    Returns
    -------
    t : ndarray
        Full (padded) knot sequence.
    """
    knots = np.asarray(knots, dtype=float)
    d = np.mean(np.diff(knots))
    min_k, max_k = knots.min(), knots.max()
    n_knots = len(knots)

    t = np.zeros(2 * degree + n_knots)

    for j in range(degree):
        t[j] = min_k - d * (degree - j)

    t[degree: degree + n_knots] = knots

    for j in range(degree):
        t[degree + n_knots + j] = max_k + d * (j + 1)

    return t


# ---------------------------------------------------------------------------
# Jump matrix (penalty structure)
# ---------------------------------------------------------------------------

def jump_bsplines(t, degree):
    """Compute the jump matrix for the B-spline penalty.

    Captures discontinuities in the d-th derivative at interior knots.
    Corresponds to ``jump_bsplines()`` in APRS.cpp.

    Parameters
    ----------
    t : ndarray
        Full knot sequence.
    degree : int
        B-spline degree.

    Returns
    -------
    jump : ndarray of shape (J, J - order)
        Jump matrix.
    """
    t = np.asarray(t, dtype=float)
    order = degree + 1
    dimension = len(t) - order
    n_penalty = dimension - order

    if n_penalty <= 0:
        return np.zeros((dimension, 0))

    jump = np.zeros((dimension, n_penalty))

    # Midpoints between adjacent interior knots
    x_mid = np.array([
        0.5 * (t[j + order - 1] + t[j + order])
        for j in range(n_penalty + 1)
    ])

    for j in range(dimension):
        deriv = _bspline_vec(x_mid, t, degree, j, derivative=degree)
        for l in range(n_penalty):
            jump[j, l] = deriv[l + 1] - deriv[l]

    return jump
