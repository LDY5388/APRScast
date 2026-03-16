"""
Change point detection via APRS (Adaptive Penalized Regression Spline).

Translated from APRS.cpp (Rcpp implementation) to pure NumPy.
Implements the coordinate descent algorithm with pathwise optimization.

References
----------
- Lee, D.-Y., Bak, K.-Y., & Jhong, J.-H. (2024). ANZJS.
- Lee, D.-Y., Bak, K.-Y., & Jhong, J.-H. (2025). Generalized APRS.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from aprscast.bspline import (
    bspline_basis, dim2knots, knots2t, jump_bsplines,
)

TINY = 1e-30
EPSILON = 1e-30


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Result of change point detection.

    Attributes
    ----------
    changepoints : ndarray
        Detected change point locations.
    fitted : ndarray
        Fitted values at the observation points.
    coefficients : ndarray
        Estimated B-spline coefficients (at optimal lambda).
    knot_sequence : ndarray
        Final knot sequence after pruning.
    n_changepoints : int
        Number of detected change points.
    bic : float
        BIC value at the selected model.
    aic : float
        AIC value at the selected model.
    optimal_lambda : float
        Selected regularization parameter.
    method : str
        Name of the method used.
    path : dict
        Full solution path (all lambda results).
    """
    changepoints: np.ndarray
    fitted: np.ndarray
    coefficients: np.ndarray
    knot_sequence: np.ndarray
    n_changepoints: int
    bic: float
    aic: float
    optimal_lambda: float
    method: str
    path: dict

    def summary(self):
        """Print a summary of detection results."""
        print(f"Method:           {self.method}")
        print(f"Change points:    {self.n_changepoints}")
        print(f"BIC:              {self.bic:.4f}")
        print(f"AIC:              {self.aic:.4f}")
        print(f"Optimal lambda:   {self.optimal_lambda:.6f}")
        if self.n_changepoints > 0:
            print(f"Locations:        {self.changepoints}")


# ---------------------------------------------------------------------------
# Coordinate descent helper functions
# ---------------------------------------------------------------------------

def _q_lambda_point(z, a, b, c, d, lam):
    """Evaluate q_lambda at a single candidate z.

    q_lambda = (b/2)(z - c)^2 + lambda * sum_k d_k |z - a_k|

    Corresponds to ``q_lambda_point()`` in APRS.cpp.
    """
    return 0.5 * b * (z - c) ** 2 + lam * np.sum(d * np.abs(z - a))


def _delta(d):
    """Compute delta vector from ordered d weights.

    Corresponds to ``delta()`` in APRS.cpp.
    """
    size = len(d)
    delta_vec = np.zeros(size + 1)
    sum_d = np.sum(d)

    delta_vec[0] = -sum_d
    delta_vec[1] = sum_d

    cumsum = 0.0
    for j in range(size - 1):
        cumsum += d[j]
        delta_vec[j + 2] = 2 * cumsum - sum_d

    return delta_vec


def _zlambda(a, b, c, d, lam):
    """Find the minimizer of the penalized univariate objective.

    Solves: min_z (b/2)(z-c)^2 + lambda * sum d_k |z - a_k|

    Corresponds to ``zlambda()`` in APRS.cpp.
    """
    # Sort a and reorder d accordingly
    order = np.argsort(a)
    ordered_d = d[order]
    delta_d = _delta(ordered_d)

    # Enumerate candidates
    z_candidates = np.concatenate([
        c - lam * delta_d / b,
        a,
    ])
    z_candidates = np.sort(z_candidates)

    # Find zstar (exploit convexity — stop at first increase)
    zstar = z_candidates[0]
    q_best = _q_lambda_point(z_candidates[0], a, b, c, d, lam)

    for i in range(1, len(z_candidates)):
        q_val = _q_lambda_point(z_candidates[i], a, b, c, d, lam)
        if q_val < q_best:
            zstar = z_candidates[i]
            q_best = q_val
        else:
            break  # convex — no need to continue

    return zstar


# ---------------------------------------------------------------------------
# Lambda path computation
# ---------------------------------------------------------------------------

def _compute_lambda_max(response, predictors, knots, degree, dimension,
                        weight):
    """Compute lambda_max via KKT conditions (Theorem 4.1).

    Corresponds to ``lambdas_all()`` lambda_max computation in APRS.cpp.
    """
    order = degree + 1
    K = order + 1
    n_penalty = dimension - order

    original_t = knots2t(knots, degree)

    # Build minimal 2-segment knot sequence
    t_local = np.zeros(2 * order + 1)
    t_local[:order] = original_t[:order]

    idx = len(original_t) - 1
    for j in range(2 * order, order, -1):
        t_local[j] = original_t[idx]
        idx -= 1

    lambda_max_candidates = np.zeros(n_penalty)

    for k in range(n_penalty):
        t_local[order] = knots[k + 1]

        basis = bspline_basis(predictors, t_local, degree)
        BY = basis.T @ response
        BB = basis.T @ basis

        BB_inv = np.linalg.inv(BB)
        u_mat = jump_bsplines(t_local, degree)
        u_vec = u_mat[:, 0]

        uBB_inv = u_vec @ BB_inv
        A = uBB_inv @ BY
        B_val = uBB_inv @ u_vec

        lambda_max_candidates[k] = np.abs(A / B_val) / weight[k]

    return np.max(lambda_max_candidates) + EPSILON


def _build_lambda_path(lambda_max, n_lambdas, epsilon_lambda, degree):
    """Build log-spaced lambda path from lambda_min to lambda_max.

    Corresponds to the lambda path construction in ``lambdas_all()``.
    """
    d3 = degree * 3
    eps_adj = epsilon_lambda * (10.0 ** (-d3))
    lambda_min = eps_adj * lambda_max
    ratio = 1.0 / eps_adj

    lambdas = np.zeros(n_lambdas)
    div = n_lambdas - 1
    for i in range(n_lambdas):
        exponent = i / div
        lambdas[i] = lambda_min * (ratio ** exponent)

    return lambdas


# ---------------------------------------------------------------------------
# Main APRS class
# ---------------------------------------------------------------------------

class APRS:
    """Adaptive Penalized Regression Spline for change point detection.

    Parameters
    ----------
    degree : int, default=0
        B-spline degree. 0=piecewise constant, 1=linear, 2=quadratic, 3=cubic.
    n_lambdas : int, default=100
        Number of regularization parameters in the path.
    adaptive : bool, default=True
        Whether to use adaptive weights. If False, uniform weights are used.
    monotone : str, default="none"
        Monotonicity constraint: "none", "increasing", "decreasing".
    epsilon_lambda : float, default=1e-4
        Controls lambda_min / lambda_max ratio.
    maxiter : int, default=1000
        Maximum coordinate descent iterations per lambda.
    tol : float, default=1e-5
        Convergence tolerance for objective function.
    criterion : str, default="bic"
        Model selection criterion ("bic" or "aic").

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 200)
    >>> y = np.where(x < 0.5, 0.0, 2.0) + np.random.normal(0, 0.3, 200)
    >>> model = APRS(degree=0)
    >>> result = model.fit(x, y)
    >>> result.summary()
    """

    def __init__(
        self,
        degree: int = 0,
        n_lambdas: int = 100,
        adaptive: bool = True,
        monotone: str = "none",
        epsilon_lambda: float = 1e-4,
        maxiter: int = 1000,
        tol: float = 1e-5,
        criterion: str = "bic",
    ):
        self.degree = degree
        self.n_lambdas = n_lambdas
        self.adaptive = adaptive
        self.monotone = monotone
        self.epsilon_lambda = epsilon_lambda
        self.maxiter = maxiter
        self.tol = tol
        self.criterion = criterion

    def fit(self, x, y, dimension=None) -> DetectionResult:
        """Fit APRS to observed data.

        Parameters
        ----------
        x : array-like of shape (n,)
            Input points (time). Must be ordered.
        y : array-like of shape (n,)
            Observed response values.
        dimension : int or None, default=None
            Number of initial basis functions. If None, uses n // 2.

        Returns
        -------
        DetectionResult
            Full detection results with solution path.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)

        if dimension is None:
            dimension = n // 2

        degree = self.degree
        order = degree + 1
        n_penalty = dimension - order

        # ------ Knots & Basis ------
        knots = dim2knots(x, dimension, degree)
        t = knots2t(knots, degree)
        basis = bspline_basis(x, t, degree)
        jump = jump_bsplines(t, degree)

        # ------ Step 1: Compute LSE (initial unpenalized fit) ------
        beta = np.zeros(dimension)
        residuals = y.copy()
        store_obj = np.inf

        for iteration in range(self.maxiter):
            for j in range(dimension):
                bj = basis[:, j]
                partial_res = residuals + beta[j] * bj
                b_val = np.dot(bj, bj)
                c_val = np.dot(bj, partial_res) / (b_val + TINY)
                beta[j] = c_val
                residuals = partial_res - beta[j] * bj

            fitted = basis @ beta
            residuals = y - fitted
            obj = 0.5 * np.dot(residuals, residuals)

            if abs(obj - store_obj) < self.tol:
                break
            store_obj = obj

        # ------ Step 2: Adaptive weights ------
        weight = np.ones(n_penalty)
        if self.adaptive and n_penalty > 0:
            for k in range(n_penalty):
                val = np.dot(jump[:, k], beta)
                weight[k] = 1.0 / abs(val) if abs(val) > EPSILON else 1.0 / EPSILON
            weight = np.abs(weight)

        # ------ Step 3: Lambda path ------
        lambda_max = _compute_lambda_max(
            y, x, knots, degree, dimension, weight
        )
        lambdas = _build_lambda_path(
            lambda_max, self.n_lambdas, self.epsilon_lambda, degree
        )

        # ------ Step 4: Pathwise coordinate descent ------
        bic_vec = np.zeros(self.n_lambdas)
        aic_vec = np.zeros(self.n_lambdas)
        dim_vec = np.zeros(self.n_lambdas, dtype=int)
        path_results = []

        for li in range(self.n_lambdas):
            lam = lambdas[li]

            # Current working copies (may be pruned)
            cur_beta = beta.copy()
            cur_t = t.copy()
            cur_basis = basis.copy()
            cur_jump = jump.copy() if n_penalty > 0 else np.zeros((dimension, 0))
            cur_weight = weight.copy()
            cur_dim = len(cur_beta)
            cur_n_penalty = cur_dim - order
            store_obj = np.inf

            for iteration in range(self.maxiter):
                # --- Coordinate descent update ---
                for j in range(cur_dim):
                    bj = cur_basis[:, j]
                    partial_res = residuals + cur_beta[j] * bj
                    b_val = np.dot(bj, bj)
                    c_val = np.dot(bj, partial_res) / (b_val + TINY)

                    if cur_dim == order:
                        cur_beta[j] = c_val
                    else:
                        rowjump_j = cur_jump[j, :]
                        nonzero_mask = np.abs(rowjump_j) > EPSILON
                        d_vals = rowjump_j[nonzero_mask]
                        w_vals = cur_weight[nonzero_mask]

                        if len(d_vals) == 0:
                            cur_beta[j] = c_val
                        else:
                            # Compute a vector
                            nonzero_idx = np.where(nonzero_mask)[0]
                            a_vals = np.zeros(len(d_vals))
                            for kk in range(len(d_vals)):
                                col_sum = np.dot(cur_jump[:, nonzero_idx[kk]],
                                                 cur_beta)
                                a_vals[kk] = cur_beta[j] - col_sum / d_vals[kk]

                            d_abs = np.abs(d_vals)
                            cur_beta[j] = _zlambda(
                                a_vals, b_val, c_val, d_abs * w_vals, lam
                            )

                    # Monotonicity constraint
                    if self.monotone == "increasing" and j > 0:
                        if cur_beta[j] < cur_beta[j - 1]:
                            cur_beta[j] = cur_beta[j - 1]
                    elif self.monotone == "decreasing" and j > 0:
                        if cur_beta[j] > cur_beta[j - 1]:
                            cur_beta[j] = cur_beta[j - 1]

                    residuals = partial_res - cur_beta[j] * bj

                # --- Prune zero-penalty knots ---
                if cur_n_penalty > 0:
                    penalty_vals = np.array([
                        np.dot(cur_jump[:, k], cur_beta)
                        for k in range(cur_n_penalty)
                    ])
                    zero_mask = np.abs(penalty_vals) < EPSILON

                    if np.any(zero_mask):
                        prune_idx = np.where(zero_mask)[0]

                        # Remove corresponding beta, t, weight entries
                        keep_beta = np.ones(cur_dim, dtype=bool)
                        keep_beta[prune_idx] = False

                        keep_t = np.ones(len(cur_t), dtype=bool)
                        keep_t[prune_idx + order] = False

                        keep_w = np.ones(cur_n_penalty, dtype=bool)
                        keep_w[prune_idx] = False

                        cur_beta = cur_beta[keep_beta]
                        cur_t = cur_t[keep_t]
                        cur_weight = cur_weight[keep_w]

                        cur_dim = len(cur_beta)
                        cur_n_penalty = cur_dim - order
                        cur_basis = bspline_basis(x, cur_t, degree)

                        if cur_n_penalty > 0:
                            cur_jump = jump_bsplines(cur_t, degree)
                        else:
                            cur_jump = np.zeros((cur_dim, 0))

                # --- Update fitted values & check convergence ---
                fitted = cur_basis @ cur_beta
                residuals = y - fitted

                R = 0.5 * np.dot(residuals, residuals)
                obj = R
                for k in range(cur_n_penalty):
                    obj += lam * abs(np.dot(cur_jump[:, k], cur_beta)
                                     * cur_weight[k])

                if abs(obj - store_obj) < self.tol:
                    break
                store_obj = obj

            # --- Store results for this lambda ---
            dim_vec[li] = cur_dim
            NlogR = n * np.log(2.0 * R / n) if R > 0 else -np.inf
            bic_vec[li] = NlogR + cur_dim * np.log(n)
            aic_vec[li] = NlogR + cur_dim * 2.0

            path_results.append({
                "lambda": lam,
                "dimension": cur_dim,
                "beta": cur_beta.copy(),
                "fitted": fitted.copy(),
                "t": cur_t.copy(),
            })

            # Warm start: carry forward for next lambda
            beta = cur_beta
            t = cur_t
            basis = cur_basis
            jump = cur_jump
            weight = cur_weight
            dimension = cur_dim
            n_penalty = cur_n_penalty

        # ------ Step 5: Model selection ------
        if self.criterion == "bic":
            best_idx = np.argmin(bic_vec)
        else:
            best_idx = np.argmin(aic_vec)

        best = path_results[best_idx]

        # Extract change point locations from the knot sequence
        best_t = best["t"]
        interior_knots = best_t[order: order + best["dimension"] - order]

        return DetectionResult(
            changepoints=interior_knots,
            fitted=best["fitted"],
            coefficients=best["beta"],
            knot_sequence=best["t"],
            n_changepoints=len(interior_knots),
            bic=bic_vec[best_idx],
            aic=aic_vec[best_idx],
            optimal_lambda=best["lambda"],
            method=f"APRS (d={degree})",
            path={
                "lambdas": lambdas,
                "bic": bic_vec,
                "aic": aic_vec,
                "dimensions": dim_vec,
                "results": path_results,
            },
        )


# ---------------------------------------------------------------------------
# Comparison interface
# ---------------------------------------------------------------------------

def compare_methods(x, y, methods=None, degree=1, **kwargs):
    """Compare APRS against other change point detection methods.

    Parameters
    ----------
    x : array-like of shape (n,)
        Input points.
    y : array-like of shape (n,)
        Observed values.
    methods : list of str, optional
        Methods to compare. Default: ["aprs"].
    degree : int, default=1
        B-spline degree.

    Returns
    -------
    dict
        Mapping of method names to DetectionResult objects.
    """
    if methods is None:
        methods = ["aprs"]

    results = {}

    for method in methods:
        if method == "aprs":
            model = APRS(degree=degree, **kwargs)
            results["aprs"] = model.fit(x, y)
        # Future: add "trend_filtering", "pelt", "not"
        else:
            raise NotImplementedError(
                f"Method '{method}' not yet implemented. "
                f"Available: ['aprs']"
            )

    return results
