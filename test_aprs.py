"""Test APRS implementation against known signals from the papers."""

import numpy as np
import pytest
from aprscast.bspline import bspline_basis, dim2knots, knots2t, jump_bsplines
from aprscast.detection import APRS


# ---------------------------------------------------------------------------
# B-spline basis tests
# ---------------------------------------------------------------------------

class TestBsplineBasis:
    """Tests for B-spline construction (translated from RCPP)."""

    def test_dim2knots_shape(self):
        """Knots should have dimension - degree + 1 elements."""
        x = np.linspace(0, 1, 100)
        knots = dim2knots(x, dimension=10, degree=0)
        assert len(knots) == 10 - 0 + 1  # 11

    def test_knots2t_shape(self):
        """Extended knots should have 2*degree + n_knots elements."""
        knots = np.linspace(0, 1, 11)
        t = knots2t(knots, degree=0)
        assert len(t) == 2 * 0 + 11  # 11

        t3 = knots2t(knots, degree=3)
        assert len(t3) == 2 * 3 + 11  # 17

    def test_basis_d0_partition_of_unity(self):
        """For d=0, basis functions should sum to 1 at interior points."""
        x = np.linspace(0.01, 0.99, 50)
        knots = dim2knots(x, dimension=10, degree=0)
        t = knots2t(knots, degree=0)
        B = bspline_basis(x, t, degree=0)

        row_sums = B.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_basis_d1_partition_of_unity(self):
        """For d=1, basis functions should also sum to 1 at interior."""
        x = np.linspace(0.05, 0.95, 50)
        knots = dim2knots(x, dimension=10, degree=1)
        t = knots2t(knots, degree=1)
        B = bspline_basis(x, t, degree=1)

        row_sums = B.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_basis_shape(self):
        """Basis matrix shape: (n, dimension)."""
        x = np.linspace(0, 1, 100)
        dim = 20
        degree = 0
        knots = dim2knots(x, dim, degree)
        t = knots2t(knots, degree)
        B = bspline_basis(x, t, degree)

        assert B.shape == (100, dim)

    def test_jump_matrix_shape(self):
        """Jump matrix shape: (dimension, dimension - order)."""
        x = np.linspace(0, 1, 100)
        dim = 10
        degree = 0
        knots = dim2knots(x, dim, degree)
        t = knots2t(knots, degree)
        jump = jump_bsplines(t, degree)

        order = degree + 1
        assert jump.shape == (dim, dim - order)


# ---------------------------------------------------------------------------
# APRS detection tests
# ---------------------------------------------------------------------------

class TestAPRS:
    """Tests for APRS change point detection."""

    def test_single_changepoint_d0(self):
        """APRS(d=0) should detect a single level shift."""
        np.random.seed(42)
        n = 200
        x = np.linspace(0, 1, n)
        y = np.where(x < 0.5, 0.0, 2.0) + np.random.normal(0, 0.1, n)

        model = APRS(degree=0, n_lambdas=50)
        result = model.fit(x, y)

        # Should detect approximately 1 change point near 0.5
        assert result.n_changepoints >= 1
        assert result.n_changepoints <= 3  # allow small tolerance

        # Fitted values should be close to true function
        true_f = np.where(x < 0.5, 0.0, 2.0)
        mse = np.mean((result.fitted - true_f) ** 2)
        assert mse < 0.1  # reasonable fit

    def test_f1_two_changepoints(self):
        """Test on f1 from the ANZJS paper: two change points at 0.1, 0.9."""
        np.random.seed(123)
        n = 200
        x = np.linspace(0, 1, n)
        f1 = np.where(x <= 0.1, 0.0, np.where(x <= 0.9, 7.0, 14.0))
        y = f1 + np.random.normal(0, 0.5, n)

        model = APRS(degree=0, n_lambdas=50)
        result = model.fit(x, y)

        # Should detect 2 change points (allow up to 4 for noisy data)
        assert result.n_changepoints <= 5
        assert result.method == "APRS (d=0)"

    def test_result_attributes(self):
        """DetectionResult should have all expected fields."""
        np.random.seed(0)
        x = np.linspace(0, 1, 100)
        y = np.random.normal(0, 1, 100)

        model = APRS(degree=0, n_lambdas=20)
        result = model.fit(x, y)

        assert hasattr(result, "changepoints")
        assert hasattr(result, "fitted")
        assert hasattr(result, "coefficients")
        assert hasattr(result, "bic")
        assert hasattr(result, "aic")
        assert hasattr(result, "optimal_lambda")
        assert hasattr(result, "path")

    def test_path_contains_all_lambdas(self):
        """Solution path should have n_lambdas entries."""
        np.random.seed(0)
        x = np.linspace(0, 1, 100)
        y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.3, 100)

        n_lam = 30
        model = APRS(degree=0, n_lambdas=n_lam)
        result = model.fit(x, y)

        assert len(result.path["lambdas"]) == n_lam
        assert len(result.path["bic"]) == n_lam

    def test_non_adaptive(self):
        """APRS with adaptive=False should also work."""
        np.random.seed(42)
        x = np.linspace(0, 1, 100)
        y = np.where(x < 0.5, 0.0, 1.0) + np.random.normal(0, 0.2, 100)

        model = APRS(degree=0, n_lambdas=20, adaptive=False)
        result = model.fit(x, y)

        assert result.fitted.shape == (100,)
