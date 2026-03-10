"""Tests for inequality measures against analytically known values."""

import numpy as np
import pytest
from core.inequality_measures import (
    gini,
    mld,
    theil_t,
    var_logs,
    atkinson,
    compute_inequality,
)


class TestGini:
    def test_perfect_equality(self):
        """Gini of identical incomes should be 0."""
        y = np.array([100.0, 100.0, 100.0, 100.0])
        assert gini(y) == pytest.approx(0.0, abs=1e-10)

    def test_two_person_economy(self):
        """Gini of [1, 3]: |1-3|/(2*2*2) = 0.25 (population Gini)."""
        y = np.array([1.0, 3.0])
        # Population Gini = sum|yi-yj| / (2 * n^2 * mean) = 2 / (2*4*2) = 0.25
        assert gini(y) == pytest.approx(0.25, abs=1e-10)

    def test_known_distribution(self):
        """Gini of [1, 2, 3, 4, 5] = 0.2667 (analytical)."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Analytical: sum|yi - yj| / (2 * n * n * mean)
        n = len(y)
        mu = y.mean()
        abs_diffs = sum(abs(yi - yj) for yi in y for yj in y)
        expected = abs_diffs / (2 * n * n * mu)
        assert gini(y) == pytest.approx(expected, abs=1e-6)

    def test_weighted_gini(self):
        """Weighted Gini should adjust for weights."""
        y = np.array([1.0, 2.0])
        w = np.array([3.0, 1.0])  # 3 people with 1, 1 with 2
        result = gini(y, weights=w)
        # Should be less than unweighted gini([1,2])
        assert 0 < result < gini(y)

    def test_non_negative(self):
        """Gini should be non-negative for positive incomes."""
        rng = np.random.default_rng(42)
        y = rng.exponential(scale=1000, size=100)
        assert gini(y) >= 0

    def test_less_than_one(self):
        """Gini should be < 1 for non-degenerate distributions."""
        rng = np.random.default_rng(42)
        y = rng.exponential(scale=1000, size=100)
        assert gini(y) < 1.0


class TestMLD:
    def test_perfect_equality(self):
        y = np.array([5.0, 5.0, 5.0, 5.0])
        assert mld(y) == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        """MLD = log(mean) - mean(log)."""
        y = np.array([1.0, 2.0, 4.0, 8.0])
        expected = np.log(y.mean()) - np.log(y).mean()
        assert mld(y) == pytest.approx(expected, abs=1e-10)

    def test_non_negative(self):
        """MLD is always >= 0."""
        rng = np.random.default_rng(42)
        y = rng.exponential(scale=1000, size=100)
        assert mld(y) >= -1e-10  # numerical tolerance

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError, match="strictly positive"):
            mld(np.array([0.0, 1.0, 2.0]))


class TestTheilT:
    def test_perfect_equality(self):
        y = np.array([10.0, 10.0, 10.0])
        assert theil_t(y) == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        """T = mean( (y/mu) * log(y/mu) )."""
        y = np.array([1.0, 2.0, 3.0])
        mu = y.mean()
        ratios = y / mu
        expected = (ratios * np.log(ratios)).mean()
        assert theil_t(y) == pytest.approx(expected, abs=1e-10)

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        y = rng.exponential(scale=1000, size=100)
        assert theil_t(y) >= -1e-10


class TestVarLogs:
    def test_perfect_equality(self):
        y = np.array([7.0, 7.0, 7.0])
        assert var_logs(y) == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        y = np.array([1.0, np.e, np.e**2])
        log_y = np.log(y)  # [0, 1, 2]
        expected = np.var(log_y)  # population variance
        assert var_logs(y) == pytest.approx(expected, abs=1e-10)


class TestAtkinson:
    def test_perfect_equality(self):
        y = np.array([5.0, 5.0, 5.0])
        assert atkinson(y, epsilon=0.5) == pytest.approx(0.0, abs=1e-10)
        assert atkinson(y, epsilon=1.0) == pytest.approx(0.0, abs=1e-10)
        assert atkinson(y, epsilon=2.0) == pytest.approx(0.0, abs=1e-10)

    def test_epsilon_1_is_geometric_ratio(self):
        """A(eps=1) = 1 - geometric_mean / arithmetic_mean."""
        y = np.array([1.0, 4.0, 16.0])
        geo_mean = np.exp(np.log(y).mean())
        arith_mean = y.mean()
        expected = 1.0 - geo_mean / arith_mean
        assert atkinson(y, epsilon=1.0) == pytest.approx(expected, abs=1e-10)

    def test_bounded_0_1(self):
        """Atkinson is in [0, 1) for positive incomes."""
        rng = np.random.default_rng(42)
        y = rng.exponential(scale=1000, size=100)
        for eps in [0.5, 1.0, 2.0]:
            val = atkinson(y, epsilon=eps)
            assert 0 <= val < 1.0

    def test_higher_epsilon_more_sensitive(self):
        """Higher epsilon gives more weight to bottom, so higher inequality."""
        rng = np.random.default_rng(42)
        y = rng.exponential(scale=1000, size=100)
        a05 = atkinson(y, epsilon=0.5)
        a1 = atkinson(y, epsilon=1.0)
        a2 = atkinson(y, epsilon=2.0)
        assert a05 < a1 < a2


class TestDispatcher:
    def test_all_measures_work(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        measures = ["gini", "mld", "theil_t", "var_logs", "atkinson_0.5", "atkinson_1", "atkinson_2"]
        for m in measures:
            result = compute_inequality(y, m)
            assert isinstance(result, float)
            assert np.isfinite(result)

    def test_unknown_measure_raises(self):
        with pytest.raises(ValueError, match="Unknown measure"):
            compute_inequality(np.array([1.0, 2.0]), "unknown")


class TestEdgeCases:
    def test_minimum_observations(self):
        with pytest.raises(ValueError, match="at least 2"):
            gini(np.array([1.0]))

    def test_large_array_performance(self):
        """Sanity check: should handle 100k obs without issues."""
        rng = np.random.default_rng(42)
        y = rng.exponential(scale=1000, size=100_000)
        g = gini(y)
        assert 0 < g < 1

    def test_weighted_vs_repeated(self):
        """Weighted [1,2] with w=[3,1] should approximate gini([1,1,1,2])."""
        y_weighted = np.array([1.0, 2.0])
        w = np.array([3.0, 1.0])
        y_repeated = np.array([1.0, 1.0, 1.0, 2.0])
        assert gini(y_weighted, weights=w) == pytest.approx(gini(y_repeated), abs=0.05)
