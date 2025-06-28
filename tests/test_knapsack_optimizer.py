"""
Test suite for KnapsackOptimizer class in knapsack_solver.knapsack_optimizer
"""

import pytest
import numpy as np
from knapsack_solver.knapsack_optimizer import KnapsackOptimizer

# Define test cases using known values

DUMMY_VALUES = [505, 352, 458, 220, 354, 414, 498, 545, 473, 543]
DUMMY_WEIGHTS = [23, 26, 20, 18, 32, 27, 29, 26, 30, 27]
DUMMY_CAPACITY = 67

# Expected known results from greedy and dynamic programming approaches
EXPECTED_GREEDY_VALUE = 1515.0
EXPECTED_GREEDY_WEIGHT = 63.0
EXPECTED_GREEDY_ITEMS = [3, 2, 0, 7, 5]

EXPECTED_DP_VALUE = 1544.0
EXPECTED_DP_WEIGHT = 67.0
EXPECTED_DP_ITEMS = [0, 2, 3, 7, 9]

class TestKnapsackOptimizer:
    """
    Test class for KnapsackOptimizer
    """

    @pytest.fixture
    def optimizer(self):
        """Fixture to initialize a KnapsackOptimizer instance"""
        return KnapsackOptimizer(DUMMY_VALUES, DUMMY_WEIGHTS, DUMMY_CAPACITY)

    def test_init_validation(self):
        """Test that initialization raises error when values and weights have different lengths"""
        with pytest.raises(ValueError):
            KnapsackOptimizer([1, 2], [1], DUMMY_CAPACITY)

    def test_greedy_solver(self, optimizer):
        """Test greedy solver against known expected values"""
        items, value, weight = optimizer.greedy_solver()
        assert value == EXPECTED_GREEDY_VALUE
        assert weight == EXPECTED_GREEDY_WEIGHT
        assert sorted(items) == sorted(EXPECTED_GREEDY_ITEMS)

    def test_dynamic_programming_solver(self, optimizer):
        """Test dynamic programming solver against known expected values"""
        items, value, weight = optimizer.dynamic_programming_solver()
        assert value == EXPECTED_DP_VALUE
        assert weight == EXPECTED_DP_WEIGHT
        assert sorted(items) == sorted(EXPECTED_DP_ITEMS)

    def test_solve(self, optimizer):
        """Test MCMC-based solve returns valid solution"""
        solution = optimizer.solve(draws=500, tune=250, chains=2)
        assert isinstance(solution, np.ndarray)
        assert solution.dtype == bool
        assert len(solution) == len(DUMMY_VALUES)

    def test_summary(self, optimizer, capsys):
        """Test summary method output contains expected keys"""
        optimizer.solve(draws=500, tune=250, chains=2)
        optimizer.summary(include_baseline=True)
        captured = capsys.readouterr().out

        assert "Path Integral Optimization Results:" in captured
        assert "Greedy Algorithm Results:" in captured
        assert "Dynamic Programming Results:" in captured
        assert "Optimal Value" in captured
        assert "Total Weight" in captured

    def test_plot_results(self, optimizer):
        """Test plot_results runs without error"""
        optimizer.solve(draws=500, tune=250, chains=2)
        try:
            optimizer.plot_results()
        except Exception as e:
            pytest.fail(f"plot_results() raised {e} unexpectedly!")

    def test_fptas_solver(self, optimizer):
        """Test FPTAS solver with known values"""
        items, value, weight = optimizer.fptas_solver(epsilon=0.1)
        assert value <= EXPECTED_DP_VALUE * (1 + 0.1)
        assert weight <= optimizer.capacity
        assert all(isinstance(i, int) for i in items)
