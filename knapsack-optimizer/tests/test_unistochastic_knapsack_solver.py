import pytest
import numpy as np
from unistochastic_knapsack_solver import UnistochasticKnapsackSolver

DUMMY_VALUES = [505, 352, 458, 220, 354, 414, 498, 545, 473, 543]
DUMMY_WEIGHTS = [23, 26, 20, 18, 32, 27, 29, 26, 30, 27]
DUMMY_CAPACITY = 67

class TestUnistochasticKnapsackSolver:
    """Test class for UnistochasticKnapsackSolver"""

    @pytest.fixture
    def solver(self):
        return UnistochasticKnapsackSolver(DUMMY_VALUES, DUMMY_WEIGHTS, DUMMY_CAPACITY)

    def test_init_validation(self):
        with pytest.raises(ValueError):
            UnistochasticKnapsackSolver([1,2], [3], 5)

    def test_solve(self, solver):
        solution = solver.solve(samples=100)
        assert len(solution) == len(DUMMY_VALUES)
        assert solution.dtype == bool
        assert solver._calculate_weight(solution) <= DUMMY_CAPACITY

    def test_summary(self, solver, capsys):
        solver.solve(samples=100)
        solver.summary(include_baseline=False)
        captured = capsys.readouterr()
        assert "Optimal Value" in captured.out
        assert "Total Weight" in captured.out

    def test_plot_results(self, solver):
        solver.solve(samples=100)
        solver.plot_results()  # Just test that it runs without errors

    def test_edge_cases(self):
        # Test empty case
        solver = UnistochasticKnapsackSolver([], [], 0)
        with pytest.raises(RuntimeError):
            solver.solve()

        # Test single item that fits
        solver = UnistochasticKnapsackSolver([10], [5], 5)
        solution = solver.solve()
        assert solution[0] == True

        # Test single item that doesn't fit
        solver = UnistochasticKnapsackSolver([10], [6], 5)
        with pytest.raises(RuntimeError):
            solver.solve()
