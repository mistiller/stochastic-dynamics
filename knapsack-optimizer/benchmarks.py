"""
Benchmark suite for Knapsack solver performance tracking
"""
import time
import argparse
import numpy as np
import pandas as pd
from loguru import logger
from knapsack_optimizer import KnapsackOptimizer
from unistochastic_knapsack_solver import UnistochasticKnapsackSolver

def run_scaling_benchmark(max_items: int = 50, runs_per_size: int = 10, 
                         solver_type: str = "knapsack") -> pd.DataFrame:
    """Run comprehensive performance benchmarks across problem sizes and solvers."""
    results = []
    
    for n_items in range(10, max_items + 1):
        start_time = time.time()
        logger.info(f"Running {solver_type} benchmarks for {n_items} items...")
        
        # Generate test instance
        values = np.random.randint(1, 100, size=n_items)
        weights = np.random.randint(1, 50, size=n_items)
        capacity = max(np.sum(weights) // 2, np.min(weights))
        
        # Initialize appropriate solver
        if solver_type == "unistochastic":
            solver = UnistochasticKnapsackSolver(values, weights, capacity)
        else:  # default to knapsack optimizer
            solver = KnapsackOptimizer(values, weights, capacity)
        
        # Run and time the solver
        try:
            solver.solve()
            solve_time = time.time() - start_time
            # Handle different solver types
            is_unistochastic = isinstance(solver, UnistochasticKnapsackSolver)
            results.append({
                'items': n_items,
                'solver_type': solver_type,
                'time': solve_time,
                'valid_solutions': solver.best_solution.sum() if is_unistochastic else solver.valid_mask.sum(),
                'agreement_rate': 1.0 if is_unistochastic else solver.compare_solvers_scaling(runs_per_size=1)['agreement_rate'].iloc[0]
            })
        except Exception as e:
            logger.error(f"Failed benchmark for {n_items} items: {str(e)}")
        
        logger.info(f"Completed {n_items} items in {time.time()-start_time:.2f}s")
    
    return pd.DataFrame(results)

def save_benchmark_results(df: pd.DataFrame, filename: str = "results/benchmark_results.csv"):
    """Save benchmark results to CSV with timestamp."""
    df['timestamp'] = pd.Timestamp.now()
    df.to_csv(filename, index=False)
    logger.info(f"Saved benchmark results to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run knapsack solver benchmarks')
    parser.add_argument('--solver-type', type=str, default='knapsack',
                       choices=['knapsack', 'unistochastic'],
                       help='Type of solver to benchmark (knapsack|unistochastic)')
    parser.add_argument('--max-items', type=int, default=50,
                       help='Maximum number of items to test')
    args = parser.parse_args()

    benchmark_df = run_scaling_benchmark(
        max_items=args.max_items,
        solver_type=args.solver_type
    )
    save_benchmark_results(benchmark_df)
