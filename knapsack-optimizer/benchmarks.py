"""
Benchmark suite for KnapsackOptimizer performance tracking
"""
import time
import numpy as np
import pandas as pd
from loguru import logger
from knapsack_optimizer import KnapsackOptimizer

def run_scaling_benchmark(max_items: int = 50, runs_per_size: int = 10) -> pd.DataFrame:
    """Run comprehensive performance benchmarks across problem sizes."""
    results = []
    
    for n_items in range(3, max_items + 1):
        start_time = time.time()
        logger.info(f"Running benchmarks for {n_items} items...")
        
        # Generate test instance
        values = np.random.randint(1, 100, size=n_items)
        weights = np.random.randint(1, 50, size=n_items)
        capacity = max(np.sum(weights) // 2, np.min(weights))
        
        # Initialize optimizer with adaptive params
        ko = KnapsackOptimizer(values, weights, capacity)
        
        # Run and time the solver
        try:
            ko.solve()
            solve_time = time.time() - start_time
            results.append({
                'items': n_items,
                'time': solve_time,
                'valid_solutions': ko.valid_mask.sum(),
                'agreement_rate': ko.compare_solvers_scaling(runs_per_size=1)['agreement_rate'].iloc[0]
            })
        except Exception as e:
            logger.error(f"Failed benchmark for {n_items} items: {str(e)}")
        
        logger.info(f"Completed {n_items} items in {time.time()-start_time:.2f}s")
    
    return pd.DataFrame(results)

def save_benchmark_results(df: pd.DataFrame, filename: str = "benchmark_results.csv"):
    """Save benchmark results to CSV with timestamp."""
    df['timestamp'] = pd.Timestamp.now()
    df.to_csv(filename, index=False)
    logger.info(f"Saved benchmark results to {filename}")

if __name__ == "__main__":
    benchmark_df = run_scaling_benchmark()
    save_benchmark_results(benchmark_df)
