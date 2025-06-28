from typing import Final
from datetime import datetime, timezone
from knapsack_optimizer import KnapsackOptimizer

MAX_ITEMS:Final[int]=20
RUNS_PER_SIZE:Final[int]=10

if __name__ == "__main__":
    # Run scaling comparison simulation
    dummy_values = [1]  # Values will be generated in the simulation
    dummy_weights = [1] # Weights will be generated in the simulation
    
    solver = KnapsackOptimizer(dummy_values, dummy_weights, capacity=1, hbar=0.5)
    results = solver.compare_solvers_scaling(max_items=MAX_ITEMS, runs_per_size=RUNS_PER_SIZE)
    
    # Print results dataframe
    print("\nSolver Scaling Results DataFrame:")
    print(results.to_string(index=False, formatters={
        'agreement_rate': '{:.0%}'.format,
        'avg_time': '{:.2f}s'.format,
        'max_time': '{:.2f}s'.format,
        'avg_percent_diff': '{:.1f}%'.format
    }))

    t=datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f'Completed at {t}')

    results.to_csv(f'simulation_results/{t}.csv')
