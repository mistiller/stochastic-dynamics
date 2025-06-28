from knapsack_optimizer import KnapsackOptimizer

if __name__ == "__main__":
    # Run scaling comparison simulation
    dummy_values = [1]  # Values will be generated in the simulation
    dummy_weights = [1] # Weights will be generated in the simulation
    
    solver = KnapsackOptimizer(dummy_values, dummy_weights, capacity=1, hbar=0.5)
    results = solver.compare_solvers_scaling(max_items=10, runs_per_size=5)
    
    # Print results dataframe
    print("\nSolver Scaling Results DataFrame:")
    print(results.to_string(index=False, formatter={
        'agreement_rate': '{:.0%}'.format,
        'avg_time': '{:.2f}s'.format,
        'max_time': '{:.2f}s'.format
    }))
