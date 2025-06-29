def main():
    """Main entry point for command line usage"""
    parser = argparse.ArgumentParser(description='Compare knapsack solvers')
    parser.add_argument('--max-items', type=int, default=20,
                       help='Maximum number of items for comparison')
    parser.add_argument('--runs', type=int, default=10,
                       help='Number of runs per item count')
    args = parser.parse_args()

    # Run comparative analysis
    ko = KnapsackOptimizer([1], [1], 1)  # Dummy instance for method access
    results = ko.compare_solvers_scaling(max_items=args.max_items, runs_per_size=args.runs)

    # Print formatted results
    print("\nSolver Comparison Results:")
    print(f"{'Items':<6} {'PI Value':<9} {'Greedy':<9} {'DP':<9} {'Uni':<9} {'PI Time':<8} {'Greedy Time':<11} {'DP Time':<8} {'Uni Time':<8}")
    for _, row in results.iterrows():
        print(f"{row['items']:<6} "
              f"{row['avg_pi_value']:>8.1f} "
              f"{row['avg_greedy_value']:>8.1f} "
              f"{row['avg_dp_value']:>8.1f} "
              f"{row['avg_uni_value']:>8.1f} "
              f"{row['pi_time']:>7.2f}s "
              f"{row['greedy_time']:>10.2f}s "
              f"{row['dp_time']:>7.2f}s "
              f"{row['uni_time']:>7.2f}s")

if __name__ == "__main__":
    import argparse
    from knapsack_optimizer import KnapsackOptimizer
    from unistochastic_knapsack_solver import UnistochasticKnapsackSolver
    main()
