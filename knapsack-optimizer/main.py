def main():
    """Main entry point for command line usage"""
    parser = argparse.ArgumentParser(description='Solve knapsack problems')
    parser.add_argument('--solver-type', type=str, default='knapsack',
                       choices=['knapsack', 'unistochastic'],
                       help='Type of solver to use (knapsack|unistochastic)')
    parser.add_argument('--max-items', type=int, default=10,
                       help='Maximum number of items for generated problems')
    args = parser.parse_args()

    # Example usage with configurable solver
    values = [505, 352, 458, 220, 354, 414, 498, 545, 473, 543]
    weights = [23, 26, 20, 18, 32, 27, 29, 26, 30, 27]
    capacity = 67

    if args.solver_type == "unistochastic":
        solver = UnistochasticKnapsackSolver(values, weights, capacity)
    else:
        solver = KnapsackOptimizer(values, weights, capacity)
        
    solution = solver.solve()
    solver.summary()

if __name__ == "__main__":
    import argparse
    from knapsack_optimizer import KnapsackOptimizer
    from unistochastic_knapsack_solver import UnistochasticKnapsackSolver
    main()
