from knapsack_optimizer import KnapsackOptimizer

if __name__ == "__main__":
    # Example usage
    values = [10, 5, 1]
    weights = [5,5, 1]
    capacity = 9
    
    solver = KnapsackOptimizer(values, weights, capacity, hbar=0.5)
    solution = solver.solve()
    solver.summary()
    solver.plot_results()
