from knapsack_optimizer import KnapsackOptimizer

if __name__ == "__main__":
    # Example usage
    values = [505, 352, 458, 220, 354, 414, 498, 545, 473, 543]
    weights = [23, 26, 20, 18, 32, 27, 29, 26, 30, 27]
    capacity = 67
    
    solver = KnapsackOptimizer(values, weights, capacity, hbar=0.5)
    solution = solver.solve()
    solver.summary()
    solver.plot_results()
