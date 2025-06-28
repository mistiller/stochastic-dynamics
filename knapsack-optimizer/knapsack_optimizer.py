"""
Quantum-inspired Knapsack Solver using Path Integral Optimization
Implements a Bayesian approach to the 0-1 knapsack problem using PyMC
"""

import math
import numpy as np
import pymc as pm
import pymc.smc as smc
import arviz as az
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from loguru import logger
from functools import lru_cache

class KnapsackOptimizer:
    """
    Solves 0-1 knapsack problems using Hamiltonian Monte Carlo with constraint embedding
    
    Attributes:
        values: List of item values
        weights: List of item weights
        capacity: Maximum allowed total weight
        hbar: Quantum fluctuation parameter (higher = more exploration)
    """
    
    def __init__(self, values: List[float], weights: List[float], 
                 capacity: float, hbar: float = 0.1):
        """Initialize a knapsack problem instance.
        
        Args:
            values: List of item values
            weights: List of item weights
            capacity: Maximum allowed total weight
            hbar: Quantum fluctuation parameter (higher = more exploration)
        """
        if len(values) != len(weights):
            raise ValueError("Values and weights must have same length")
            
        self.values = np.array(values)
        self.weights = np.array(weights)
        self.capacity = capacity
        self.hbar = hbar
        self.trace = None
        self.best_solution = None
        self._items = list(zip(self.values, self.weights))
        
    def build_model(self):
        """Construct PyMC model with constrained action potential"""
        n_items = len(self.values)
        
        with pm.Model() as model:
            # Bernoulli variables for item inclusion
            inclusion = pm.Bernoulli('inclusion', p=0.5, shape=n_items)
            
            # Calculate total value and weight
            total_value = pm.math.sum(inclusion * self.values)
            total_weight = pm.math.sum(inclusion * self.weights)
            
            # Constraint handling with smooth penalty
            constraint = pm.math.switch(total_weight > self.capacity,
                                      -(total_weight - self.capacity)**2,
                                      0)
            
            # Action potential combining objective and constraint
            pm.Potential('action', 
                        (total_value + self.hbar * constraint) / self.hbar)
            
        return model
        
    def solve(self, draws=2000, tune=1000, chains=4):
        """Run MCMC sampling to find optimal solution"""
        model = self.build_model()
        
        with model:
            # Use Sequential Monte Carlo for discrete variables
            self.trace = smc.sample_smc(
                draws=draws,
                tune=tune,
                chains=chains,
                model=model,
                kernel=smc.kernels.IMH,
                compute_convergence_checks=False,
            )
            
        # Extract best solution
        posterior = az.extract(self.trace, 'inclusion')
        best_idx = np.argmax(posterior.sum('sample').values)
        self.best_solution = posterior.sel(chain=best_idx).values.astype(bool)
        
        return self.best_solution
        
    def greedy_solver(self) -> Tuple[List[int], float, float]:
        """Solve knapsack using greedy approach by value/weight ratio.
        
        Returns:
            Tuple of (selected item indices, total value, total weight)
        """
        # Sort items by value/weight ratio in descending order
        items = sorted(
            [(i, v, w) for i, (v, w) in enumerate(zip(self.values, self.weights))],
            key=lambda x: x[1]/x[2] if x[2] > 0 else float('inf'),
            reverse=True
        )
        
        total_weight = 0
        total_value = 0
        selected = []
        
        for i, v, w in items:
            if total_weight + w <= self.capacity:
                selected.append(i)
                total_weight += w
                total_value += v
                
        return selected, total_value, total_weight
    
    def dynamic_programming_solver(self) -> Tuple[List[int], float, float]:
        """Solve 0-1 knapsack using pseudo-polynomial DP approach.
        
        Returns:
            Tuple of (selected item indices, total value, total weight)
        """
        n = len(self.values)
        capacity = int(self.capacity)
        weights = self.weights.astype(int)

        # Initialize DP table
        dp = np.zeros((n + 1, capacity + 1))

        # Build DP table
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(
                        self.values[i - 1] + dp[i - 1][w - weights[i - 1]],
                        dp[i - 1][w],
                    )
                else:
                    dp[i][w] = dp[i - 1][w]

        # Backtrack to find selected items
        selected = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected.append(i - 1)
                w -= weights[i - 1]

        total_value = dp[n][capacity]
        total_weight = sum(self.weights[i] for i in selected)

        return selected, total_value, total_weight
    
    def fptas_solver(self, epsilon: float = 0.1) -> Tuple[List[int], float, float]:
        """Solve knapsack using FPTAS approximation scheme.
        
        Args:
            epsilon: Approximation factor (0 < epsilon < 1)
            
        Returns:
            Tuple of (selected item indices, total value, total weight)
        """
        if any(w <= 0 for w in self.weights):
            raise ValueError("Weights must be positive for FPTAS")
            
        # Scale profits for FPTAS
        max_value = max(self.values)
        k = epsilon * max_value / len(self.values)
        scaled_values = [int(v / k) for v in self.values]
        
        # Use dynamic programming with scaled values
        capacity = int(self.capacity)
        n = len(scaled_values)
        weights = self.weights.astype(int)
        
        dp = np.zeros((capacity + 1, n + 1))
        
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                if weights[i-1] <= w:
                    dp[w][i] = max(
                        dp[w][i-1],
                        scaled_values[i-1] + dp[w - weights[i-1]][i-1]
                    )
                else:
                    dp[w][i] = dp[w][i-1]
        
        # Backtrack to find selected items
        selected = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[w][i] != dp[w][i-1]:
                selected.append(i-1)
                w -= weights[i-1]
                
        total_value = sum(self.values[i] for i in selected)
        total_weight = sum(self.weights[i] for i in selected)
        
        return selected, total_value, total_weight
        
    def summary(self, include_baseline: bool = True):
        """Print optimization results with optional baseline comparison"""
        if self.trace is None:
            raise RuntimeError("Run solve() first")
            
        # Get path integral solution metrics
        path_integral_value = self.values[self.best_solution].sum()
        path_integral_weight = self.weights[self.best_solution].sum()
        
        print(f"Path Integral Optimization Results:")
        print(f"Optimal Value: {path_integral_value:.2f}")
        print(f"Total Weight: {path_integral_weight:.2f}/{self.capacity:.2f}")
        print(f"Included Items: {np.where(self.best_solution)[0]}")
        
        if include_baseline:
            # Run greedy baseline
            greedy_items, greedy_value, greedy_weight = self.greedy_solver()
            print("\nGreedy Algorithm Results:")
            print(f"Optimal Value: {greedy_value:.2f}")
            print(f"Total Weight: {greedy_weight:.2f}/{self.capacity:.2f}")
            print(f"Included Items: {greedy_items}")
            
            # Run dynamic programming baseline
            dp_items, dp_value, dp_weight = self.dynamic_programming_solver()
            print("\nDynamic Programming Results:")
            print(f"Optimal Value: {dp_value:.2f}")
            print(f"Total Weight: {dp_weight:.2f}/{self.capacity:.2f}")
            print(f"Included Items: {dp_items}")
            
            # Run FPTAS baseline with epsilon=0.1
            try:
                fptas_items, fptas_value, fptas_weight = self.fptas_solver(epsilon=0.1)
                print("\nFPTAS Algorithm Results:")
                print(f"Optimal Value: {fptas_value:.2f}")
                print(f"Total Weight: {fptas_weight:.2f}/{self.capacity:.2f}")
                print(f"Included Items: {fptas_items}")
            except ValueError as e:
                print("\nFPTAS not applicable:", str(e))
        
    def plot_results(self):
        """Visualize sampling results"""
        if self.trace is None:
            raise RuntimeError("Run solve() first")
            
        _, ax = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot value distribution
        az.plot_posterior(self.trace, var_names=['action'], 
                        ax=ax[0], ref_val=self.values.sum())
        ax[0].set_title("Total Value Distribution")
        
        # Plot weight distribution
        az.plot_posterior(self.trace, var_names=['total_weight'], 
                        ax=ax[1], ref_val=self.capacity)
        ax[1].set_title("Total Weight Distribution")
        
        plt.tight_layout()
        plt.show()
