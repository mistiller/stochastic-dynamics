"""
Quantum-inspired Knapsack Solver using Path Integral Optimization
Implements a Bayesian approach to the 0-1 knapsack problem using PyMC
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from loguru import logger

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
        if len(values) != len(weights):
            raise ValueError("Values and weights must have same length")
            
        self.values = np.array(values)
        self.weights = np.array(weights)
        self.capacity = capacity
        self.hbar = hbar
        self.trace = None
        self.best_solution = None
        
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
            # Use NUTS for continuous relaxation and Gibbs for discrete variables
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains,
                                 target_accept=0.95, step=pm.SMC())
            
        # Extract best solution
        posterior = az.extract(self.trace, 'inclusion')
        best_idx = np.argmax(posterior.sum('sample').values)
        self.best_solution = posterior.sel(chain=best_idx).values.astype(bool)
        
        return self.best_solution
        
    def summary(self):
        """Print optimization results"""
        if self.trace is None:
            raise RuntimeError("Run solve() first")
            
        total_value = self.values[self.best_solution].sum()
        total_weight = self.weights[self.best_solution].sum()
        
        print(f"Optimal Value: {total_value:.2f}")
        print(f"Total Weight: {total_weight:.2f}/{self.capacity:.2f}")
        print(f"Included Items: {np.where(self.best_solution)[0]}")
        
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

if __name__ == "__main__":
    # Example usage
    values = [505, 352, 458, 220, 354, 414, 498, 545, 473, 543]
    weights = [23, 26, 20, 18, 32, 27, 29, 26, 30, 27]
    capacity = 67
    
    solver = KnapsackOptimizer(values, weights, capacity, hbar=0.5)
    solution = solver.solve()
    solver.summary()
    solver.plot_results()
