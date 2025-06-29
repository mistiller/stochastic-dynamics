"""
Unistochastic Knapsack Solver using Quantum-inspired Measurement Collapse
Implements the measurement collapse approach from measurement_collapse.md
"""

import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from loguru import logger

class UnistochasticKnapsackSolver:
    """
    Solves 0-1 knapsack problems using quantum-inspired unistochastic evolution
    
    Attributes:
        values: List of item values
        weights: List of item weights
        capacity: Maximum allowed total weight
        hbar: Measurement collapse parameter (higher = faster collapse)
    """
    
    def __init__(self, values: List[float], weights: List[float], 
                 capacity: float, hbar: float = 1.0):
        """Initialize a knapsack problem instance."""
        if len(values) != len(weights):
            raise ValueError("Values and weights must have same length")
            
        self.values = np.array(values)
        self.weights = np.array(weights)
        self.capacity = capacity
        self.hbar = hbar
        self.n_items = len(values)
        self.probabilities = None
        self.best_solution = None
        
    def build_model(self):
        """Construct unitary matrix encoding problem constraints"""
        # Size of state space grows exponentially with items (2^n)
        self.state_space_size = 2 ** self.n_items
        self.unitary_matrix = np.zeros((self.state_space_size, self.state_space_size), dtype=complex)
        
        # Build diagonal unitary matrix encoding values/weights
        for i in range(self.state_space_size):
            selection = self._int_to_selection(i)
            total_value = self._calculate_value(selection)
            total_weight = self._calculate_weight(selection)
            
            # Encode value in phase, zero out invalid states
            if total_weight <= self.capacity:
                phase = np.exp(1j * total_value / self.hbar)
            else:
                phase = 0
                
            self.unitary_matrix[i, i] = phase
            
    def solve(self, samples: int = 1000):
        """Run unistochastic evolution and measurement collapse"""
        if not hasattr(self, 'unitary_matrix'):
            self.build_model()
            
        # Initial state: uniform superposition
        initial_state = np.ones(self.state_space_size) / np.sqrt(self.state_space_size)
        
        # Apply unitary evolution
        evolved_state = self.unitary_matrix @ initial_state
        
        # Calculate probabilities
        self.probabilities = np.abs(evolved_state) ** 2
        self.probabilities /= self.probabilities.sum()  # Normalize
        
        # Sample from probability distribution
        sampled_states = np.random.choice(self.state_space_size, size=samples, p=self.probabilities)
        
        # Find best valid solution from samples
        best_value = -np.inf
        best_solution = None
        for state in sampled_states:
            selection = self._int_to_selection(state)
            value = self._calculate_value(selection)
            weight = self._calculate_weight(selection)
            
            if weight <= self.capacity and value > best_value:
                best_value = value
                best_solution = selection
                
        if best_solution is None:
            raise RuntimeError("No valid solutions found in sampling")
            
        self.best_solution = best_solution
        return best_solution
        
    def summary(self, include_baseline: bool = True):
        """Print optimization results with optional baseline comparison"""
        if self.best_solution is None:
            raise RuntimeError("Run solve() first")
            
        total_value = self._calculate_value(self.best_solution)
        total_weight = self._calculate_weight(self.best_solution)
        
        print(f"Unistochastic Optimization Results:")
        print(f"Optimal Value: {total_value:.2f}")
        print(f"Total Weight: {total_weight:.2f}/{self.capacity:.2f}")
        print(f"Included Items: {np.where(self.best_solution)[0].tolist()}\n")
        
        print("Quantum-inspired Diagnostics:")
        print(f"State Space Size: {self.state_space_size}")
        print(f"Max Probability: {self.probabilities.max():.2e}")
        print(f"Entropy: {self._calculate_entropy():.2f}")
        
        if include_baseline:
            # Run greedy baseline
            greedy = self.greedy_solver()
            print("\nGreedy Algorithm Results:")
            print(f"Optimal Value: {greedy[1]:.2f}")
            print(f"Total Weight: {greedy[2]:.2f}/{self.capacity:.2f}")
            print(f"Included Items: {greedy[0]}")
            
    def plot_results(self):
        """Visualize quantum state probabilities"""
        if self.probabilities is None:
            raise RuntimeError("Run solve() first")
            
        _, ax = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot probability distribution
        ax[0].plot(self.probabilities)
        ax[0].set_title("State Probabilities")
        ax[0].set_xlabel("State Index")
        ax[0].set_ylabel("Probability")
        
        # Plot item inclusion probabilities
        inclusion_probs = np.zeros(self.n_items)
        for i in range(self.state_space_size):
            selection = self._int_to_selection(i)
            inclusion_probs += selection * self.probabilities[i]
            
        ax[1].bar(range(self.n_items), inclusion_probs)
        ax[1].set_title("Item Inclusion Probabilities")
        ax[1].set_xlabel("Item Index")
        ax[1].set_ylabel("Probability")
        
        plt.tight_layout()
        plt.show()
        
    def _int_to_selection(self, state: int) -> np.ndarray:
        """Convert integer state to binary item selection vector"""
        return np.array([int(b) for b in f"{state:0{self.n_items}b}"])
        
    def _calculate_value(self, selection: np.ndarray) -> float:
        """Calculate total value for a given selection"""
        return np.dot(self.values, selection)
        
    def _calculate_weight(self, selection: np.ndarray) -> float:
        """Calculate total weight for a given selection"""
        return np.dot(self.weights, selection)
        
    def _calculate_entropy(self) -> float:
        """Calculate Shannon entropy of probability distribution"""
        return -np.sum(self.probabilities * np.log(self.probabilities + 1e-10))
        
    def greedy_solver(self) -> Tuple[List[int], float, float]:
        """Mirror of KnapsackOptimizer's greedy solver for comparison"""
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
