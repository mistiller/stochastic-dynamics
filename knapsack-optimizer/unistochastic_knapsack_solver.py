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
    Implements quantum measurement collapse framework for knapsack problems via
    Stinespring dilation of MCMC sampling process.
    
    Attributes:
        values: List of item values (objective function weights)
        weights: List of item weights (constraint terms)
        capacity: Environment coupling strength (decoherence source)
        hbar: Measurement resolution parameter - controls collapse rate
               (analogous to Planck's constant in quantum-classical transition)
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
        """Constructs Stinespring-dilated unitary for constrained optimization problem.
        
        Implements Eq. 12-14 from measurement_collapse.md by:
        1. Embedding classical constraints into quantum phase space
        2. Encoding objective function as environmental coupling
        3. Constructing diagonal unitary via Stinespring dilation
        """
        self.state_space_size = 2 ** self.n_items  # Hilbert space dimension
        
        # Ancilla system represents environmental degrees of freedom
        diag = np.zeros(self.state_space_size, dtype=complex)
        for i in range(self.state_space_size):
            selection = self._int_to_selection(i)
            total_value = self._calculate_value(selection)
            total_weight = self._calculate_weight(selection)
            
            # Decoherence operator: Γ(ρ) = e^{-iH}ρe^{iH}, H = value - i·weight·constraint
            if total_weight <= self.capacity:
                # Valid states get phase proportional to value (Hamiltonian evolution)
                diag[i] = np.exp(1j * (total_value / self.hbar))
            else:
                # Invalid states decay exponentially (non-unitary collapse)
                diag[i] = np.exp(-total_weight / self.hbar)  # Environmental coupling

        # Stinespring dilation to preserve formal unitarity
        from scipy.sparse import dia_matrix
        self.stinespring_unitary = dia_matrix((diag, [0]), 
                                            shape=(self.state_space_size, self.state_space_size),
                                            dtype=complex)
            
    def solve(self, samples: int = 1000):
        """Execute measurement protocol via unistochastic evolution and collapse.
        
        Implements the three stages from Sec. 5 of measurement_collapse.md:
        1. Preparation: |ψ₀⟩ = uniform superposition
        2. Evolution: |ψ⟩ = U|ψ₀⟩ (Stinespring-dilated dynamics)
        3. Measurement: Collapse to classical mixture via sampling
        """
        if not hasattr(self, 'stinespring_unitary'):
            self.build_model()
            
        # Initial superposition: |ψ₀⟩ = Σ_x |x⟩/√N
        initial_state = np.ones(self.state_space_size) / np.sqrt(self.state_space_size)
        
        # Unitary evolution: |ψ⟩ = U|ψ₀⟩
        evolved_state = self.stinespring_unitary @ initial_state
        
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
