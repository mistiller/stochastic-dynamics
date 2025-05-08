import numpy as np
from typing import Dict

from .dataset import Dataset

class SyntheticDataset:
    """Generates synthetic time series data for testing parameter estimation.
    
    Produces a Dataset object with:
    - 't': Time indices (1 to T)
    - 'input': Random walk input values
    - 'cost': Function of input + time-dependent component
    - 'benefit': Function of input with noise
    """
    def __init__(self, 
                 T: int = 12,
                 base_cost: float = 0.5,
                 base_benefit: float = 1.0,
                 cost_noise: float = 0.1,
                 benefit_noise: float = 0.05,
                 random_walk_std: float = 0.2):
        """
        Args:
            T: Number of time periods
            base_cost: Baseline cost coefficient
            base_benefit: Baseline benefit coefficient 
            cost_noise: Std dev of cost noise
            benefit_noise: Std dev of benefit noise
            random_walk_std: Std dev for input random walk steps
        """
        self.T = T
        self.base_cost = base_cost
        self.base_benefit = base_benefit
        self.cost_noise = cost_noise
        self.benefit_noise = benefit_noise
        self.random_walk_std = random_walk_std

    def generate(self) -> Dataset:
        """Generate synthetic dataset with random walk input and noisy observations."""
        # Generate random walk input starting from 1.0
        input_series = np.cumprod(np.exp(np.random.normal(
            scale=self.random_walk_std, 
            size=self.T
        ))) 
        input_series /= input_series[0]  # Start at 1.0
        
        # Create time-varying cost component (quadratic trend + noise)
        t = np.arange(1, self.T+1)
        time_effect = 0.1 * t + 0.05 * t**2
        
        # Generate cost and benefit series
        cost = (
            self.base_cost * input_series 
            + time_effect
            + np.random.normal(scale=self.cost_noise, size=self.T)
        )
        
        benefit = (
            self.base_benefit * input_series 
            + np.random.normal(scale=self.benefit_noise, size=self.T)
        )
        
        return Dataset(
            t,
            input_series, 
            cost, 
            benefit
        )