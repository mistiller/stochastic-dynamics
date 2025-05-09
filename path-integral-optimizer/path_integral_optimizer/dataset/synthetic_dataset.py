import numpy as np
from typing import Dict

from .dataset import Dataset

class SyntheticDataset:
    """Generates synthetic time series data for testing parameter estimation.
    
    Produces a Dataset object with:
    - 't': Time indices (1 to T)
    - 'input': Allocated input values from a total resource.
    - 'cost': Function of input (base_cost * input^d_t_value) + noise.
    - 'benefit': Function of input (base_benefit * input^scale_benefit) + noise.
    """
    def __init__(self,
                 T: int = 12,
                 total_resource: float = 100.0,
                 base_cost: float = 0.5,
                 base_benefit: float = 1.0,
                 scale_benefit: float = 0.8, # Typical range (0,1)
                 d_t_value: float = 1.2,     # Example exponent for cost
                 cost_noise: float = 0.1,
                 benefit_noise: float = 0.05):
        """
        Args:
            T: Number of time periods.
            total_resource: Total amount to be allocated across T periods.
            base_cost: Baseline cost coefficient.
            base_benefit: Baseline benefit coefficient.
            scale_benefit: Exponent for input in the benefit function.
            d_t_value: Exponent for input in the cost function (constant for synthetic data).
            cost_noise: Std dev of cost noise.
            benefit_noise: Std dev of benefit noise.
        """
        self.T = T
        self.total_resource = total_resource
        self.base_cost = base_cost
        self.base_benefit = base_benefit
        self.scale_benefit = scale_benefit
        self.d_t_value = d_t_value
        self.cost_noise = cost_noise
        self.benefit_noise = benefit_noise

    def generate(self) -> Dataset:
        """Generates synthetic dataset with allocated input and derived cost/benefit."""
        t = np.arange(1, self.T + 1)

        # Generate input series as an allocation of total_resource
        # 1. Generate random raw values
        raw_allocations = np.random.rand(self.T) + 1e-6 # Add small epsilon to avoid zero if all rand are 0
        # 2. Apply softmax to get proportions
        softmax_allocations = np.exp(raw_allocations) / np.sum(np.exp(raw_allocations))
        # 3. Scale by total_resource
        input_series = softmax_allocations * self.total_resource
        # Ensure sum is exactly total_resource due to potential floating point inaccuracies
        input_series = input_series / np.sum(input_series) * self.total_resource
        input_series = np.clip(input_series, 1e-9, None) # Ensure inputs are positive for power functions

        # Generate cost and benefit series based on the input
        # Cost = base_cost * input^d_t_value + noise
        cost = (
            self.base_cost * (input_series ** self.d_t_value)
            + np.random.normal(scale=self.cost_noise, size=self.T)
        )

        # Benefit = base_benefit * input^scale_benefit + noise
        benefit = (
            self.base_benefit * (input_series ** self.scale_benefit)
            + np.random.normal(scale=self.benefit_noise, size=self.T)
        )
        
        # Ensure cost and benefit are positive, if necessary for the model
        cost = np.maximum(cost, 1e-9)
        benefit = np.maximum(benefit, 1e-9)

        return Dataset(
            t,
            input_series,
            cost,
            benefit
        )
