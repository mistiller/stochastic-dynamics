import numpy as np
import matplotlib.pyplot as plt
from typing import List # Added for type hinting
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from pytensor import tensor as pt
from loguru import logger
import arviz as az

class PathIntegralOptimizer:
    """A class for performing path integral optimization using Markov Chain Monte Carlo (MCMC)."""
    def __init__(self, a: float, b: float, c: float, S: float, T: int, hbar: float, num_steps: int, burn_in: int, proposal_stddev: float, seed: int = 42) -> None:
        """Initializes the PathIntegralOptimizer.

        Args:
            a (float): Parameter for benefit function.
            b (float): Parameter for benefit function.
            c (float): Parameter for cost function.
            S (float): Total resource.
            T (int): Number of time steps.
            hbar (float): Noise parameter.
            num_steps (int): Number of MCMC steps.
            burn_in (int): Number of burn-in steps.
            proposal_stddev: float = Standard deviation for the proposal distribution.
            seed (int): Random seed for reproducibility.
        """
        self.a: float = a
        self.b: float = b
        self.c: float = c
        self.S: float = S
        self.T: float = T
        self.hbar: float = hbar
        self.num_steps: int = num_steps
        self.burn_in: int = burn_in
        self.proposal_stddev: float = proposal_stddev
        self.seed: int = seed
        np.random.seed(self.seed)
        self.mcmc_paths: List[np.ndarray] = []
        self.actions: List[float] = []

    def d(self, t: int) -> float:
        """Defines the time-dependent function d(t).

        Args:
            t (int): The time step.

        Returns:
            float: The value of d(t).
        """
        return 2 + 0.1 * t

    def compute_action(self, x_path: pt.TensorVariable) -> pt.TensorVariable:
        """Computes the action for a given path using PyTensor.

        Args:
            x_path (pt.TensorVariable): The allocation path tensor.

        Returns:
            pt.TensorVariable: The action value tensor.
        """
        try:
            t = np.arange(1, self.T + 1)
            x_safe = x_path + 1e-9
            benefit = self.a * x_safe ** self.b
            cost = self.c * x_safe ** (2 + 0.1 * t)  # Vectorized d(t) implementation
            return -np.sum(benefit - cost)
        except Exception as e:
            logger.exception(f"Error in compute_action: {e}")
            raise

    def _compute_action_numpy(self, x_path: np.ndarray) -> float:
        """Computes the action for a given path using NumPy.

        Args:
            x_path (np.ndarray): The allocation path array.

        Returns:
            float: The action value.
        """
        try:
            t = np.arange(1, self.T + 1)
            x_safe = x_path + 1e-9  # Add small epsilon for numerical stability
            benefit = self.a * x_safe ** self.b
            cost = self.c * x_safe ** (2 + 0.1 * t)
            action = -np.sum(benefit - cost)
            if not np.isfinite(action):
                 logger.warning(f"Non-finite action computed for path: {x_path}. Action: {action}")
                 # Handle non-finite actions, e.g., return a large penalty or skip
                 return np.inf # Or some large finite number
            return float(action)
        except Exception as e:
            logger.exception(f"Error in _compute_action_numpy: {e}")
            raise


    def run_mcmc(self) -> None:
        """Runs the Hamiltonian Monte Carlo (HMC) simulation using PyTensor/PyMC.

        This method uses NUTS sampling for more efficient exploration of the parameter space.
        """
        logger.info("Starting PyTensor/PyMC MCMC sampling...")
        try:
            with pm.Model() as model:
                # Define constrained allocation path using Dirichlet distribution scaled by S
                x_path = pm.Dirichlet("x_path", a=np.ones(self.T)) * self.S
                
                # Enforce non-negative and finite values
                pm.Potential("non_negative", pt.switch(x_path < 0, -np.inf, 0))
                pm.Potential("finite_check", pt.switch(pt.isnan(x_path), -np.inf, 0))
                
                # Define action with simplified exponent for stability
                action = self.compute_action(x_path)
                pm.Potential("action", -action / self.hbar)
                
                # Run NUTS sampler
                self.trace = pm.sample(
                    draws=self.num_steps,
                    tune=self.burn_in,
                    target_accept=0.9,
                    chains=4,
                    cores=4,
                    return_inferencedata=True)
                
            logger.info("MCMC sampling finished. Storing samples.")
            self.mcmc_paths = self.trace.posterior.x_path.values.reshape(-1, self.T).tolist()
            self.actions = [-self.trace.log_likelihood['action'].values.mean(axis=1)]
        except Exception as e:
            logger.exception(f"Error in run_mcmc: {e}")
            raise

    def plot_top_paths(self, num_paths_to_plot: int = 10) -> None:
        """Plots the top N most probable paths.

        Args:
            num_paths_to_plot (int): The number of paths to plot.
        """
        try:
            # Select top paths based on lowest action (highest probability)
            # Sort indices based on action (ascending, lower action is better)
            sorted_indices: np.ndarray = np.argsort(self.actions)
            top_indices: np.ndarray = sorted_indices[:num_paths_to_plot]

            # Plot top most probable paths from MCMC samples
            plt.figure(figsize=(10, 6))
            for i, idx in enumerate(top_indices):
                plt.plot(range(1, self.T + 1), self.mcmc_paths[idx], label=f"Path {i+1} (Action: {self.actions[idx]:.2f})", alpha=0.7)

            # Plot average path
            average_path = np.mean(self.mcmc_paths, axis=0)
            plt.plot(range(1, self.T + 1), average_path, label="Average Path", color='black', linewidth=2, linestyle='--')

            plt.xlabel("Time (t)")
            plt.ylabel("Allocation (x(t))")
            plt.title(f"Top {num_paths_to_plot} MCMC Sampled Allocation Paths (Lowest Action)")
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as e:
            logger.exception(f"Error in plot_top_paths: {e}")
            raise

    def generate_summary(self):
        """Generates and logs a summary of the MCMC results using ArviZ."""
        try:
            if not hasattr(self, 'trace'):
                logger.warning("No trace collected. Cannot generate summary.")
                return

            # Convert to ArviZ InferenceData
            idata = az.convert_to_inference_data(self.trace)
            
            # Get summary statistics
            mean_path = idata.posterior.x_path.mean(dim=("chain", "draw")).values
            std_path = idata.posterior.x_path.std(dim=("chain", "draw")).values
            # Calculate action values from posterior samples using the NumPy version
            x_path_samples = idata.posterior.x_path.values.reshape(-1, self.T)
            action_values = np.array([self._compute_action_numpy(x) for x in x_path_samples])

            # Filter out potential non-finite values before finding min/mean/std
            finite_action_values = action_values[np.isfinite(action_values)]
            if len(finite_action_values) == 0:
                logger.warning("No finite action values found in the posterior samples.")
                best_action = np.nan
                action_mean = np.nan
                action_std = np.nan
            else:
                best_action = float(np.min(finite_action_values))
                action_mean = float(np.mean(finite_action_values))
                action_std = float(np.std(finite_action_values))

            logger.info("=== MCMC Summary ===")
            logger.info(f"Number of samples: {len(x_path_samples)} (Finite actions: {len(finite_action_values)})")
            logger.info(f"Parameters: a={self.a}, b={self.b}, c={self.c}, S={self.S}, T={self.T}, hbar={self.hbar}")
            logger.info(f"Best path action: {best_action:.4f}")
            logger.info(f"Mean action: {action_mean:.4f} ± {action_std:.4f}")
            logger.info("Mean allocation per time step:")
            logger.info("Time : Mean ± Std Dev")
            for t in range(self.T):
                logger.info(f"{t+1:2d}   : {mean_path[t]:.4f} ± {std_path[t]:.4f}")
            logger.info("====================")
        except Exception as e:
            logger.exception(f"Error in generate_summary: {e}")
            raise
