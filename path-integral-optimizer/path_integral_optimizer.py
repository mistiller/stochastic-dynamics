import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import pymc as pm
from pytensor import tensor as pt
from pytensor import config
from loguru import logger
import arviz as az

# Assuming PathIntegralOptimizerResult will be updated to handle posterior summaries
from path_integral_optimizer_result import PathIntegralOptimizerResult

# Helper function to create PyMC distributions from dict descriptions
def get_pymc_distribution(name: str, prior_info: Dict[str, Any]):
    dist_name = prior_info["dist"]
    params = {k: v for k, v in prior_info.items() if k != "dist"}
    if hasattr(pm, dist_name):
        return getattr(pm, dist_name)(name, **params)
    else:
        raise ValueError(f"Unknown distribution name: {dist_name}")

class PathIntegralOptimizer:
    """
    A class for performing path integral optimization using MCMC,
    allowing for uncertainty in model parameters 'base_benefit', 'scale_benefit', and GP-based d(t).
    """
    def __init__(self,
                 base_benefit: Dict[str, Any],
                 scale_benefit: Dict[str, Any],
                 gp_eta_prior: Dict[str, Any],
                 gp_ell_prior: Dict[str, Any],
                 gp_mean_prior: Dict[str, Any],
                 base_cost: float,
                 total_resource: float,
                 T: int,
                 hbar: float,
                 num_steps: int,
                 burn_in: int) -> None:
        """Initializes the PathIntegralOptimizer.

        Args:
            base_benefit (Dict): Prior definition for parameter 'base_benefit'. Ex: {"dist": "Normal", "mu": 1.0, "sigma": 0.1}
            scale_benefit (Dict): Prior definition for parameter 'scale_benefit'. Ex: {"dist": "TruncatedNormal", "mu": 0.5, "sigma": 0.05, "upper": 1.0}
            gp_eta_prior (Dict): Prior for GP amplitude 'eta'. Ex: {"dist": "HalfNormal", "sigma": 1}
            gp_ell_prior (Dict): Prior for GP lengthscale 'ell'. Ex: {"dist": "Gamma", "alpha": 5, "beta": 1}
            gp_mean_prior (Dict): Prior for GP mean 'mean_d'. Ex: {"dist": "Normal", "mu": 2, "sigma": 0.5}
            base_cost (float): Fixed parameter for cost function.
            total_resource (float): Fixed total resource.
            T (int): Fixed number of time steps.
            hbar (float): Fixed noise parameter.
            num_steps (int): Number of MCMC steps.
            burn_in (int): Number of burn-in steps.
        """
        if not isinstance(base_benefit, Dict):
            raise ValueError('base_benefit has to be of type dict')
        if not isinstance(scale_benefit, Dict):
            raise ValueError('scale_benefit has to be of type dict')
        if not isinstance(gp_eta_prior, Dict):
            raise ValueError('gp_eta_prior has to be of type dict')
        if not isinstance(gp_ell_prior, Dict):
            raise ValueError('gp_ell_prior has to be of type dict')
        if not isinstance(gp_mean_prior, Dict):
            raise ValueError('gp_mean_prior has to be of type dict')

        #config.mode = 'JAX'  # very slow

        # Store priors and fixed values
        self.a_prior_def: Dict[str, Any] = base_benefit
        self.b_prior_def: Dict[str, Any] = scale_benefit
        self.gp_eta_prior_def: Dict[str, Any] = gp_eta_prior
        self.gp_ell_prior_def: Dict[str, Any] = gp_ell_prior
        self.gp_mean_prior_def: Dict[str, Any] = gp_mean_prior
        self.base_cost: float = base_cost
        self.total_resource: float = total_resource
        self.T: int = T
        self.hbar: float = hbar
        self.num_steps: int = num_steps
        self.burn_in: int = burn_in

        # Initialize containers
        self.mcmc_paths: Optional[np.ndarray] = None  # Store all paths from trace later
        self.actions: Optional[np.ndarray] = None     # Store actions for each sampled path/parameter combo
        self.trace: Optional[az.InferenceData] = None # Store the full InferenceData

    def compute_action(self,
                       x_path: pt.TensorVariable,
                       base_benefit: pt.TensorVariable,
                       scale_benefit: pt.TensorVariable,
                       d_t: pt.TensorVariable) -> pt.TensorVariable:
        """Computes the action using PyTensor, accepting 'base_benefit' and 'scale_benefit' as variables."""
        try:
            # Use passed 'base_benefit' and 'scale_benefit', and fixed 'self.base_cost'
            benefit = base_benefit * x_path ** scale_benefit
            cost = self.base_cost * x_path ** d_t  # Vectorized d(t) implementation
            return -pt.sum(benefit - cost)
        except Exception as e:
            logger.exception(f"Error in compute_action: {e}")
            raise

    def _compute_action_numpy(self,
                             x_path: np.ndarray,
                             base_benefit: float,
                             scale_benefit: float,
                             d_t: np.ndarray) -> float:
        """Computes the action using NumPy, accepting specific 'base_benefit' and 'scale_benefit' values."""
        try:
            # Ensure inputs are valid numbers
            if not (np.isfinite(base_benefit) and np.isfinite(scale_benefit)):
                logger.warning(f"Non-finite parameter values passed: base_benefit={base_benefit}, scale_benefit={scale_benefit}")
                return np.inf

            # Ensure x_path has no non-positive values before exponentiation, esp. with scale_benefit<1
            if np.any(x_path <= 0):
                 logger.warning(f"Path contains non-positive values: {x_path}. Setting action to inf.")
                 return np.inf
            x_safe = x_path # Use directly if checks pass, or add epsilon if needed based on errors

            # Check for potential issues with scale_benefit < 1 and x near 0
            benefit = base_benefit * np.power(x_safe + 1e-12, scale_benefit)

            cost = self.base_cost * x_safe ** d_t
            action = -np.sum(benefit - cost)

            if not np.isfinite(action):
                 logger.warning(f"Non-finite action computed for path: {x_path}, base_benefit={base_benefit:.3f}, scale_benefit={scale_benefit:.3f}. Action: {action}")
                 return np.inf
            return float(action)

        except FloatingPointError as fpe:
             logger.warning(f"Floating point error in _compute_action_numpy for path: {x_path}, base_benefit={base_benefit:.3f}, scale_benefit={scale_benefit:.3f}. Error: {fpe}")
             return np.inf
        except Exception as e:
            logger.exception(f"Error in _compute_action_numpy (base_benefit={base_benefit:.3f}, scale_benefit={scale_benefit:.3f}): {e}")
            raise

    def run_mcmc(self) -> None:
        """Runs the NUTS simulation inferring 'base_benefit', 'scale_benefit', and the path 'x_path'."""
        logger.info("Starting PyTensor/PyMC MCMC sampling for x_path, base_benefit, scale_benefit, and GP-based d(t)...")
        try:
            with pm.Model(coords={"t": np.arange(self.T)}) as model:
                # --- Priors for Parameters ---
                base_benefit = get_pymc_distribution("base_benefit", self.a_prior_def)
                scale_benefit = get_pymc_distribution("scale_benefit", self.b_prior_def)

                # --- GP for d(t): Prior Hyperparameters ---
                eta = get_pymc_distribution("eta", self.gp_eta_prior_def)
                ell = get_pymc_distribution("ell", self.gp_ell_prior_def)
                mean_d = get_pymc_distribution("mean_d", self.gp_mean_prior_def)

                # --- GP Covariance Function ---
                cov_d = eta**2 * pm.gp.cov.ExpQuad(1, ell) + pm.gp.cov.WhiteNoise(1e-6)  # Add jitter

                # --- GP Definition ---
                X = np.arange(1, self.T + 1)[:, None]  # Input: time steps as 2D array
                X = (X - np.mean(X)) / np.std(X)  # Standardize input for GP

                gp_d = pm.gp.Latent(
                    mean_func=pm.gp.mean.Constant(mean_d),
                    cov_func=cov_d
                )
                f_d = gp_d.prior("f_d", X=X)  # Latent GP function values
                d_t = pt.softplus(f_d)  # Safer positivity constraint than exp()

                # --- Prior for Path (Reparameterized) ---
                x_raw = pm.Normal("x_raw", mu=0, sigma=1, dims="t")
                softmax_x_raw = pt.exp(x_raw) / pt.exp(x_raw).sum(axis=0)
                x_path = pm.Deterministic("x_path", softmax_x_raw * self.total_resource, dims="t")

                # --- Potentials ---
                # Constraints (optional but good practice)
                pm.Potential("non_negative_path", pt.switch(pt.any(x_path < 0), -np.inf, 0))
                pm.Potential("finite_check_path", pt.switch(pt.any(pt.isnan(x_path)), -np.inf, 0))

                # Action Potential: depends on sampled base_benefit, scale_benefit and fixed self.hbar, self.base_cost
                action = self.compute_action(x_path, base_benefit, scale_benefit, d_t)
                # Ensure action is finite before using in potential
                finite_action = pt.switch(pt.isnan(action) | pt.isinf(action), -np.inf, -action / self.hbar)
                pm.Potential("action", finite_action)

                # --- Sampling ---
                self.trace = pm.sample(
                    draws=self.num_steps,
                    tune=self.burn_in,
                    target_accept=0.99, # Increased to reduce divergences
                    max_treedepth=12,   # Increased to allow deeper exploration
                    chains=4,           # Standard practice
                    cores=4,            # Use multiple cores if available
                    return_inferencedata=True
                )

            logger.info("MCMC sampling finished. Processing trace...")

            # Extract posterior samples
            # Shape: (chains * draws, T) for path
            self.mcmc_paths = self.trace.posterior["x_path"].values.reshape(-1, self.T)
            # Shape: (chains * draws,) for params
            a_samples = self.trace.posterior["base_benefit"].values.flatten()
            b_samples = self.trace.posterior["scale_benefit"].values.flatten()
            f_d_samples = self.trace.posterior["f_d"].values.reshape(-1, self.T)  # GP latent function
            d_samples = np.exp(f_d_samples)  # Transform to d(t)

            # Calculate action for each sample using corresponding sampled parameters
            num_samples = len(self.mcmc_paths)
            self.actions = np.full(num_samples, np.nan, dtype=float) # Initialize with NaN
            for i in range(num_samples):
                self.actions[i] = self._compute_action_numpy(
                    x_path=self.mcmc_paths[i], 
                    base_benefit=a_samples[i], 
                    scale_benefit=b_samples[i],
                    d_t=d_samples[i]  # Pass GP-derived d(t)
                )

            # Log how many actions were finite
            num_finite_actions = np.sum(np.isfinite(self.actions))
            logger.info(f"Computed actions for {num_samples} samples ({num_finite_actions} finite).")
            if num_finite_actions == 0:
                 logger.error("No finite actions computed. Cannot plot top paths.")
                 # Consider raising an error or handling this case explicitly

        except Exception as e:
            logger.exception(f"Error in run_mcmc: {e}")
            # Clean up potentially partially assigned state
            self.trace = None
            self.mcmc_paths = None
            self.actions = None
            raise # Re-raise the exception

    def plot_top_paths(self, num_paths_to_plot: int = 10) -> None:
        """Plots the top N paths based on lowest computed action (highest probability)."""
        if self.mcmc_paths is None or self.actions is None:
             logger.warning("MCMC data not available. Run run_mcmc() first.")
             return
        if np.all(~np.isfinite(self.actions)):
             logger.warning("No finite actions found. Cannot plot top paths.")
             return

        try:
            # Get indices of finite actions and sort them
            finite_indices = np.where(np.isfinite(self.actions))[0]
            if len(finite_indices) == 0:
                logger.warning("No finite actions found. Cannot plot top paths.")
                return

            # Sort finite actions (ascending, lower action is better)
            sorted_finite_indices = finite_indices[np.argsort(self.actions[finite_indices])]
            top_indices = sorted_finite_indices[:num_paths_to_plot]

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

    def plot(self) -> None:
        """Plots generated paths with shaded quintiles and a dashed line for the mean path."""
        if self.mcmc_paths is None:
            logger.warning("No paths available. Run run_mcmc() first.")
            return

        plt.figure(figsize=(10, 6))
        
        # Calculate quintiles for shaded areas
        quintiles = [20, 40, 60, 80]
        colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4']  # Different shades of blue
        alphas = [0.2, 0.3, 0.4, 0.3]  # Different transparency levels
        
        # Plot shaded areas for each quintile
        for i, quintile in enumerate(quintiles):
            lower = np.percentile(self.mcmc_paths, quintile, axis=0)
            upper = np.percentile(self.mcmc_paths, 100 - quintile, axis=0)
            plt.fill_between(range(1, self.T + 1), lower, upper, 
                           color=colors[i], alpha=alphas[i], 
                           label=f'{quintile}% - {100-quintile}%')
        
        # Calculate and plot mean path
        mean_path = np.mean(self.mcmc_paths, axis=0)
        plt.plot(range(1, 1 + self.T), mean_path, color='red', 
                linestyle='--', linewidth=2, label='Mean Path')
        
        plt.xlabel("Time (t)")
        plt.ylabel("Allocation (x(t))")
        plt.title("Generated Paths with Quintiles and Mean Path")
        plt.legend()
        plt.grid(True)
        plt.show()

    def generate_summary(self) -> PathIntegralOptimizerResult | None:
        """Generates base_benefit summary of the MCMC results.

        Returns:
            PathIntegralOptimizerResult | None: An object containing the summary results,
                                                or None if no trace is available.
        """
        try:
            if not hasattr(self, 'trace') or self.trace is None:
                logger.warning("No trace collected. Cannot generate summary.")
                return None

            # Convert to ArviZ InferenceData if it'total_resource not already
            if not isinstance(self.trace, az.InferenceData):
                 idata = az.convert_to_inference_data(self.trace)
            else:
                 idata = self.trace

            # Get summary statistics
            mean_path = idata.posterior.x_path.mean(dim=("chain", "draw")).values
            std_path = idata.posterior.x_path.std(dim=("chain", "draw")).values

            # Calculate action values from posterior samples using the NumPy version
            x_path_samples = idata.posterior.x_path.values.reshape(-1, self.T)
            a_samples = idata.posterior["base_benefit"].values.flatten() # Get base_benefit samples
            b_samples = idata.posterior["scale_benefit"].values.flatten() # Get scale_benefit samples
            f_d_samples = idata.posterior["f_d"].values.reshape(-1, self.T)  # GP latent function
            d_samples = np.exp(f_d_samples)  # Transform to d(t)

            # Pass base_benefit and scale_benefit samples to _compute_action_numpy
            action_values = np.array([self._compute_action_numpy(x, base_benefit, scale_benefit, d) for x, base_benefit, scale_benefit, d in zip(x_path_samples, a_samples, b_samples, d_samples)])

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

            # Create and return the result object
            result = PathIntegralOptimizerResult(
                base_benefit=self.a_prior_def,
                scale_benefit=self.b_prior_def,
                base_cost=self.base_cost,
                total_resource=self.total_resource,
                T=self.T,
                hbar=self.hbar,
                num_samples=len(x_path_samples),
                num_finite_actions=len(finite_action_values),
                best_action=best_action,
                action_mean=action_mean,
                action_std=action_std,
                mean_path=mean_path,
                std_path=std_path
            )
            return result

        except Exception as e:
            logger.exception(f"Error in generate_summary: {e}")
            raise
        return None # Ensure None is returned on exception path as well
