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
    allowing for uncertainty in model parameters 'a' and 'b'.
    """
    def __init__(self,
                 a_prior: Dict[str, Any],
                 b_prior: Dict[str, Any],
                 c: float,
                 S: float,
                 T: int,
                 hbar: float,
                 num_steps: int,
                 burn_in: int) -> None:
        """Initializes the PathIntegralOptimizer.

        Args:
            a_prior (Dict): Prior definition for parameter 'a'. Ex: {"dist": "Normal", "mu": 1.0, "sigma": 0.1}
            b_prior (Dict): Prior definition for parameter 'b'. Ex: {"dist": "TruncatedNormal", "mu": 0.5, "sigma": 0.05, "upper": 1.0}
            c (float): Fixed parameter for cost function.
            S (float): Fixed total resource.
            T (int): Fixed number of time steps.
            hbar (float): Fixed noise parameter.
            num_steps (int): Number of MCMC steps.
            burn_in (int): Number of burn-in steps.
        """
        if not isinstance(a_prior, Dict):
            raise ValueError('a_prior has to be of type dict')
        if not isinstance(b_prior, Dict):
            raise ValueError('a_prior has to be of type dict')
        
        config.mode == 'JAX'

        # Store priors and fixed values
        self.a_prior_def: Dict[str, Any] = a_prior
        self.b_prior_def: Dict[str, Any] = b_prior
        self.c: float = c
        self.S: float = S
        self.T: int = T
        self.hbar: float = hbar
        self.num_steps: int = num_steps
        self.burn_in: int = burn_in

        # Initialize containers
        self.mcmc_paths: Optional[np.ndarray] = None # Store all paths from trace later
        self.actions: Optional[np.ndarray] = None    # Store actions for each sampled path/parameter combo
        self.trace: Optional[az.InferenceData] = None # Store the full InferenceData

    # ... d(t) remains the same ...
    def d(self, t: int) -> float:
        """Defines the time-dependent function d(t)."""
        return 2 + 0.1 * t

    def compute_action(self,
                       x_path: pt.TensorVariable,
                       a: pt.TensorVariable,
                       b: pt.TensorVariable,
                       d_t: pt.TensorVariable) -> pt.TensorVariable:
        """Computes the action using PyTensor, accepting 'a' and 'b' as variables."""
        try:
            # Use passed 'a' and 'b', and fixed 'self.c'
            benefit = a * x_path ** b
            cost = self.c * x_path ** d_t  # Vectorized d(t) implementation
            return -pt.sum(benefit - cost)
        except Exception as e:
            logger.exception(f"Error in compute_action: {e}")
            raise

    def _compute_action_numpy(self,
                             x_path: np.ndarray,
                             a: float,
                             b: float,
                             d_t: np.ndarray) -> float:
        """Computes the action using NumPy, accepting specific 'a' and 'b' values."""
        try:
            # Ensure inputs are valid numbers
            if not (np.isfinite(a) and np.isfinite(b)):
                logger.warning(f"Non-finite parameter values passed: a={a}, b={b}")
                return np.inf

            # Ensure x_path has no non-positive values before exponentiation, esp. with b<1
            if np.any(x_path <= 0):
                 logger.warning(f"Path contains non-positive values: {x_path}. Setting action to inf.")
                 return np.inf
            x_safe = x_path # Use directly if checks pass, or add epsilon if needed based on errors

            # Check for potential issues with b < 1 and x near 0
            # benefit = a * x_safe ** b
            # A very small positive value might be safer if b < 1
            benefit = a * np.power(x_safe + 1e-12, b)

            cost = self.c * x_safe ** d_t
            action = -np.sum(benefit - cost)

            if not np.isfinite(action):
                 logger.warning(f"Non-finite action computed for path: {x_path}, a={a:.3f}, b={b:.3f}. Action: {action}")
                 return np.inf # Return infinity for non-finite actions
            return float(action)

        except FloatingPointError as fpe:
             logger.warning(f"Floating point error in _compute_action_numpy for path: {x_path}, a={a:.3f}, b={b:.3f}. Error: {fpe}")
             return np.inf # Treat FPEs as invalid paths
        except Exception as e:
            logger.exception(f"Error in _compute_action_numpy (a={a:.3f}, b={b:.3f}): {e}")
            raise

    def run_mcmc(self) -> None:
        """Runs the NUTS simulation inferring 'a', 'b', and the path 'x_path'."""
        logger.info("Starting PyTensor/PyMC MCMC sampling for x_path, a, b, and GP-based d(t)...")
        try:
            with pm.Model(coords={"t": np.arange(self.T)}) as model:
                # --- Priors for Parameters ---
                a = get_pymc_distribution("a", self.a_prior_def)
                b = get_pymc_distribution("b", self.b_prior_def)

                # --- GP for d(t): Prior Hyperparameters ---
                eta = pm.HalfNormal("eta", sigma=1)  # More stable than HalfCauchy
                ell = pm.Gamma("ell", alpha=5, beta=1)  # More informative lengthscale prior
                mean_d = pm.Normal("mean_d", mu=2, sigma=0.5)  # Tighter mean prior

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
                x_path = pm.Deterministic("x_path", pt.nnet.softmax(x_raw) * self.S, dims="t")

                # --- Potentials ---
                # Constraints (optional but good practice)
                pm.Potential("non_negative_path", pt.switch(pt.any(x_path < 0), -np.inf, 0))
                pm.Potential("finite_check_path", pt.switch(pt.any(pt.isnan(x_path)), -np.inf, 0))

                # Action Potential: depends on sampled a, b and fixed self.hbar, self.c
                action = self.compute_action(x_path, a, b, d_t)
                # Ensure action is finite before using in potential
                finite_action = pt.switch(pt.isnan(action) | pt.isinf(action), -np.inf, -action / self.hbar)
                pm.Potential("action", finite_action)

                # --- Sampling ---
                self.trace = pm.sample(
                    draws=self.num_steps,
                    tune=self.burn_in,
                    target_accept=0.99, # Increased to reduce divergences
                    chains=4,         # Standard practice
                    cores=4,          # Use multiple cores if available
                    return_inferencedata=True
                )

            logger.info("MCMC sampling finished. Processing trace...")

            # Extract posterior samples
            # Shape: (chains * draws, T) for path
            self.mcmc_paths = self.trace.posterior["x_path"].values.reshape(-1, self.T)
            # Shape: (chains * draws,) for params
            a_samples = self.trace.posterior["a"].values.flatten()
            b_samples = self.trace.posterior["b"].values.flatten()
            f_d_samples = self.trace.posterior["f_d"].values.reshape(-1, self.T)  # GP latent function
            d_samples = np.exp(f_d_samples)  # Transform to d(t)

            # Calculate action for each sample using corresponding sampled parameters
            num_samples = len(self.mcmc_paths)
            self.actions = np.full(num_samples, np.nan, dtype=float) # Initialize with NaN
            for i in range(num_samples):
                self.actions[i] = self._compute_action_numpy(
                    x_path=self.mcmc_paths[i], 
                    a=a_samples[i], 
                    b=b_samples[i],
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

    def generate_summary(self) -> PathIntegralOptimizerResult | None:
        """Generates a summary of the MCMC results.

        Returns:
            PathIntegralOptimizerResult | None: An object containing the summary results,
                                                or None if no trace is available.
        """
        try:
            if not hasattr(self, 'trace') or self.trace is None:
                logger.warning("No trace collected. Cannot generate summary.")
                return None

            # Convert to ArviZ InferenceData if it's not already
            if not isinstance(self.trace, az.InferenceData):
                 idata = az.convert_to_inference_data(self.trace)
            else:
                 idata = self.trace
            
            # Get summary statistics
            mean_path = idata.posterior.x_path.mean(dim=("chain", "draw")).values
            std_path = idata.posterior.x_path.std(dim=("chain", "draw")).values

            # Calculate action values from posterior samples using the NumPy version
            x_path_samples = idata.posterior.x_path.values.reshape(-1, self.T)
            a_samples = idata.posterior["a"].values.flatten() # Get a samples
            b_samples = idata.posterior["b"].values.flatten() # Get b samples
            f_d_samples = idata.posterior["f_d"].values.reshape(-1, self.T)  # GP latent function
            d_samples = np.exp(f_d_samples)  # Transform to d(t)

            # Pass a and b samples to _compute_action_numpy
            action_values = np.array([self._compute_action_numpy(x, a, b, d) for x, a, b, d in zip(x_path_samples, a_samples, b_samples, d_samples)])

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
                a=self.a_prior_def,
                b=self.b_prior_def,
                c=self.c,
                S=self.S,
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
