import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import pymc as pm
from pytensor import tensor as pt
from pytensor import config
from loguru import logger
import arviz as az

# Assuming PathIntegralOptimizerResult will be updated to handle posterior summaries
from .path_integral_optimizer_result import PathIntegralOptimizerResult
from .parameter_estimator import ParameterEstimator
from .parameter_estimation_result import ParameterEstimationResult
from .dataset import Dataset

# Helper function to create PyMC distributions from dict descriptions
def get_pymc_distribution(name: str, prior_info: Dict[str, Any])->pm.Distribution:
    """
    Creates a PyMC distribution based on a dictionary description.
    Handles conversion from {'mu': ..., 'sigma': ...} to distribution-specific
    parameters for common distributions like TruncatedNormal, Beta, HalfNormal, Gamma.
    """
    if not isinstance(prior_info, Dict) or "dist" not in prior_info:
         raise ValueError(f"Prior info for '{name}' must be a dictionary with a 'dist' key. Got {prior_info}")

    dist_name = prior_info["dist"]
    params = {k: v for k, v in prior_info.items() if k != "dist"}

    try:
        if dist_name == "TruncatedNormal":
            # Expects mu, sigma, lower, upper. 'lower' is required for this use case.
            mu = params.get("mu")
            sigma = params.get("sigma")
            lower = params.get("lower") # 'lower' is now expected in the dict
            upper = params.get("upper", np.inf)
            if mu is None or sigma is None or lower is None:
                 raise ValueError(f"TruncatedNormal prior for {name} requires 'mu', 'sigma', and 'lower'. Got {prior_info}")
            # Ensure sigma is positive
            sigma = max(sigma, 1e-6)
            return pm.TruncatedNormal(name, mu=mu, sigma=sigma, lower=lower, upper=upper)

        elif dist_name == "Beta":
            # Expects alpha, beta. Can convert from mu, sigma if provided.
            if "alpha" in params and "beta" in params:
                alpha = params["alpha"]
                beta = params["beta"]
            elif "mu" in params and "sigma" in params:
                mu = params["mu"]
                sigma = params["sigma"]
                 # Ensure mu and sigma are provided for conversion
                if mu is None or sigma is None:
                     raise ValueError(f"Beta prior for {name} requires either 'alpha'/'beta' or 'mu'/'sigma'. Got {prior_info}")
                # Convert mean/std to alpha/beta
                # Ensure mu is within (0, 1) and sigma is valid
                if not (0 < mu < 1):
                     logger.warning(f"Mean {mu:.4f} for Beta prior '{name}' is not in (0, 1). Clamping.")
                     mu = np.clip(mu, 1e-6, 1 - 1e-6)
                variance = sigma**2
                # Variance of Beta is mu*(1-mu)/(alpha+beta+1). Need variance < mu*(1-mu)
                max_variance = mu * (1 - mu)
                if variance >= max_variance:
                     logger.warning(f"Variance {variance:.4f} for Beta prior '{name}' is too large for mean {mu:.4f}. Clamping variance.")
                     variance = max_variance * 0.9 # Use 90% of max variance

                # alpha+beta = mu*(1-mu)/variance - 1
                sum_ab = mu * (1 - mu) / variance - 1
                if sum_ab <= 0: # Should not happen if variance < mu*(1-mu)
                     logger.warning(f"Calculated alpha+beta <= 0 for Beta prior '{name}'. Using default Beta(1,1).")
                     alpha = 1.0
                     beta = 1.0
                else:
                    alpha = mu * sum_ab
                    beta = (1 - mu) * sum_ab

                # Ensure alpha, beta are positive
                alpha = max(alpha, 1e-6)
                beta = max(beta, 1e-6)
                logger.info(f"Converted mu={mu:.3f}, sigma={sigma:.3f} to Beta(alpha={alpha:.3f}, beta={beta:.3f}) for '{name}'.")
            else:
                raise ValueError(f"Beta prior for {name} requires either 'alpha' and 'beta' or 'mu' and 'sigma'. Got {prior_info}")
            return pm.Beta(name, alpha=alpha, beta=beta)

        elif dist_name == "HalfNormal":
            # Expects sigma.
            if "sigma" in params:
                sigma = params["sigma"]
                # Ensure sigma is positive
                sigma = max(sigma, 1e-6)
                return pm.HalfNormal(name, sigma=sigma)
            else:
                 raise ValueError(f"HalfNormal prior for {name} requires 'sigma'. Got {prior_info}")

        elif dist_name == "Gamma":
            # Expects alpha, beta. Can convert from mu, sigma if provided.
            if "alpha" in params and "beta" in params:
                alpha = params["alpha"]
                beta = params["beta"]
            elif "mu" in params and "sigma" in params:
                mu = params["mu"]
                sigma = params["sigma"]
                # Ensure mu and sigma are provided for conversion
                if mu is None or sigma is None:
                     raise ValueError(f"Gamma prior for {name} requires either 'alpha'/'beta' or 'mu'/'sigma'. Got {prior_info}")
                # Convert mean/std to alpha/beta
                # Mean = alpha/beta, Variance = alpha/beta^2
                variance = sigma**2
                if variance <= 0:
                     logger.warning(f"Variance {variance:.4f} for Gamma prior '{name}' is non-positive. Using default Gamma(1,1).")
                     alpha = 1.0
                     beta = 1.0
                else:
                    beta = mu / variance
                    alpha = mu * beta
                # Ensure alpha, beta are positive
                alpha = max(alpha, 1e-6)
                beta = max(beta, 1e-6)
                logger.info(f"Converted mu={mu:.3f}, sigma={sigma:.3f} to Gamma(alpha={alpha:.3f}, beta={beta:.3f}) for '{name}'.")

                return pm.Gamma(name, alpha, beta)
            else:
                raise ValueError(f"Gamma prior for {name} requires either 'alpha' and 'beta' or 'mu' and 'sigma'. Got {prior_info}")

        elif dist_name == "Normal":
             # Expects mu, sigma.
             mu = params.get("mu")
             sigma = params.get("sigma")
             if mu is None or sigma is None:
                  raise ValueError(f"Normal prior for {name} requires 'mu' and 'sigma'. Got {prior_info}")
             # Ensure sigma is positive
             sigma = max(sigma, 1e-6)
             return pm.Normal(name, mu=mu, sigma=sigma)

        elif hasattr(pm, dist_name):
            # Handle other distributions directly if parameters match
            return getattr(pm, dist_name)(name, **params)
        else:
            raise ValueError(f"Unknown distribution name: {dist_name}")

    except Exception as e:
        logger.exception(f"Error creating PyMC distribution '{dist_name}' for '{name}' with params {params}: {e}")
        raise

class PathIntegralOptimizer:
    """
    A class for performing path integral optimization using MCMC,
    allowing for uncertainty in model parameters 'base_cost', 'base_benefit',
    'scale_benefit', and GP-based d(t).
    """
    def __init__(self,
                 base_cost_prior: Dict[str, Any],
                 base_benefit_prior: Dict[str, Any],
                 scale_benefit_prior: Dict[str, Any],
                 gp_eta_prior: Dict[str, Any],
                 gp_ell_prior: Dict[str, Any],
                 gp_mean_prior: Dict[str, Any],
                 total_resource: float,
                 T: int,
                 hbar: float,
                 num_steps: int,
                 burn_in: int,
                 historical_t: Optional[np.ndarray] = None,
                 historical_input: Optional[np.ndarray] = None) -> None:
        """Initializes the PathIntegralOptimizer.

        Args:
            base_cost_prior (Dict): Prior definition for parameter 'base_cost'.
                                    Ex: {"dist": "TruncatedNormal", "mu": 0.5, "sigma": 0.05, "lower": 0.0}
            base_benefit_prior (Dict): Prior definition for parameter 'base_benefit'.
                                       Ex: {"dist": "TruncatedNormal", "mu": 1.0, "sigma": 0.1, "lower": 0.0}
            scale_benefit_prior (Dict): Prior definition for parameter 'scale_benefit'.
                                        Ex: {"dist": "Beta", "alpha": 5, "beta": 5} or {"dist": "Beta", "mu": 0.5, "sigma": 0.1}
            gp_eta_prior (Dict): Prior for GP amplitude 'eta'. Ex: {"dist": "HalfNormal", "sigma": 1}
            gp_ell_prior (Dict): Prior for GP lengthscale 'ell'. Ex: {"dist": "Gamma", "alpha": 5, "beta": 1} or {"dist": "Gamma", "mu": 5, "sigma": 5}
            gp_mean_prior (Dict): Prior for GP mean 'mean_d'. Ex: {"dist": "Normal", "mu": 2, "sigma": 0.5}
            total_resource (float): Fixed total resource.
            T (int): Fixed number of time steps for the optimization horizon (historical + forecast).
            hbar (float): Fixed noise parameter.
            num_steps (int): Number of MCMC steps.
            burn_in (int): Number of burn-in steps.
            historical_t (Optional[np.ndarray]): Time steps of the historical input data.
            historical_input (Optional[np.ndarray]): Values of the historical input data.
        """
        # Validate that all required prior definitions are dictionaries with 'dist'
        if not isinstance(base_cost_prior, Dict) or 'dist' not in base_cost_prior:
             raise ValueError('base_cost_prior must be a dictionary with a "dist" key')
        if not isinstance(base_benefit_prior, Dict) or 'dist' not in base_benefit_prior:
             raise ValueError('base_benefit_prior has to be of type dict with a "dist" key')
        if not isinstance(scale_benefit_prior, Dict) or 'dist' not in scale_benefit_prior:
             raise ValueError('scale_benefit_prior has to be of type dict with a "dist" key')
        if not isinstance(gp_eta_prior, Dict) or 'dist' not in gp_eta_prior:
             raise ValueError('gp_eta_prior has to be of type dict with a "dist" key')
        if not isinstance(gp_ell_prior, Dict) or 'dist' not in gp_ell_prior:
             raise ValueError('gp_ell_prior has to be of type dict with a "dist" key')
        if not isinstance(gp_mean_prior, Dict) or 'dist' not in gp_mean_prior:
             raise ValueError('gp_mean_prior has to be of type dict with a "dist" key')

        #config.mode = 'JAX'  # very slow

        # Store priors and fixed values
        self.base_cost_prior_def: Dict[str, Any] = base_cost_prior
        self.base_benefit_prior_def: Dict[str, Any] = base_benefit_prior
        self.scale_benefit_prior_def: Dict[str, Any] = scale_benefit_prior
        self.gp_eta_prior_def: Dict[str, Any] = gp_eta_prior
        self.gp_ell_prior_def: Dict[str, Any] = gp_ell_prior
        self.gp_mean_prior_def: Dict[str, Any] = gp_mean_prior

        self.total_resource: float = total_resource
        self.T: int = T
        self.hbar: float = hbar
        self.num_steps: int = num_steps
        self.burn_in: int = burn_in
        self.historical_t: Optional[np.ndarray] = historical_t
        self.historical_input: Optional[np.ndarray] = historical_input

        # Initialize containers
        self.mcmc_paths: Optional[np.ndarray] = None  # Store all paths from trace later
        self.actions: Optional[np.ndarray] = None     # Store actions for each sampled path/parameter combo
        self.trace: Optional[az.InferenceData] = None # Store the full InferenceData

    def compute_action(self,
                       x_path: pt.TensorVariable,
                       base_cost: pt.TensorVariable, # base_cost is now a variable
                       base_benefit: pt.TensorVariable,
                       scale_benefit: pt.TensorVariable,
                       d_t: pt.TensorVariable) -> pt.TensorVariable:
        """Computes the action using PyTensor, accepting parameters as variables."""
        try:
            # Use passed parameters
            benefit = base_benefit * x_path ** scale_benefit
            cost = base_cost * x_path ** d_t  # Use base_cost variable
            return -pt.sum(benefit - cost)
        except Exception as e:
            logger.exception(f"Error in compute_action: {e}")
            raise

    def _compute_action_numpy(self,
                             x_path: np.ndarray,
                             base_cost: float, # base_cost is now a float argument
                             base_benefit: float,
                             scale_benefit: float,
                             d_t: np.ndarray) -> float:
        """Computes the action using NumPy, accepting specific parameter values."""
        try:
            # Ensure inputs are valid numbers
            if not (np.isfinite(base_cost) and np.isfinite(base_benefit) and np.isfinite(scale_benefit)):
                logger.warning(f"Non-finite parameter values passed: base_cost={base_cost}, base_benefit={base_benefit}, scale_benefit={scale_benefit}")
                return np.inf

            # Add clipping/validation for base_cost
            base_cost = np.clip(base_cost, a_min=1e-12, a_max=1e6) # Ensure positive and finite

            x_path = np.clip(x_path, a_min=1e-12, a_max=1e6)  #Adjust min/max based on problem
            base_benefit = np.clip(base_benefit, a_min=0.0, a_max=1e6)
            scale_benefit = np.clip(scale_benefit, a_min=0.0, a_max=1.0) # scale_benefit is typically between 0 and 1

            # Ensure x_path has no non-positive values before exponentiation, esp. with scale_benefit<1
            if np.any(x_path <= 0):
                 logger.warning(f"Path contains non-positive values: {x_path}. Setting action to inf.")
                 return np.inf
            x_safe = x_path # Use directly if checks pass, or add epsilon if needed based on errors

            # Check for potential issues with scale_benefit < 1 and x near 0
            benefit = base_benefit * np.power(x_safe + 1e-12, scale_benefit)
            if not np.all(np.isfinite(benefit)):
                logger.warning(f"Non-finite benefit values: {benefit}")
                return np.inf

            cost = base_cost * x_safe ** d_t # Use base_cost argument
            if not np.all(np.isfinite(cost)):
                logger.warning(f"Non-finite cost values: {cost}")
                return np.inf

            action = -np.sum(benefit - cost)

            if not np.isfinite(action):
                 logger.warning(f"Non-finite action computed for path: {x_path}, base_cost={base_cost:.3f}, base_benefit={base_benefit:.3f}, scale_benefit={scale_benefit:.3f}. Action: {action}")
                 return np.inf
            return float(action)

        except FloatingPointError as fpe:
             logger.warning(f"Floating point error in _compute_action_numpy for path: {x_path}, base_cost={base_cost:.3f}, base_benefit={base_benefit:.3f}, scale_benefit={scale_benefit:.3f}. Error: {fpe}")
             return np.inf
        except Exception as e:
            logger.exception(f"Error in _compute_action_numpy (base_cost={base_cost:.3f}, base_benefit={base_benefit:.3f}, scale_benefit={scale_benefit:.3f}): {e}")
            raise

    def run_mcmc(self) -> None:
        """Runs the NUTS simulation inferring 'base_cost', 'base_benefit', 'scale_benefit', and the path 'x_path'."""
        logger.info("Starting PyTensor/PyMC MCMC sampling for x_path, base_cost, base_benefit, scale_benefit, and GP-based d(t)...")
        try:
            with pm.Model(coords={"t": np.arange(self.T)}) as model:
                # --- Priors for Parameters ---
                # Use get_pymc_distribution with the stored prior definitions
                base_cost = get_pymc_distribution("base_cost", self.base_cost_prior_def)
                base_benefit = get_pymc_distribution("base_benefit", self.base_benefit_prior_def)
                scale_benefit = get_pymc_distribution("scale_benefit", self.scale_benefit_prior_def)

                # --- GP for d(t): Prior Hyperparameters ---
                eta = get_pymc_distribution("eta", self.gp_eta_prior_def)
                ell = get_pymc_distribution("ell", self.gp_ell_prior_def)
                mean_d = get_pymc_distribution("mean_d", self.gp_mean_prior_def)

                # --- GP Covariance Function ---
                # FIX: Wrap ell + 1e-9 in pt.as_tensor_variable to ensure correct handling
                if ell is None:
                    raise ValueError("Lengthscale 'ell' cannot be None. Check the prior definition.")
                if eta is None:
                    raise ValueError("Amplitude 'eta' cannot be None. Check the prior definition.")
                cov_d = eta**2 * pm.gp.cov.ExpQuad(1, ls=pt.as_tensor_variable(ell + 1e-9)) + pm.gp.cov.WhiteNoise(1e-6)  # Add jitter

                # --- GP Definition ---
                X = np.arange(1, self.T + 1)[:, None]  # Input: time steps as 2D array
                X = (X - np.mean(X)) / np.std(X)  # Standardize input for GP

                gp_d = pm.gp.Latent(
                    mean_func=pm.gp.mean.Constant(mean_d),
                    cov_func=cov_d
                )
                f_d = gp_d.prior("f_d", X=X)  # Latent GP function values
                # Transform f_d to ensure d(t) > 1
                # softplus(x) > 0, so softplus(x) + 1 > 1. Add 1e-6 for numerical stability.
                d_t = pm.Deterministic("d_t", pt.softplus(f_d) + 1 + 1e-6)

                # --- Prior for Path (Reparameterized) ---
                x_raw = pm.Normal("x_raw", mu=0, sigma=1, dims="t")
                softmax_x_raw = pt.exp(x_raw) / pt.exp(x_raw).sum(axis=0)
                x_path = pm.Deterministic("x_path", softmax_x_raw * self.total_resource, dims="t")

                # --- Potentials ---
                # Constraints (optional but good practice)
                pm.Potential("non_negative_path", pt.switch(pt.any(x_path < 0), -np.inf, 0))
                pm.Potential("finite_check_path", pt.switch(pt.any(pt.isnan(x_path)), -np.inf, 0))

                # Action Potential: depends on sampled base_cost, base_benefit, scale_benefit and fixed self.hbar
                action = self.compute_action(x_path, base_cost, base_benefit, scale_benefit, d_t) # Pass base_cost variable
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
            c_samples = self.trace.posterior["base_cost"].values.flatten() # Extract base_cost samples
            a_samples = self.trace.posterior["base_benefit"].values.flatten()
            b_samples = self.trace.posterior["scale_benefit"].values.flatten()
            # d_t is now a Deterministic variable, extract directly
            d_samples = self.trace.posterior["d_t"].values.reshape(-1, self.T)

            # Calculate action for each sample using corresponding sampled parameters
            num_samples = len(self.mcmc_paths)
            self.actions = np.full(num_samples, np.nan, dtype=float) # Initialize with NaN
            for i in range(num_samples):
                self.actions[i] = self._compute_action_numpy(
                    x_path=self.mcmc_paths[i],
                    base_cost=c_samples[i], # Pass sampled base_cost
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
        """Generates summary of the MCMC results.

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

            # Get summary statistics for path
            mean_path = idata.posterior.x_path.mean(dim=("chain", "draw")).values
            std_path = idata.posterior.x_path.std(dim=("chain", "draw")).values

            # Calculate action values from posterior samples using the NumPy version
            x_path_samples = idata.posterior.x_path.values.reshape(-1, self.T)
            c_samples = idata.posterior["base_cost"].values.flatten() # Get base_cost samples
            a_samples = idata.posterior["base_benefit"].values.flatten() # Get base_benefit samples
            b_samples = idata.posterior["scale_benefit"].values.flatten() # Get scale_benefit samples
            d_samples = idata.posterior["d_t"].values.reshape(-1, self.T) # Get d_t samples

            # Pass all sampled parameters to _compute_action_numpy
            action_values = np.array([self._compute_action_numpy(x, c, a, b, d) for x, c, a, b, d in zip(x_path_samples, c_samples, a_samples, b_samples, d_samples)])

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

            # --- Parameter Summaries ---
            # Note: Cannot add base_cost summary to PathIntegralOptimizerResult as it's read-only.
            # The result object only has fields for base_benefit, scale_benefit, base_cost (fixed float), etc.
            # It does NOT have fields for the posterior summaries of base_cost, base_benefit, scale_benefit, GP params.
            # This summary generation needs to be updated if PathIntegralOptimizerResult is modified.
            logger.warning("PathIntegralOptimizerResult object structure is read-only and does not support adding summaries for inferred parameters (base_cost, base_benefit, scale_benefit, GP params). The generated result object will only contain fixed/prior info and path/action summaries.")

            # Create and return the result object
            # Note: The result object expects the *prior definitions* for base_benefit and scale_benefit,
            # and the *fixed* base_cost value. This doesn't align with summarizing the *posterior*
            # of the inferred parameters. This part of the code might need review depending on
            # the intended use of PathIntegralOptimizerResult. For now, populating based on
            # the existing structure.
            result = PathIntegralOptimizerResult(
                base_benefit=self.base_benefit_prior_def, # Using prior def as per current Result structure
                scale_benefit=self.scale_benefit_prior_def, # Using prior def as per current Result structure
                # Attempt to get a representative value for base_cost from its prior definition
                base_cost=self.base_cost_prior_def.get('mu', np.nan), # Using mu from prior def as a placeholder
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
                # Cannot add GP parameter summaries or posterior base_cost/base_benefit/scale_benefit summaries here
            )
            return result

        except Exception as e:
            logger.exception(f"Error in generate_summary: {e}")
            raise
        # return None # Ensure None is returned on exception path as well - already handled by re-raising

    def plot_forecast(self, output_file: Optional[str] = None) -> None:
        """
        Plots the historical input data followed by the mean forecasted input data.
        A vertical line indicates the transition from historical to forecasted data.

        Args:
            output_file (Optional[str]): If provided, saves the plot to this file.
                                         Otherwise, shows the plot.
        """
        if self.historical_input is None or self.historical_t is None:
            logger.warning("Historical data not available. Cannot plot forecast.")
            return
        if self.mcmc_paths is None:
            logger.warning("MCMC paths not available. Run run_mcmc() first. Cannot plot forecast.")
            return
        if self.T is None or len(self.historical_t) >= self.T :
            logger.warning("No forecast steps available (T <= historical length). Cannot plot forecast.")
            return


        try:
            plt.figure(figsize=(12, 7))

            # Plot historical data
            plt.plot(self.historical_t, self.historical_input, label="Historical Input", color='blue', marker='o', linestyle='-')

            # Calculate mean of optimized paths (this covers the full T horizon)
            mean_optimized_path = np.mean(self.mcmc_paths, axis=0)

            num_historical_points = len(self.historical_t)
            forecast_segment = mean_optimized_path[num_historical_points:]
            
            # Create time axis for the forecast segment
            # Assuming historical_t are integers and regularly spaced (e.g., step of 1)
            last_historical_time = self.historical_t[-1]
            # Determine step from historical data if possible, otherwise assume 1
            time_step = 1
            if num_historical_points > 1:
                time_step = self.historical_t[1] - self.historical_t[0]

            forecast_time_axis = last_historical_time + np.arange(1, len(forecast_segment) + 1) * time_step

            # Plot forecasted data
            plt.plot(forecast_time_axis, forecast_segment, label="Mean Forecasted Input", color='red', marker='x', linestyle='--')

            # Add a vertical line at the end of historical data
            plt.axvline(x=last_historical_time, color='gray', linestyle=':', linewidth=2, label="Forecast Horizon Start")

            plt.xlabel("Time (t)")
            plt.ylabel("Input / Allocation (x(t))")
            plt.title("Historical Input and Mean Forecasted Path")
            plt.legend()
            plt.grid(True)
            
            if output_file:
                plt.savefig(output_file)
                logger.info(f"Forecast plot saved to {output_file}")
            else:
                plt.show()

        except Exception as e:
            logger.exception(f"Error in plot_forecast: {e}")
            # Do not re-raise, just log, so other operations can continue if needed.

    @staticmethod
    def from_data(
        input:np.array,
        cost:np.array,
        benefit:np.array,
        total_resource:float,
        hbar:float,
        num_steps:int,
        burn_in:int,
        forecast_steps: int,
        t:Optional[np.array]=None
        ):
        _t_hist=t or np.arange(1, len(input)+1)
        optimizer_T = len(_t_hist) + forecast_steps

        data:Dataset=Dataset(
            t=_t_hist,
            input=input,
            cost=cost,
            benefit=benefit
        )

        logger.info('Building model from input data')
        # ParameterEstimator returns ParameterEstimationResult which contains
        # {'mu': ..., 'sigma': ...} for each estimated parameter.
        parameters:ParameterEstimationResult=ParameterEstimator(data) \
            .get_parameters()

        # Construct the full prior dictionaries from the estimation results
        # based on the desired distribution types.
        base_cost_prior_dict = {
            'dist': 'TruncatedNormal',
            'mu': parameters.base_cost['mu'],
            'sigma': parameters.base_cost['sigma'],
            'lower': 0.0 # base_cost must be positive
        }
        base_benefit_prior_dict = {
            'dist': 'TruncatedNormal',
            'mu': parameters.base_benefit['mu'],
            'sigma': parameters.base_benefit['sigma'],
            'lower': 0.0 # base_benefit must be positive
        }
        scale_benefit_prior_dict = {
            'dist': 'Beta',
            'mu': parameters.scale_benefit['mu'],
            'sigma': parameters.scale_benefit['sigma']
            # Beta distribution handles the (0, 1) range
        }
        gp_eta_prior_dict = {
            'dist': 'HalfNormal',
            'sigma': parameters.gp_eta_prior['sigma'] # HalfNormal takes sigma
        }
        gp_ell_prior_dict = {
            'dist': 'Gamma',
            'mu': parameters.gp_ell_prior['mu'], # Gamma conversion from mu/sigma
            'sigma': parameters.gp_ell_prior['sigma']
        }
        gp_mean_prior_dict = {
            'dist': 'Normal',
            'mu': parameters.gp_mean_prior['mu'],
            'sigma': parameters.gp_mean_prior['sigma']
        }

        # Pass the constructed prior dictionaries to the constructor
        return PathIntegralOptimizer(
            base_cost_prior=base_cost_prior_dict,
            base_benefit_prior=base_benefit_prior_dict,
            scale_benefit_prior=scale_benefit_prior_dict,
            gp_eta_prior=gp_eta_prior_dict,
            gp_ell_prior=gp_ell_prior_dict,
            gp_mean_prior=gp_mean_prior_dict,
            total_resource=total_resource,
            T=optimizer_T,
            hbar=hbar,
            num_steps=num_steps,
            burn_in=burn_in,
            historical_t=_t_hist,
            historical_input=input
        )
