import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Callable
import pymc as pm
from pytensor import tensor as pt
from pytensor import config
from loguru import logger
import arviz as az
from pydantic import BaseModel, Field, validator, root_validator
from typing import Literal, Union

from .path_integral_optimizer_result import PathIntegralOptimizerResult, ParameterSummary
from .parameter_estimator import ParameterEstimator
from .parameter_estimation_result import ParameterEstimationResult
from .dataset import Dataset
from .prior_definition import PriorDefinition

# --- Main Optimizer Class ---

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
            T (int): Fixed number of time steps for the optimization/forecast horizon.
            hbar (float): Fixed noise parameter.
            num_steps (int): Number of MCMC steps.
            burn_in (int): Number of burn-in steps.
            historical_t (Optional[np.ndarray]): Time steps of the historical input data.
            historical_input (Optional[np.ndarray]): Values of the historical input data.
        """
        # Validate priors using Pydantic models
        try:
            self.base_cost_prior_def = PriorDefinition(**base_cost_prior)
            self.base_benefit_prior_def = PriorDefinition(**base_benefit_prior)
            self.scale_benefit_prior_def = PriorDefinition(**scale_benefit_prior)
            self.gp_eta_prior_def = PriorDefinition(**gp_eta_prior)
            self.gp_ell_prior_def = PriorDefinition(**gp_ell_prior)
            self.gp_mean_prior_def = PriorDefinition(**gp_mean_prior)
        except Exception as e:
            logger.exception(f"Error validating prior definitions: {e}")
            raise

        # Validate historical data if provided
        if historical_t is not None:
            if not isinstance(historical_t, np.ndarray) or historical_t.ndim != 1:
                raise ValueError("historical_t must be a 1D numpy array or None.")
            if historical_input is None:
                 raise ValueError("historical_input must be provided if historical_t is provided.")
            if not isinstance(historical_input, np.ndarray) or historical_input.ndim != 1:
                raise ValueError("historical_input must be a 1D numpy array or None.")
            if len(historical_t) != len(historical_input):
                raise ValueError("historical_t and historical_input must have the same length.")

        # Store fixed values and validated historical data
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
                       base_cost: pt.TensorVariable,
                       base_benefit: pt.TensorVariable,
                       scale_benefit: pt.TensorVariable,
                       d_t: pt.TensorVariable) -> pt.TensorVariable:
        """Computes the action using PyTensor, accepting parameters as variables."""
        # No change needed here, PyTensor handles symbolic computation
        benefit = base_benefit * x_path ** scale_benefit
        cost = base_cost * x_path ** d_t
        return -pt.sum(benefit - cost)

    def _compute_action_numpy(self,
                             x_path: np.ndarray,
                             base_cost: float,
                             base_benefit: float,
                             scale_benefit: float,
                             d_t: np.ndarray) -> float:
        """Computes the action using NumPy, accepting specific parameter values."""
        try:
            # Basic validation for inputs
            if not (np.isfinite(base_cost) and np.isfinite(base_benefit) and np.isfinite(scale_benefit)):
                return np.inf

            # Add clipping/validation for parameters
            base_cost = np.clip(base_cost, a_min=1e-12, a_max=1e6)
            base_benefit = np.clip(base_benefit, a_min=0.0, a_max=1e6)
            scale_benefit = np.clip(scale_benefit, a_min=0.0, a_max=1.0)

            # Ensure x_path has no non-positive values before exponentiation, esp. with scale_benefit<1
            # Use a small epsilon for numerical stability if needed, but clipping might be better handled
            # by ensuring the MCMC samples stay positive (e.g., through parameterization).
            # For now, check and return inf if non-positive values are present.
            if np.any(x_path <= 0):
                 return np.inf # Return inf without logging for every sample

            # Use a small epsilon for numerical stability in power calculation if needed
            x_safe = x_path + 1e-12 # Add epsilon to avoid log(0) or 0^power issues

            benefit = base_benefit * np.power(x_safe, scale_benefit)
            cost = base_cost * np.power(x_safe, d_t)

            # Check for non-finite results after calculation
            if not np.all(np.isfinite(benefit)) or not np.all(np.isfinite(cost)):
                 return np.inf # Return inf without logging for every sample

            action = -np.sum(benefit - cost)

            if not np.isfinite(action):
                 return np.inf # Return inf without logging for every sample

            return float(action)

        except FloatingPointError as fpe:
             # logger.warning(f"Floating point error in _compute_action_numpy. Error: {fpe}")
             return np.inf # Return inf without logging for every sample
        # Removed general Exception catch
        except Exception as e:
            raise RuntimeError(e) from e

    def run_mcmc(self) -> None:
        """Runs the NUTS simulation inferring 'base_cost', 'base_benefit', 'scale_benefit', and the path 'x_path'."""
        logger.info("Starting PyTensor/PyMC MCMC sampling for x_path, base_cost, base_benefit, scale_benefit, and GP-based d(t)...")
        # Removed try...except Exception block
        with pm.Model(coords={"t": np.arange(self.T)}) as model:
            # --- Priors for Parameters ---
            base_cost = self.base_cost_prior_def.create_pymc_distribution("base_cost")
            base_benefit = self.base_benefit_prior_def.create_pymc_distribution("base_benefit")
            scale_benefit = self.scale_benefit_prior_def.create_pymc_distribution("scale_benefit")

            # --- GP for d(t): Prior Hyperparameters ---
            eta = self.gp_eta_prior_def.create_pymc_distribution("eta")
            ell = self.gp_ell_prior_def.create_pymc_distribution("ell")
            mean_d = self.gp_mean_prior_def.create_pymc_distribution("mean_d")

            # --- GP Covariance Function ---
            # Ensure ell is positive and add jitter
            cov_d = eta**2 * pm.gp.cov.ExpQuad(1, ls=pt.as_tensor_variable(ell + 1e-9)) + pm.gp.cov.WhiteNoise(1e-6)

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
            # Use a Dirichlet distribution over the raw values before softmax
            # This encourages the sum to be total_resource
            x_raw = pm.Dirichlet("x_raw", a=np.ones(self.T), dims="t") # Dirichlet prior
            x_path = pm.Deterministic("x_path", x_raw * self.total_resource, dims="t")

            # --- Potentials ---
            # Action Potential: depends on sampled base_cost, base_benefit, scale_benefit and fixed self.hbar
            action = self.compute_action(x_path, base_cost, base_benefit, scale_benefit, d_t)
            # Ensure action is finite before using in potential
            finite_action = pt.switch(pt.isnan(action) | pt.isinf(action), -np.inf, -action / self.hbar)
            pm.Potential("action", finite_action)

            # --- Sampling ---
            self.trace = pm.sample(
                draws=self.num_steps,
                tune=self.burn_in,
                target_accept=0.95, # Adjusted target_accept
                max_treedepth=15,   # Increased max_treedepth
                chains=4,
                cores=4,
                return_inferencedata=True
            )

        logger.info("MCMC sampling finished. Processing trace...")

        # Extract posterior samples
        # Shape: (chains * draws, T) for path
        self.mcmc_paths = self.trace.posterior["x_path"].values.reshape(-1, self.T)
        # Shape: (chains * draws,) for params
        c_samples = self.trace.posterior["base_cost"].values.flatten()
        a_samples = self.trace.posterior["base_benefit"].values.flatten()
        b_samples = self.trace.posterior["scale_benefit"].values.flatten()
        d_samples = self.trace.posterior["d_t"].values.reshape(-1, self.T)

        # Calculate action for each sample using corresponding sampled parameters
        num_samples = len(self.mcmc_paths)
        self.actions = np.array([
            self._compute_action_numpy(
                x_path=self.mcmc_paths[i],
                base_cost=c_samples[i],
                base_benefit=a_samples[i],
                scale_benefit=b_samples[i],
                d_t=d_samples[i]
            ) for i in range(num_samples)
        ])

        # Log how many actions were finite
        num_finite_actions = np.sum(np.isfinite(self.actions))
        logger.info(f"Computed actions for {num_samples} samples ({num_finite_actions} finite).")
        if num_finite_actions == 0:
             logger.error("No finite actions computed. Cannot plot top paths or generate meaningful summary.")

        # Removed re-raise

    def _get_top_paths_indices(self, top_percent: float) -> np.ndarray:
        """Helper to get indices of top paths based on action."""
        if self.actions is None:
            return np.array([])
        finite_indices = np.where(np.isfinite(self.actions))[0]
        if len(finite_indices) == 0:
            return np.array([])
        sorted_finite_indices = finite_indices[np.argsort(self.actions[finite_indices])]
        num_top_paths = max(1, int(len(finite_indices) * top_percent / 100)) # Calculate based on finite actions
        return sorted_finite_indices[:num_top_paths]

    def plot_top_paths(self, top_percent: float = 5.0) -> None:
        """Plots top paths based on lowest computed action and their mean.

        Args:
            top_percent: Percentage of top paths to plot (default: 5%)
        """
        if self.mcmc_paths is None or self.actions is None:
            logger.warning("MCMC data not available. Run run_mcmc() first.")
            return

        top_indices = self._get_top_paths_indices(top_percent)
        if len(top_indices) == 0:
            logger.warning("No finite actions found or no top paths selected. Cannot plot top paths.")
            return

        # Removed try...except Exception block
        plt.figure(figsize=(12, 7))

        # Plot all top paths with transparency
        for idx in top_indices:
            plt.plot(range(1, self.T + 1), self.mcmc_paths[idx], color='blue', alpha=0.3)

        # Calculate and plot mean of top paths
        top_paths = self.mcmc_paths[top_indices]
        mean_top_path = np.mean(top_paths, axis=0)
        plt.plot(range(1, self.T + 1), mean_top_path, label=f"Mean of Top {top_percent}% Paths",
                color='darkblue', linewidth=2, linestyle='--')

        plt.xlabel("Time (t)")
        plt.ylabel("Allocation (x(t))")
        plt.title(f"Top {top_percent}% MCMC Sampled Allocation Paths (Lowest Action)\n(Showing {len(top_indices)} paths in total)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        # Removed re-raise

    def plot(self, top_percent: float = 5.0) -> None:
        """Diagnostic plot showing distribution of action values with vertical line at mean of top paths.

        Args:
            top_percent: Percentage of top paths to highlight (default: 5%)
        """
        if self.actions is None:
            logger.warning("MCMC actions not available. Run run_mcmc() first.")
            return

        finite_actions = self.actions[np.isfinite(self.actions)]
        if len(finite_actions) == 0:
            logger.warning("No finite actions found. Cannot create diagnostic plot.")
            return

        # Removed try...except Exception block
        top_indices = self._get_top_paths_indices(top_percent)
        top_actions = self.actions[top_indices]

        plt.figure(figsize=(12, 7))

        # Plot KDE of all finite actions
        az.plot_kde(finite_actions, plot_kwargs={'color': 'lightblue'})

        # Add vertical line for mean of top paths
        mean_top_action = np.mean(top_actions) if len(top_actions) > 0 else np.nan
        if np.isfinite(mean_top_action):
            plt.axvline(mean_top_action, color='darkblue', linestyle='--', linewidth=2,
                      label=f'Mean Action of Top {top_percent}% Paths')

        plt.xlabel("Action Value")
        plt.ylabel("Density")
        plt.title(f"Distribution of Path Actions with Top {top_percent}% Highlight\n(Lower action = Higher probability)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        # Removed re-raise

    def generate_summary(self) -> PathIntegralOptimizerResult | None:
        """Generates summary of the MCMC results.

        Returns:
            PathIntegralOptimizerResult | None: An object containing the summary results,
                                                or None if no trace is available.
        """
        # Removed try...except Exception block
        if not hasattr(self, 'trace') or self.trace is None:
            logger.warning("No trace collected. Cannot generate summary.")
            return None

        idata = self.trace # trace is already InferenceData

        # Get summary statistics for path
        mean_path = idata.posterior.x_path.mean(dim=("chain", "draw")).values
        std_path = idata.posterior.x_path.std(dim=("chain", "draw")).values

        # Use pre-calculated actions
        finite_action_values = self.actions[np.isfinite(self.actions)]

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
        # Extract summary statistics for each inferred parameter using ArviZ
        summary_df = az.summary(idata, var_names=["base_cost", "base_benefit", "scale_benefit", "eta", "ell", "mean_d"])

        def create_parameter_summary(param_name: str) -> ParameterSummary:
            """Helper to create ParameterSummary object from ArviZ summary."""
            if param_name not in summary_df.index:
                logger.warning(f"Parameter '{param_name}' not found in summary.")
                # Return a default ParameterSummary with NaNs
                return ParameterSummary(mean=np.nan, sd=np.nan, hdi_3=(np.nan, np.nan), hdi_97=(np.nan, np.nan), mcse_mean=np.nan, mcse_sd=np.nan, ess_bulk=np.nan, ess_tail=np.nan, r_hat=np.nan)

            param_summary = summary_df.loc[param_name]
            return ParameterSummary(
                mean=float(param_summary['mean']),
                sd=float(param_summary['sd']),
                hdi_3=(float(param_summary['hdi_3%']), float(param_summary['hdi_97%'])), # Note: ArviZ uses hdi_3% and hdi_97%
                hdi_97=(float(param_summary['hdi_3%']), float(param_summary['hdi_97%'])), # Using the same for both for simplicity, adjust if different HDIs are needed
                mcse_mean=float(param_summary['mcse_mean']),
                mcse_sd=float(param_summary['mcse_sd']),
                ess_bulk=float(param_summary['ess_bulk']),
                ess_tail=float(param_summary['ess_tail']),
                r_hat=float(param_summary['r_hat'])
            )

        base_cost_summary = create_parameter_summary("base_cost")
        base_benefit_summary = create_parameter_summary("base_benefit")
        scale_benefit_summary = create_parameter_summary("scale_benefit")
        gp_eta_summary = create_parameter_summary("eta")
        gp_ell_summary = create_parameter_summary("ell")
        gp_mean_summary = create_parameter_summary("mean_d")


        # Create and return the result object
        result = PathIntegralOptimizerResult(
            total_resource=self.total_resource,
            T=self.T,
            hbar=self.hbar,
            num_samples=len(self.actions), # Total samples attempted
            num_finite_actions=len(finite_action_values),

            base_cost_summary=base_cost_summary,
            base_benefit_summary=base_benefit_summary,
            scale_benefit_summary=scale_benefit_summary,
            gp_eta_summary=gp_eta_summary,
            gp_ell_summary=gp_ell_summary,
            gp_mean_summary=gp_mean_summary,

            best_action=best_action,
            action_mean=action_mean,
            action_std=action_std,
            mean_path=mean_path,
            std_path=std_path
        )
        return result
        # Removed re-raise

    def _calculate_historical_forecasted_metrics(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Helper to calculate historical and forecasted cost/benefit."""
        if self.historical_input is None or self.historical_t is None or self.trace is None or self.mcmc_paths is None or self.T is None or self.T <= 0:
            raise ValueError("Missing data for historical/forecasted metric calculation.")

        # Removed try...except Exception block
        # Access posterior means for parameters
        base_cost_mean = self.trace.posterior["base_cost"].values.mean()
        base_benefit_mean = self.trace.posterior["base_benefit"].values.mean()
        scale_benefit_mean = self.trace.posterior["scale_benefit"].values.mean()
        d_t_mean = self.trace.posterior["d_t"].values.mean(dim=("chain", "draw")).values

        mean_forecast_path = np.mean(self.mcmc_paths, axis=0)

        # Calculate historical cost and benefit
        hist_len = len(self.historical_input)
        if hist_len > len(d_t_mean):
             logger.warning("Historical input length exceeds d_t length. Cannot calculate historical cost/benefit accurately.")
             historical_cost = np.full_like(self.historical_input, np.nan)
             historical_benefit = np.full_like(self.historical_input, np.nan)
        else:
            historical_d_t = d_t_mean[:hist_len]
            historical_cost = base_cost_mean * self.historical_input ** historical_d_t
            historical_benefit = base_benefit_mean * self.historical_input ** scale_benefit_mean

        # Calculate forecasted cost and benefit
        forecast_len = len(mean_forecast_path)
        if len(d_t_mean) < hist_len + forecast_len:
             logger.warning("d_t length is less than historical + forecast length. Cannot calculate forecasted cost/benefit accurately.")
             forecast_cost = np.full_like(mean_forecast_path, np.nan)
             forecast_benefit = np.full_like(mean_forecast_path, np.nan)
        else:
            forecast_d_t = d_t_mean[hist_len : hist_len + forecast_len]
            forecast_cost = base_cost_mean * mean_forecast_path ** forecast_d_t
            forecast_benefit = base_benefit_mean * mean_forecast_path ** scale_benefit_mean

        return historical_cost, historical_benefit, forecast_cost, forecast_benefit
        # Removed re-raise


    def plot_forecast(self, output_file: Optional[str] = None) -> None:
        """
        Creates three subplots showing historical data and forecasts for input, cost, and benefit.

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
        if self.T is None or self.T <= 0:
            logger.warning("No forecast period defined (T <= 0). Cannot plot forecast.")
            return
        if self.trace is None:
             logger.warning("No MCMC trace available. Cannot calculate historical/forecasted cost/benefit.")
             # Still plot input if possible
             plot_cost_benefit = False
        else:
             plot_cost_benefit = True


        # Removed try...except Exception block
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 18))
        ax1, ax2, ax3 = axes

        # Calculate time axis for forecast
        last_historical_time = self.historical_t[-1]
        time_step = 1
        if len(self.historical_t) > 1:
            time_step = self.historical_t[1] - self.historical_t[0]
        forecast_time_axis = last_historical_time + np.arange(1, self.T + 1) * time_step

        # Get mean forecast path
        mean_forecast_path = np.mean(self.mcmc_paths, axis=0)

        # Plot input data
        ax1.plot(self.historical_t, self.historical_input, label="Historical Input",
                color='blue', marker='o', linestyle='-')
        ax1.plot(forecast_time_axis, mean_forecast_path, label="Mean Forecasted Input",
                color='red', marker='x', linestyle='--')
        ax1.axvline(x=last_historical_time, color='gray', linestyle=':', linewidth=2,
                   label="Forecast Horizon Start")
        ax1.set_xlabel("Time (t)")
        ax1.set_ylabel("Allocation (x(t))")
        ax1.set_title("Historical Input and Mean Forecasted Allocation")
        ax1.legend()
        ax1.grid(True)

        if plot_cost_benefit:
            historical_cost, historical_benefit, forecast_cost, forecast_benefit = self._calculate_historical_forecasted_metrics()

            if historical_cost is not None and historical_benefit is not None and forecast_cost is not None and forecast_benefit is not None:
                # Plot historical cost
                ax2.plot(self.historical_t, historical_cost,
                        label="Historical Cost (Estimated)", color='green', marker='o', linestyle='-')

                # Plot forecasted cost
                ax2.plot(forecast_time_axis, forecast_cost,
                        label="Forecasted Cost (Mean)", color='darkgreen', marker='x', linestyle='--')

                # Plot historical benefit
                ax3.plot(self.historical_t, historical_benefit,
                        label="Historical Benefit (Estimated)", color='orange', marker='o', linestyle='-')

                # Plot forecasted benefit
                ax3.plot(forecast_time_axis, forecast_benefit,
                        label="Forecasted Benefit (Mean)", color='darkorange', marker='x', linestyle='--')
            else:
                logger.warning("Skipping cost/benefit plots due to calculation errors.")


        ax2.axvline(x=last_historical_time, color='gray', linestyle=':', linewidth=2,
                   label="Forecast Horizon Start")
        ax2.set_xlabel("Time (t)")
        ax2.set_ylabel("Cost Value")
        ax2.set_title("Historical and Forecasted Cost")
        ax2.legend()
        ax2.grid(True)

        ax3.axvline(x=last_historical_time, color='gray', linestyle=':', linewidth=2,
                   label="Forecast Horizon Start")
        ax3.set_xlabel("Time (t)")
        ax3.set_ylabel("Benefit Value")
        ax3.set_title("Historical and Forecasted Benefit")
        ax3.legend()
        ax3.grid(True)

        # Add overall title and adjust layout
        plt.suptitle("Historical Data and Forecasts for Input, Cost, and Benefit Metrics", y=0.95)
        plt.tight_layout()

        # Save or show the plot
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Forecast plot saved to {output_file}")
        else:
            plt.show()
            logger.info("Forecast plot displayed")
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
        """
        Creates a PathIntegralOptimizer instance by first estimating parameters
        from historical data and using the estimates to define priors.
        """
        if t is None:
            _t_hist = np.arange(1, len(input) + 1)
        else:
            _t_hist = t

        data:Dataset=Dataset(
            t=_t_hist,
            input=input,
            cost=cost,
            benefit=benefit
        )

        logger.info('Estimating parameters from input data to define priors...')

        # ParameterEstimator returns ParameterEstimationResult which contains
        # {'mu': ..., 'sigma': ...} for each estimated parameter.
        parameters:ParameterEstimationResult=ParameterEstimator(data) \
            .get_parameters()

        # Construct the full prior dictionaries from the estimation results
        # based on the desired distribution types.
        # Use Pydantic models for validation during construction

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
            T=forecast_steps, # T is the forecast period length
            hbar=hbar,
            num_steps=num_steps,
            burn_in=burn_in,
            historical_t=_t_hist,
            historical_input=input
        )
