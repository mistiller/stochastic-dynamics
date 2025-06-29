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

from .path_integral_optimizer_result import (
    PathIntegralOptimizerResult,
    ParameterSummary,
)
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

    def __init__(
        self,
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
        historical_input: Optional[np.ndarray] = None,
    ) -> None:
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
            T (int): Fixed number of time steps for the optimization/forecast horizon. Must be >= 1.
            hbar (float): Fixed noise parameter.
            num_steps (int): Number of MCMC steps (recommended >= 4000).
            burn_in (int): Number of burn-in steps (recommended >= 2000).
            historical_t (Optional[np.ndarray]): Time steps of the historical input data.
            historical_input (Optional[np.ndarray]): Values of the historical input data.
        """
        if T < 1:
            raise ValueError("T (forecast horizon) must be at least 1.")

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
                raise ValueError(
                    "historical_input must be provided if historical_t is provided."
                )
            if (
                not isinstance(historical_input, np.ndarray)
                or historical_input.ndim != 1
            ):
                raise ValueError("historical_input must be a 1D numpy array or None.")
            if len(historical_t) != len(historical_input):
                raise ValueError(
                    "historical_t and historical_input must have the same length."
                )

        # Store fixed values and validated historical data
        self.total_resource: float = total_resource
        self._T: int = T  # Make T immutable after init
        self.hbar: float = hbar

    @property
    def T(self) -> int:
        """Read-only property for time horizon"""
        return self._T
        self.num_steps: int = num_steps
        self.burn_in: int = burn_in
        self.historical_t: Optional[np.ndarray] = historical_t
        self.historical_input: Optional[np.ndarray] = historical_input

        self.mcmc_paths: Optional[np.ndarray] = None
        self.actions: Optional[np.ndarray] = None
        self.trace: Optional[az.InferenceData] = None

    def compute_action(
        self,
        x_path: pt.TensorVariable,
        base_cost: pt.TensorVariable,
        base_benefit: pt.TensorVariable,
        scale_benefit: pt.TensorVariable,
        d_t: pt.TensorVariable,
    ) -> pt.TensorVariable:
        """Computes the action using PyTensor, accepting parameters as variables."""
        benefit = base_benefit * x_path**scale_benefit
        cost = base_cost * x_path**d_t
        return -pt.sum(benefit - cost)

    def _compute_action_numpy(
        self,
        x_path: np.ndarray,
        base_cost: float,
        base_benefit: float,
        scale_benefit: float,
        d_t: np.ndarray,
    ) -> float:
        """Computes the action using NumPy, accepting specific parameter values."""
        try:
            if not (
                np.isfinite(base_cost)
                and np.isfinite(base_benefit)
                and np.isfinite(scale_benefit)
            ):
                return np.inf

            base_cost = np.clip(base_cost, a_min=1e-12, a_max=1e6)
            base_benefit = np.clip(base_benefit, a_min=0.0, a_max=1e6)
            scale_benefit = np.clip(scale_benefit, a_min=0.0, a_max=1.0)

            if np.any(x_path <= 0):
                return np.inf

            x_safe = x_path + 1e-12

            benefit = base_benefit * np.power(x_safe, scale_benefit)
            cost = base_cost * np.power(x_safe, d_t)

            if not np.all(np.isfinite(benefit)) or not np.all(np.isfinite(cost)):
                return np.inf

            action = -np.sum(benefit - cost)

            if not np.isfinite(action):
                return np.inf

            return float(action)

        except FloatingPointError:
            return np.inf
        except Exception as e:
            raise RuntimeError(e) from e

    def run_mcmc(self) -> None:
        """Runs the NUTS simulation inferring 'base_cost', 'base_benefit', 'scale_benefit', and the path 'x_path'."""
        logger.info(
            "Starting PyTensor/PyMC MCMC sampling for x_path, base_cost, base_benefit, scale_benefit, and GP-based d(t)..."
        )
        with pm.Model(coords={"t": np.arange(self.T)}) as model:
            base_cost = self.base_cost_prior_def.create_pymc_distribution("base_cost")
            base_benefit = self.base_benefit_prior_def.create_pymc_distribution(
                "base_benefit"
            )
            scale_benefit = self.scale_benefit_prior_def.create_pymc_distribution(
                "scale_benefit"
            )

            eta = self.gp_eta_prior_def.create_pymc_distribution("eta")
            ell = self.gp_ell_prior_def.create_pymc_distribution("ell")
            mean_d = self.gp_mean_prior_def.create_pymc_distribution("mean_d")

            cov_d = eta**2 * pm.gp.cov.ExpQuad(
                1, ls=pt.as_tensor_variable(ell + 1e-9)
            ) + pm.gp.cov.WhiteNoise(1e-6)

            # Prepare input for GP: time points for the forecast horizon
            X_coords = np.arange(self.T)[:, None].astype(config.floatX)
            if self.T > 1:
                X_coords_std = np.std(X_coords)
                if X_coords_std > 1e-9: # Avoid division by zero if all points are the same (e.g. T=1)
                    X_coords = (X_coords - np.mean(X_coords)) / X_coords_std
                else: # Handles T=1 or cases where all X_coords are identical
                    X_coords = np.zeros_like(X_coords)
            elif self.T == 1: # Explicitly handle T=1
                 X_coords = np.array([[0.0]], dtype=config.floatX)


            gp_d = pm.gp.Latent(mean_func=pm.gp.mean.Constant(mean_d), cov_func=cov_d)
            f_d = gp_d.prior("f_d", X=X_coords)
            # Transform f_d to ensure d(t) > 1 (softplus(x) > 0)
            d_t = pm.Deterministic("d_t", pt.softplus(f_d) + 1 + 1e-6)

            # Dirichlet prior encourages the sum to be total_resource
            x_raw = pm.Dirichlet(
                "x_raw", a=np.ones(self.T), dims="t"
            ) # type: ignore
            x_path = pm.Deterministic("x_path", x_raw * self.total_resource, dims="t")

            action = self.compute_action(
                x_path, base_cost, base_benefit, scale_benefit, d_t
            )
            finite_action = pt.switch(
                pt.isnan(action) | pt.isinf(action), -np.inf, -action / self.hbar
            )
            pm.Potential("action", finite_action)

            self.trace = pm.sample(
                draws=self.num_steps,
                tune=self.burn_in,
                target_accept=0.95,
                max_treedepth=12,
                chains=8,
                cores=8,
                return_inferencedata=True,
            )

        logger.info("MCMC sampling finished. Processing trace...")

        if self.trace is None: # Should not happen if pm.sample ran
            logger.error("MCMC sampling did not produce a trace.")
            return

        self.mcmc_paths = self.trace.posterior["x_path"].values.reshape(-1, self.T)
        c_samples = self.trace.posterior["base_cost"].values.flatten()
        a_samples = self.trace.posterior["base_benefit"].values.flatten()
        b_samples = self.trace.posterior["scale_benefit"].values.flatten()
        d_samples = self.trace.posterior["d_t"].values.reshape(-1, self.T)

        num_samples = len(self.mcmc_paths)
        self.actions = np.array(
            [
                self._compute_action_numpy(
                    x_path=self.mcmc_paths[i],
                    base_cost=c_samples[i],
                    base_benefit=a_samples[i],
                    scale_benefit=b_samples[i],
                    d_t=d_samples[i],
                )
                for i in range(num_samples)
            ]
        )

        # Log how many actions were finite
        num_finite_actions = np.sum(np.isfinite(self.actions))
        logger.info(
            f"Computed actions for {num_samples} samples ({num_finite_actions} finite)."
        )
        if num_finite_actions == 0:
            logger.error(
                "No finite actions computed. Cannot plot top paths or generate meaningful summary."
            )

    def _get_top_paths_indices(self, top_percent: float) -> np.ndarray:
        """Helper to get indices of top paths based on action."""
        if self.actions is None:
            return np.array([])
        finite_indices = np.where(np.isfinite(self.actions))[0]
        if len(finite_indices) == 0:
            return np.array([])
        sorted_finite_indices = finite_indices[np.argsort(self.actions[finite_indices])]
        num_top_paths = max(
            1, int(len(finite_indices) * top_percent / 100)
        )
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
            logger.warning(
                "No finite actions found or no top paths selected. Cannot plot top paths."
            )
            return

        plt.figure(figsize=(12, 7))

        # Plot individual top paths
        time_axis = np.arange(1, self.T + 1)
        for idx in top_indices:
            plt.plot(
                time_axis, self.mcmc_paths[idx], color="blue", alpha=0.3
            )

        # Plot mean of top paths
        top_paths = self.mcmc_paths[top_indices]
        mean_top_path = np.mean(top_paths, axis=0)
        plt.plot(
            time_axis,
            mean_top_path,
            label=f"Mean of Top {top_percent}% Paths",
            color="darkblue",
            linewidth=2,
            linestyle="--",
        )

        plt.xlabel("Time (t)")
        plt.ylabel("Allocation (x(t))")
        plt.title(
            f"Top {top_percent}% MCMC Sampled Allocation Paths (Lowest Action)\n(Showing {len(top_indices)} paths in total)"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

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

        top_indices = self._get_top_paths_indices(top_percent)
        top_actions = self.actions[top_indices] if len(top_indices) > 0 else np.array([])


        plt.figure(figsize=(12, 7))

        az.plot_kde(finite_actions, plot_kwargs={"color": "lightblue"})

        mean_top_action = np.mean(top_actions) if len(top_actions) > 0 else np.nan
        if np.isfinite(mean_top_action):
            plt.axvline(
                mean_top_action,
                color="darkblue",
                linestyle="--",
                linewidth=2,
                label=f"Mean Action of Top {top_percent}% Paths",
            )

        plt.xlabel("Action Value")
        plt.ylabel("Density")
        plt.title(
            f"Distribution of Path Actions with Top {top_percent}% Highlight\n(Lower action = Higher probability)"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def generate_summary(self) -> PathIntegralOptimizerResult | None:
        """Generates summary of the MCMC results.

        Returns:
            PathIntegralOptimizerResult | None: An object containing the summary results,
                                                or None if no trace is available.
        """
        if not hasattr(self, "trace") or self.trace is None:
            logger.warning("No trace collected. Cannot generate summary.")
            return None

        idata = self.trace

        mean_path = idata.posterior.x_path.mean(dim=("chain", "draw")).values
        std_path = idata.posterior.x_path.std(dim=("chain", "draw")).values

        finite_action_values = self.actions[np.isfinite(self.actions)] if self.actions is not None else np.array([])


        if len(finite_action_values) == 0:
            logger.warning("No finite action values found in the posterior samples.")
            best_action = np.nan
            action_mean = np.nan
            action_std = np.nan
        else:
            best_action = float(np.min(finite_action_values))
            action_mean = float(np.mean(finite_action_values))
            action_std = float(np.std(finite_action_values))

        summary_df = az.summary(
            idata,
            var_names=[
                "base_cost",
                "base_benefit",
                "scale_benefit",
                "eta",
                "ell",
                "mean_d",
            ],
        )

        def create_parameter_summary(param_name: str) -> ParameterSummary:
            """Helper to create ParameterSummary object from ArviZ summary."""
            if param_name not in summary_df.index:
                logger.warning(f"Parameter '{param_name}' not found in summary.")
                # Return a default ParameterSummary with NaNs
                return ParameterSummary(
                    mean=np.nan,
                    sd=np.nan,
                    hdi_3=(np.nan, np.nan),
                    hdi_97=(np.nan, np.nan),
                    mcse_mean=np.nan,
                    mcse_sd=np.nan,
                    ess_bulk=np.nan,
                    ess_tail=np.nan,
                    r_hat=np.nan,
                )

            param_summary = summary_df.loc[param_name]
            return ParameterSummary(
                mean=float(param_summary["mean"]),
                sd=float(param_summary["sd"]),
                hdi_3=( # Note: ArviZ 'hdi_3%' is the lower bound of the 94% HDI
                    float(param_summary["hdi_3%"]),
                    float(param_summary["hdi_97%"]),
                ),
                hdi_97=( # This field should ideally represent the 97th percentile of the HDI (upper bound)
                    float(param_summary["hdi_3%"]), # Storing the full interval here for now
                    float(param_summary["hdi_97%"]),
                ),
                mcse_mean=float(param_summary["mcse_mean"]),
                mcse_sd=float(param_summary["mcse_sd"]),
                ess_bulk=float(param_summary["ess_bulk"]),
                ess_tail=float(param_summary["ess_tail"]),
                r_hat=float(param_summary["r_hat"]),
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
            num_samples=len(self.actions) if self.actions is not None else 0,
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
            std_path=std_path,
        )
        return result

    def _calculate_historical_forecasted_metrics(
        self,
    ) -> tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        Helper to calculate historical and forecasted cost/benefit using posterior mean parameters.
        Historical d(t) is approximated by the posterior mean of the GP's constant mean function ('mean_d').
        Forecasted d(t) uses the posterior mean of the time-varying d(t) from the GP.
        """
        if self.trace is None:
            logger.warning(
                "MCMC trace not available. Cannot calculate metrics."
            )
            return None, None, None, None

        # Posterior mean of global parameters
        base_cost_mean = self.trace.posterior["base_cost"].values.mean()
        base_benefit_mean = self.trace.posterior["base_benefit"].values.mean()
        scale_benefit_mean = self.trace.posterior["scale_benefit"].values.mean()

        # Posterior mean of d(t) for the forecast period (length T)
        d_t_forecast_mean = self.trace.posterior["d_t"].mean(dim=("chain", "draw")).values
        
        # Posterior mean of the GP's constant mean function (proxy for historical d(t))
        d_t_historical_proxy = self.trace.posterior["mean_d"].values.mean()

        # Mean forecasted path (length T)
        if self.mcmc_paths is None:
             logger.warning("MCMC paths not available for forecast calculation.")
             return None, None, None, None
        mean_forecast_path = np.mean(self.mcmc_paths, axis=0)


        historical_cost: Optional[np.ndarray] = None
        historical_benefit: Optional[np.ndarray] = None
        
        if self.historical_input is not None:
            try:
                historical_cost = base_cost_mean * (self.historical_input ** d_t_historical_proxy)
                historical_benefit = base_benefit_mean * (self.historical_input ** scale_benefit_mean)
            except Exception as e:
                logger.error(f"Error calculating historical metrics: {e}")
                historical_cost = np.full_like(self.historical_input, np.nan) if self.historical_input is not None else None
                historical_benefit = np.full_like(self.historical_input, np.nan) if self.historical_input is not None else None
        else:
            logger.info("No historical input data provided, skipping historical metrics calculation.")


        forecast_cost: Optional[np.ndarray] = None
        forecast_benefit: Optional[np.ndarray] = None
        try:
            forecast_cost = base_cost_mean * (mean_forecast_path ** d_t_forecast_mean)
            forecast_benefit = base_benefit_mean * (mean_forecast_path ** scale_benefit_mean)
        except Exception as e:
            logger.error(f"Error calculating forecast metrics: {e}")
            forecast_cost = np.full_like(mean_forecast_path, np.nan)
            forecast_benefit = np.full_like(mean_forecast_path, np.nan)

        return historical_cost, historical_benefit, forecast_cost, forecast_benefit

    def plot_forecast(self, output_file: Optional[str] = None) -> None:
        """
        Creates three subplots showing historical data and forecasts for input, cost, and benefit.

        Args:
            output_file (Optional[str]): If provided, saves the plot to this file.
                                         Otherwise, shows the plot.
        """
        if self.mcmc_paths is None: # Depends on run_mcmc()
            logger.warning(
                "MCMC paths not available. Run run_mcmc() first. Cannot plot forecast."
            )
            return
        if self.T is None or self.T <= 0: # Should be caught by __init__
            logger.warning("No forecast period defined (T <= 0). Cannot plot forecast.")
            return
        if self.trace is None: # Depends on run_mcmc()
            logger.warning(
                "No MCMC trace available. Cannot calculate historical/forecasted cost/benefit."
            )
            return


        fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
        ax1, ax2, ax3 = axes

        # Calculate time axis for forecast
        forecast_time_axis = np.arange(1, self.T + 1)
        historical_time_axis = None

        if self.historical_t is not None:
            historical_time_axis = self.historical_t
            last_historical_time = self.historical_t[-1]
            # Assuming historical_t are indices or regularly spaced
            time_step = 1
            if len(self.historical_t) > 1:
                # Attempt to infer time step, default to 1 if irregular
                diffs = np.diff(self.historical_t)
                if len(diffs) > 0 and np.allclose(diffs, diffs[0]):
                    time_step = diffs[0]
            
            forecast_time_axis = last_historical_time + np.arange(1, self.T + 1) * time_step
            
            # Plot historical input data
            ax1.plot(
                historical_time_axis,
                self.historical_input,
                label="Historical Input",
                color="blue",
                marker="o",
                linestyle="-",
            )
            ax1.axvline(
                x=last_historical_time,
                color="gray",
                linestyle=":",
                linewidth=2,
                label="Forecast Horizon Start",
            )
        else: # No historical data, plot forecast starting from t=1
            forecast_time_axis = np.arange(1, self.T + 1)


        # Get mean forecast path
        mean_forecast_path = np.mean(self.mcmc_paths, axis=0)
        ax1.plot(
            forecast_time_axis,
            mean_forecast_path,
            label="Mean Forecasted Input",
            color="red",
            marker="x",
            linestyle="--",
        )
        ax1.set_ylabel("Allocation (x(t))")
        ax1.set_title("Historical Input and Mean Forecasted Allocation")
        ax1.legend()
        ax1.grid(True)

        # Calculate and plot cost/benefit
        historical_cost, historical_benefit, forecast_cost, forecast_benefit = (
            self._calculate_historical_forecasted_metrics()
        )

        # Plot historical cost/benefit if available
        if historical_cost is not None and historical_time_axis is not None:
            ax2.plot(
                historical_time_axis,
                historical_cost,
                label="Historical Cost (Estimated)",
                color="green",
                marker="o",
                linestyle="-",
            )
        if historical_benefit is not None and historical_time_axis is not None:
            ax3.plot(
                historical_time_axis,
                historical_benefit,
                label="Historical Benefit (Estimated)",
                color="orange",
                marker="o",
                linestyle="-",
            )
        
        # Plot forecasted cost/benefit if available
        if forecast_cost is not None:
            ax2.plot(
                forecast_time_axis,
                forecast_cost,
                label="Forecasted Cost (Mean)",
                color="darkgreen",
                marker="x",
                linestyle="--",
            )
        if forecast_benefit is not None:
            ax3.plot(
                forecast_time_axis,
                forecast_benefit,
                label="Forecasted Benefit (Mean)",
                color="darkorange",
                marker="x",
                linestyle="--",
            )

        # Vertical line for forecast start on cost/benefit plots if historical data exists
        if self.historical_t is not None:
            last_historical_time = self.historical_t[-1]
            for ax_cb in [ax2, ax3]:
                ax_cb.axvline(
                    x=last_historical_time,
                    color="gray",
                    linestyle=":",
                    linewidth=2,
                    # label="Forecast Horizon Start" # Already labeled in ax1
                )
        
        ax2.set_ylabel("Cost Value")
        ax2.set_title("Historical and Forecasted Cost")
        ax2.legend()
        ax2.grid(True)

        ax3.set_xlabel("Time (t)") # X-axis label only on the bottom plot due to sharex
        ax3.set_ylabel("Benefit Value")
        ax3.set_title("Historical and Forecasted Benefit")
        ax3.legend()
        ax3.grid(True)

        # Add overall title and adjust layout
        fig.suptitle(
            "Historical Data and Forecasts for Input, Cost, and Benefit Metrics", y=0.98
        ) # Adjusted y for suptitle
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for suptitle


        if output_file:
            plt.savefig(output_file)
            logger.info(f"Forecast plot saved to {output_file}")
        else:
            plt.show()
            logger.info("Forecast plot displayed")

    @staticmethod
    def from_data(
        input: np.ndarray, # Changed from np.array to np.ndarray for consistency
        cost: np.ndarray,
        benefit: np.ndarray,
        total_resource: float,
        hbar: float,
        num_steps: int,
        burn_in: int,
        forecast_steps: int,
        t: Optional[np.ndarray] = None,
    ): # type: ignore
        """
        Creates a PathIntegralOptimizer instance by first estimating parameters
        from historical data and using the estimates to define priors.
        """
        if t is None:
            _t_hist = np.arange(1, len(input) + 1)
        else:
            _t_hist = t

        data: Dataset = Dataset(t=_t_hist, input=input, cost=cost, benefit=benefit)

        logger.info("Estimating parameters from input data to define priors...")

        # TODO: Consider if ParameterEstimator should return more detailed d(t) info
        # for more accurate historical plotting if desired.
        parameters: ParameterEstimationResult = ParameterEstimator(
            data
        ).get_parameters() # Default MCMC parameters used here

        base_cost_prior_dict = {
            "dist": "TruncatedNormal",
            "mu": parameters.base_cost["mu"],
            "sigma": parameters.base_cost["sigma"],
            "lower": 0.0,
        }
        base_benefit_prior_dict = {
            "dist": "TruncatedNormal",
            "mu": parameters.base_benefit["mu"],
            "sigma": parameters.base_benefit["sigma"],
            "lower": 0.0,
        }
        scale_benefit_prior_dict = {
            "dist": "Beta",
            "mu": parameters.scale_benefit["mu"],
            "sigma": parameters.scale_benefit["sigma"],
        }
        gp_eta_prior_dict = {
            "dist": "HalfNormal",
            "sigma": parameters.gp_eta_prior["sigma"],
        }
        gp_ell_prior_dict = {
            "dist": "Gamma",
            "mu": parameters.gp_ell_prior["mu"],
            "sigma": parameters.gp_ell_prior["sigma"],
        }
        gp_mean_prior_dict = {
            "dist": "Normal",
            "mu": parameters.gp_mean_prior["mu"],
            "sigma": parameters.gp_mean_prior["sigma"],
        }

        return PathIntegralOptimizer(
            base_cost_prior=base_cost_prior_dict,
            base_benefit_prior=base_benefit_prior_dict,
            scale_benefit_prior=scale_benefit_prior_dict,
            gp_eta_prior=gp_eta_prior_dict,
            gp_ell_prior=gp_ell_prior_dict,
            gp_mean_prior=gp_mean_prior_dict,
            total_resource=total_resource,
            T=forecast_steps,  # T is the forecast period length
            hbar=hbar,
            num_steps=num_steps,
            burn_in=burn_in,
            historical_t=_t_hist,
            historical_input=input,
        )
