import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ParameterSummary:
    """Holds summary statistics for an inferred parameter."""
    mean: float
    sd: float
    hdi_3: tuple[float, float] # 3% HDI
    hdi_97: tuple[float, float] # 97% HDI
    mcse_mean: float
    mcse_sd: float
    ess_bulk: float
    ess_tail: float
    r_hat: float

@dataclass
class PathIntegralOptimizerResult:
    """Holds the results of the Path Integral Optimization."""
    # Fixed/Input Parameters
    total_resource: float
    T: int
    hbar: float
    num_samples: int # Total samples attempted
    num_finite_actions: int # Samples with finite action

    # Summaries of Inferred Parameters (Posterior)
    base_cost_summary: ParameterSummary
    base_benefit_summary: ParameterSummary
    scale_benefit_summary: ParameterSummary
    gp_eta_summary: ParameterSummary
    gp_ell_summary: ParameterSummary
    gp_mean_summary: ParameterSummary

    # Path and Action Summaries
    best_action: float # Minimum finite action found
    action_mean: float # Mean of finite actions
    action_std: float  # Standard deviation of finite actions
    mean_path: np.ndarray # Mean allocation path
    std_path: np.ndarray  # Standard deviation of allocation path

    def __str__(self) -> str:
        """Returns a formatted string representation of the results."""
        summary_lines = [
            "=== Path Integral Optimization Summary ===",
            f"Total Samples: {self.num_samples}",
            f"Finite Actions: {self.num_finite_actions}",
            f"Fixed Parameters: total_resource={self.total_resource}, T={self.T}, hbar={self.hbar}",
            "",
            "--- Inferred Parameter Summaries (Posterior) ---",
            f"base_cost:      Mean={self.base_cost_summary.mean:.4f}, SD={self.base_cost_summary.sd:.4f}, R_hat={self.base_cost_summary.r_hat:.2f}",
            f"base_benefit:   Mean={self.base_benefit_summary.mean:.4f}, SD={self.base_benefit_summary.sd:.4f}, R_hat={self.base_benefit_summary.r_hat:.2f}",
            f"scale_benefit:  Mean={self.scale_benefit_summary.mean:.4f}, SD={self.scale_benefit_summary.sd:.4f}, R_hat={self.scale_benefit_summary.r_hat:.2f}",
            f"gp_eta:         Mean={self.gp_eta_summary.mean:.4f}, SD={self.gp_eta_summary.sd:.4f}, R_hat={self.gp_eta_summary.r_hat:.2f}",
            f"gp_ell:         Mean={self.gp_ell_summary.mean:.4f}, SD={self.gp_ell_summary.sd:.4f}, R_hat={self.gp_ell_summary.r_hat:.2f}",
            f"gp_mean:        Mean={self.gp_mean_summary.mean:.4f}, SD={self.gp_mean_summary.sd:.4f}, R_hat={self.gp_mean_summary.r_hat:.2f}",
            "",
            "--- Path and Action Summaries ---",
            f"Best path action: {self.best_action:.4f}",
            f"Mean action: {self.action_mean:.4f} ± {self.action_std:.4f}",
            "Mean allocation per time step:",
            "Time : Mean ± Std Dev"
        ]
        for t in range(self.T):
            summary_lines.append(f"{t+1:2d}   : {self.mean_path[t]:.4f} ± {self.std_path[t]:.4f}")
        summary_lines.append("===================================")
        return "\n".join(summary_lines)
