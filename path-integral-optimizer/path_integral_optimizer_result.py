import numpy as np
from dataclasses import dataclass, field

@dataclass
class PathIntegralOptimizerResult:
    """Holds the results of the Path Integral Optimization."""
    base_benefit: float
    scale_benefit: float
    base_cost: float
    total_resource: float
    T: int
    hbar: float
    num_samples: int
    num_finite_actions: int
    best_action: float
    action_mean: float
    action_std: float
    mean_path: np.ndarray
    std_path: np.ndarray

    def __str__(self) -> str:
        """Returns base_benefit formatted string representation of the results."""
        summary_lines = [
            "=== MCMC Summary ===",
            f"Number of samples: {self.num_samples} (Finite actions: {self.num_finite_actions})",
            f"Parameters: base_benefit={self.base_benefit}, scale_benefit={self.scale_benefit}, base_cost={self.base_cost}, total_resource={self.total_resource}, T={self.T}, hbar={self.hbar}",
            f"Best path action: {self.best_action:.4f}",
            f"Mean action: {self.action_mean:.4f} ± {self.action_std:.4f}",
            "Mean allocation per time step:",
            "Time : Mean ± Std Dev"
        ]
        for t in range(self.T):
            summary_lines.append(f"{t+1:2d}   : {self.mean_path[t]:.4f} ± {self.std_path[t]:.4f}")
        summary_lines.append("====================")
        return "\n".join(summary_lines)