from dataclasses import dataclass
from typing import Dict, Any, Optional
import arviz as az


@dataclass
class ParameterEstimationResult:
    """Holds the results of parameter estimation."""

    base_cost: Dict[str, Any]
    base_benefit: Dict[str, Any]
    scale_benefit: Dict[str, Any]
    gp_eta_prior: Dict[str, Any]
    gp_ell_prior: Dict[str, Any]
    gp_mean_prior: Dict[str, Any]
    trace: Optional[az.InferenceData] = None

    def __str__(self) -> str:
        return (
            f"Parameter Estimates:\n"
            f"Base Cost: {self.base_cost}\n"
            f"Base Benefit: {self.base_benefit}\n"
            f"Scale Benefit: {self.scale_benefit}\n"
            f"GP η: {self.gp_eta_prior}\n"
            f"GP ℓ: {self.gp_ell_prior}\n"
            f"GP Mean: {self.gp_mean_prior}"
        )
