import numpy as np
from typing import Dict, Any
from pydantic import BaseModel, Field, root_validator
import pymc as pm
from loguru import logger

class PriorDefinition(BaseModel):
    """Base model for prior definitions."""
    dist: str = Field(..., description="Name of the PyMC distribution")
    # Allow arbitrary parameters for the distribution
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the distribution")

    @root_validator(pre=True)
    def extract_dist_and_params(cls, values):
        """Extract 'dist' and remaining keys as 'params'."""
        if 'dist' not in values:
            raise ValueError("Prior definition must contain a 'dist' key.")
        dist_name = values.pop('dist')
        return {'dist': dist_name, 'params': values}

    def create_pymc_distribution(self, name: str) -> pm.Distribution:
        """Creates a PyMC distribution based on this definition."""
        try:
            dist_class = getattr(pm, self.dist, None)
            if dist_class is None:
                raise ValueError(f"Unknown distribution name: {self.dist}")

            # Special handling for distributions that can be defined by mu/sigma
            if self.dist in ["Beta", "Gamma"]:
                if "mu" in self.params and "sigma" in self.params:
                    mu = self.params["mu"]
                    sigma = self.params["sigma"]
                    if mu is None or sigma is None:
                         raise ValueError(f"{self.dist} prior for {name} requires either 'alpha'/'beta' or 'mu'/'sigma'. Got {self.params}")

                    if self.dist == "Beta":
                        # Convert mean/std to alpha/beta for Beta
                        if not (0 < mu < 1):
                            logger.warning(f"Mean {mu:.4f} for Beta prior '{name}' is not in (0, 1). Clamping.")
                            mu = np.clip(mu, 1e-6, 1 - 1e-6)
                        variance = sigma**2
                        max_variance = mu * (1 - mu)
                        if variance >= max_variance:
                            logger.warning(f"Variance {variance:.4f} for Beta prior '{name}' is too large for mean {mu:.4f}. Clamping variance.")
                            variance = max_variance * 0.9 # Use 90% of max variance

                        sum_ab = mu * (1 - mu) / variance - 1
                        if sum_ab <= 0:
                            logger.warning(f"Calculated alpha+beta <= 0 for Beta prior '{name}'. Using default Beta(1,1).")
                            alpha = 1.0
                            beta = 1.0
                        else:
                            alpha = mu * sum_ab
                            beta = (1 - mu) * sum_ab
                        params = {'alpha': max(alpha, 1e-6), 'beta': max(beta, 1e-6)}
                        logger.info(f"Converted mu={mu:.3f}, sigma={sigma:.3f} to Beta(alpha={params['alpha']:.3f}, beta={params['beta']:.3f}) for '{name}'.")

                    elif self.dist == "Gamma":
                        # Convert mean/std to alpha/beta for Gamma
                        variance = sigma**2
                        if variance <= 0:
                            logger.warning(f"Variance {variance:.4f} for Gamma prior '{name}' is non-positive. Using default Gamma(1,1).")
                            alpha = 1.0
                            beta = 1.0
                        else:
                            beta = mu / variance
                            alpha = mu * beta
                        params = {'alpha': max(alpha, 1e-6), 'beta': max(beta, 1e-6)}
                        logger.info(f"Converted mu={mu:.3f}, sigma={sigma:.3f} to Gamma(alpha={params['alpha']:.3f}, beta={params['beta']:.3f}) for '{name}'.")
                else:
                    # Use provided alpha/beta or other params directly
                    params = self.params

            elif self.dist == "TruncatedNormal":
                 # Ensure lower is present for this use case
                 if 'lower' not in self.params:
                      raise ValueError(f"TruncatedNormal prior for {name} requires 'lower'. Got {self.params}")
                 params = self.params
                 # Ensure sigma is positive
                 if 'sigma' in params:
                      params['sigma'] = max(params['sigma'], 1e-6)

            elif self.dist in ["HalfNormal", "Normal"]:
                 params = self.params
                 # Ensure sigma is positive
                 if 'sigma' in params:
                      params['sigma'] = max(params['sigma'], 1e-6)

            else:
                # For other distributions, use parameters directly
                params = self.params

            return dist_class(name, **params)

        except Exception as e:
            logger.exception(f"Error creating PyMC distribution '{self.dist}' for '{name}' with params {self.params}: {e}")
            raise
