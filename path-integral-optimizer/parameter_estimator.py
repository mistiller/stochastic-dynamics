import pymc as pm
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional
import arviz as az
from pytensor import tensor as pt
from .parameter_estimation_result import ParameterEstimationResult


class ParameterEstimator:
    """
    Estimates cost/benefit function parameters from historical time series data.
    
    Expected data format: Dict with keys:
    - 't' (time points), 'input', 'cost', 'benefit' (all numpy arrays)
    """
    
    def __init__(self, data: Dict[str, np.ndarray]):
        self.data = data
        self.T = len(data['t'])
        self.model = pm.Model()
        
        # Initialize parameter priors with reasonable defaults
        self.priors = {
            'base_cost': {'dist': 'HalfNormal', 'sigma': 1.0},
            'base_benefit': {'dist': 'Normal', 'mu': 0, 'sigma': 1.0},
            'scale_benefit': {'dist': 'HalfNormal', 'sigma': 1.0},
            'gp_eta': {'dist': 'HalfNormal', 'sigma': 1.0},
            'gp_ell': {'dist': 'Gamma', 'alpha': 2, 'beta': 0.5},
            'gp_mean': {'dist': 'Normal', 'mu': 0, 'sigma': 1.0}
        }

    def _build_model(self):
        """Construct the PyMC model for parameter estimation."""
        with self.model:
            # Priors for base parameters
            base_cost = pm.HalfNormal("base_cost", **self.priors['base_cost'])
            base_benefit = pm.Normal("base_benefit", **self.priors['base_benefit'])
            scale_benefit = pm.HalfNormal("scale_benefit", **self.priors['scale_benefit'])
            
            # GP priors for time-varying cost component
            eta = pm.HalfNormal("eta", **self.priors['gp_eta'])
            ell = pm.Gamma("ell", **self.priors['gp_ell'])
            mean_d = pm.Normal("mean_d", **self.priors['gp_mean'])
            
            # Construct GP covariance function
            cov = eta**2 * pm.gp.cov.ExpQuad(1, ls=ell)
            gp = pm.gp.Latent(mean_func=pm.gp.mean.Constant(mean_d))
            f_d = gp.prior("f_d", X=self.data['t'][:, None])
            d_t = pt.exp(f_d)  # Ensure positive values
            
            # Observation models
            cost_model = base_cost + d_t * self.data['input']
            pm.Normal("cost_obs", mu=cost_model, sigma=0.1, observed=self.data['cost'])
            
            benefit_model = base_benefit + scale_benefit * self.data['input']
            pm.Normal("benefit_obs", mu=benefit_model, sigma=0.1, observed=self.data['benefit'])

    def run_mcmc(self, draws=1000, tune=1000, chains=4) -> ParameterEstimationResult:
        """Run MCMC sampling to estimate parameters."""
        self._build_model()
        
        with self.model:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=0.95,
                return_inferencedata=True
            )
            
        # Extract posterior means for parameters
        post = trace.posterior
        return ParameterEstimationResult(
            base_cost={'mu': float(post.base_cost.mean()), 'sigma': float(post.base_cost.std())},
            base_benefit={'mu': float(post.base_benefit.mean()), 'sigma': float(post.base_benefit.std())},
            scale_benefit={'mu': float(post.scale_benefit.mean()), 'sigma': float(post.scale_benefit.std())},
            gp_eta_prior={'mu': float(post.eta.mean()), 'sigma': float(post.eta.std())},
            gp_ell_prior={'mu': float(post.ell.mean()), 'sigma': float(post.ell.std())},
            gp_mean_prior={'mu': float(post.mean_d.mean()), 'sigma': float(post.mean_d.std())},
            trace=trace
        )
