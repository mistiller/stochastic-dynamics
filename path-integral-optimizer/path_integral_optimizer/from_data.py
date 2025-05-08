from typing import Optional
from .dataset import Dataset
from .parameter_estimator import ParameterEstimator
from .path_integral_optimizer import PathIntegralOptimizer

def from_data(
    input:np.array,
    cost:np.array, 
    benefit:np.array,
    total_resource:float,
    hbar:float,
    num_steps:int,
    burn_in:int,
    t:Optional[np.array]=None
    ):
    _t=t or np.arange(1, len(input))
    data:Dataset=Dataset(
        t=_t,
        input=input,
        cost=cost,
        benefit=benefit
    )
    parameters:ParameterEstimationResult=ParameterEstimator(data) \
        .get_parameters()

    return PathIntegralOptimizer(
        base_benefit=parameters.base_benefit,
        scale_benefit=parameters.scale_benefit,
        gp_eta_prior=parameters.gp_eta_prior,
        gp_ell_prior=parameters.gp_ell_prior,
        gp_mean_prior=parameters.gp_mean_prior,
        base_cost=parameters.base_cost,
        total_resource=total_resource,
        T=_t.max(),
        hbar=hbar,
        num_steps=num_steps,
        burn_in=burn_in
    )



    
    