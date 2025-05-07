# Path Integral Optimizer for Bayesian MCMC Sampling

A Python implementation of path integral optimization combining Bayesian inference with quantum-inspired path integral methods for stochastic optimization problems.

## Installation & Requirements

```bash
git clone https://github.com/yourusername/path-integral-optimizer.git
cd path-integral-optimizer
uv pip install -r requirements.txt
```

For development with PyMC:
```bash
uv pip install -r requirements.txt --all-extras
```

## Core Components

### PathIntegralOptimizer Class

Key initialization parameters:
```python
PathIntegralOptimizer(
    base_benefit: Dict[str, Any],      # Prior for baseline benefit distribution
    scale_benefit: Dict[str, Any],     # Prior for benefit scaling factor
    gp_eta_prior: Dict[str, Any],      # GP magnitude prior (η)
    gp_ell_prior: Dict[str, Any],      # GP length scale prior (ℓ)
    gp_mean_prior: Dict[str, Any],     # GP mean function prior
    base_cost: float,                  # Fixed cost parameter
    total_resource: float,             # Total resources available (∫x(t)dt ≤ X)
    T: int,                            # Time horizon
    hbar: float = 1.0                  # Effective Planck constant
)
```

Key methods:
- `run_mcmc()`: Executes MCMC sampling using PyMC
- `plot_top_paths(n=10)`: Visualizes top n optimal paths
- `generate_summary()`: Returns PathIntegralOptimizerResult with:
  - Optimal path distributions
  - Parameter posterior statistics
  - Convergence diagnostics
  - Action value metrics

## Theoretical Framework

Combines Feynman path integrals with Bayesian inference:

```math
Z = ∫ 𝒟x(t) exp⁡(-S[x(t)]/ħ)
```

Where the action functional S[x(t)] incorporates:
- Parameter uncertainty through Bayesian priors
- Resource constraints via Lagrange multipliers
- Temporal correlations via Gaussian processes

The quantum-classical transition is controlled by ħ:
- ħ → 0: Classical deterministic paths
- ħ > 0: Quantum stochastic paths

## References

[1] Research Paper (included in repository) detailing theoretical foundations
[2] PyMC documentation for MCMC implementation details
