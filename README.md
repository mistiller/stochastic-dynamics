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

### ParameterEstimator Class

Key methods:
- `run_mcmc()`: Estimates parameters from historical data using MCMC
- Returns `ParameterEstimationResult` with:
  - Estimated parameters for cost/benefit functions
  - GP hyperparameters
  - Full MCMC trace data

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

### The Optimization Challenge
We consider stochastic optimal control problems where we aim to maximize:

```math
𝔼[∫₀ᴛ (B(x(t),θ) - C(x(t))) dt]
```

Subject to:
```math
∫₀ᴛ x(t) dt ≤ X_{total},  x(t) ≥ 0
```

Where:
- `B(x(t),θ)` is a benefit function with uncertain parameters θ
- `C(x(t))` is a convex cost function
- `X_total` is a total resource budget

### Traditional Approaches vs Path Integral Method
Classical methods (Pontryagin's maximum principle, dynamic programming) struggle with:
- High-dimensional parameter uncertainty
- Non-convex landscapes
- Temporal correlations in parameters

Our Bayesian path integral approach:
1. Encodes uncertainty through prior distributions
2. Formulates constraints via Lagrange multipliers in the action
3. Uses Gaussian processes to model temporal correlations
4. Samples optimal paths using MCMC

### Action Functional Formulation
The core physics-inspired formulation:

```math
S[x(t)] = 𝔼_θ[∫₀ᴛ L(x(t), ẋ(t), θ) dt] + λ(∫x(t)dt - X_{total})
```

Where the Lagrangian `L` incorporates:
- Bayesian priors through `θ ~ p(θ)`
- System dynamics through `ẋ(t)` terms
- GP temporal correlations via kernel matrices

### Path Integral & MCMC Connection
We compute the optimal path distribution:

```math
P[x(t)] ∝ exp(-S[x(t)]/ħ)
```

Sampled via MCMC where:
- Each chain represents a candidate path x(t)
- The action S[x(t)] acts as negative log-probability
- ħ controls exploration/exploitation tradeoff:
  - Small ħ: Focus on minimal-action paths (classical regime)
  - Larger ħ: Explore stochastic variations (quantum regime)

This enables efficient exploration of:
- Discontinuous solution spaces
- Multiple local minima
- Temporally correlated parameter uncertainties

## References

[1] Research Paper (included in repository) detailing theoretical foundations
[2] PyMC documentation for MCMC implementation details
