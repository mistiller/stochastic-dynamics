# Path Integral Optimizer for Bayesian MCMC Sampling

A Python implementation of path integral optimization combining Bayesian inference with quantum-inspired path integral methods for stochastic optimization problems, including both continuous resource allocation and discrete combinatorial problems like the 0-1 knapsack problem.

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

### Bayesian Parameter Estimator

The `ParameterEstimator` class implements hierarchical Bayesian inference using PyMC:

Key features:
- Gaussian Process priors for time-varying parameters
- Hamiltonian Monte Carlo (HMC) with NUTS sampling
- Automatic convergence diagnostics (R-hat, ESS)
- Returns `ParameterEstimationResult` with:
  - Posterior distributions for all parameters
  - GP hyperparameters (η, ℓ) 
  - MCMC trace with 4 chains × 2000 draws (1000 tune)

### Path Integral Optimizer

The `PathIntegralOptimizer` class implements the quantum-inspired stochastic optimization:

Key initialization parameters:
```python
PathIntegralOptimizer(
    base_cost_prior: Dict[str, Any],   # Prior for base_cost ~ p(base_cost)
    base_benefit_prior: Dict[str, Any],# Prior for base_benefit ~ p(base_benefit) 
    scale_benefit_prior: Dict[str, Any], # Prior for benefit scaling exponent
    gp_eta_prior: Dict[str, Any],      # GP amplitude prior η ~ HalfNormal(σ=1)
    gp_ell_prior: Dict[str, Any],      # GP length scale prior ℓ ~ Gamma(α=2, β=1)
    gp_mean_prior: Dict[str, Any],     # GP mean function μ ~ Normal(μ=1, σ=0.5)
    total_resource: float,             # Total resource constraint Σx(t) ≤ S
    T: int,                            # Time horizon (12 for monthly planning)
    hbar: float = 0.5                  # Planck constant (0.1-1.0 for exploration)
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

### Stochastic Optimal Control Formulation
We maximize the expected utility under uncertainty:

```math
𝔼_{θ∼p(⋅)} \left[ \sum_{t=1}^T \left( \text{base\_benefit} \cdot x(t)^{\text{scale\_benefit}} - \text{base\_cost} \cdot x(t)^{d(t)} \right) \right]
```

Subject to:
```math
\sum_{t=1}^T x(t) \leq S, \quad x(t) \geq 0\ \forall t
```

Where:
- `d(t) ∼ GP(μ(t), k(η, ℓ))` follows a Gaussian Process
- `base_cost`, `base_benefit`, `scale_benefit` have Bayesian priors
- `S` is the total resource budget

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

### Path Integral Implementation
The quantum-stochastic duality is implemented through:

```math
P[x(t)] propto \exp\left(-\frac{1}{\hbar} \left[\sum_{t=1}^T \left(\frac{\text{base\_cost}\ x(t)^{d(t)}}{d(t)} - \frac{\text{base\_benefit}\ x(t)^{\text{scale\_benefit}}}{\text{scale\_benefit}}\right) + \frac{1}{2}\mathbf{x}^\top K^{-1}\mathbf{x}\right]\right)
```

Where:
- `K` is the GP covariance matrix with parameters (η, ℓ)
- The action functional combines economic utility and temporal correlation
- Implemented in PyMC as:

```python
with pm.Model():
    d_t = pm.gp.Latent(mean_func=gp_mean, cov_func=eta**2 * pm.gp.cov.ExpQuad(1, ell))
    action = pm.Potential("action", 
        -(benefit_terms - cost_terms - 0.5 * x.T @ gp_cov @ x) / hbar
    )
    trace = pm.sample(draws=2000, tune=1000, chains=4, target_accept=0.95)
```

This enables efficient exploration of:
- Discontinuous solution spaces
- Multiple local minima
- Temporally correlated parameter uncertainties

## Knapsack Problem Solver

The `KnapsackOptimizer` class in `knapsack_solver/knapsack_optimizer.py` implements a Bayesian approach to the 0-1 knapsack problem using Hamiltonian Monte Carlo with constraint embedding. The script includes the `KnapsackOptimizer` class with methods for building the model (`build_model`), solving the problem (`solve`), summarizing the results (`summary`), and plotting the results (`plot_results`).

### Key Features:
- **Quantum-inspired Knapsack Solver**: Uses path integral formalism to encode the 0-1 knapsack problem with Bayesian inference and Hamiltonian Monte Carlo.
- **Action Functional**: Combines the objective (total value) and constraint (capacity) into a single action potential with smooth penalty formulation.
- **Constraint Handling**: Implements quantum-inspired constraint embedding through a continuous relaxation of the discrete problem.
- **MCMC Sampling**: Uses Sequential Monte Carlo (SMC) sampling to explore the path space of possible solutions.
- **Solution Extraction**: Identifies optimal solutions through posterior analysis of inclusion variables.
- **Visualization**: Provides diagnostic plots for value and weight distributions.

### Example Usage:
```python
# Example usage
values = [505, 352, 458, 220, 354, 414, 498, 545, 473, 543]
weights = [23, 26, 20, 18, 32, 27, 29, 26, 30, 27]
capacity = 67

solver = KnapsackOptimizer(values, weights, capacity, hbar=0.5)
solution = solver.solve()
solver.summary()
solver.plot_results()
```

### Path Integral Formulation for Knapsack:
The quantum-inspired path integral approach to discrete optimization problems like knapsack:

1. **Encoding Discrete Choices**: Uses continuous relaxation of Bernoulli variables through quantum state superposition
2. **Action Functional**: Formulates the knapsack problem as a path integral over possible solutions:
   
```math
S[x] = \frac{1}{\hbar} \left( \sum_{i=1}^N v_i x_i + \hbar \cdot \text{constraint}(x) \right)
```

3. **Constraint Handling**: Implements smooth constraint violation penalty through:
```python
constraint = pm.math.switch(total_weight > capacity,
                          -(total_weight - capacity)**2,
                          0)
```

4. **Quantum Exploration**: Uses path integral formalism to explore solution space with quantum-inspired fluctuations:
```python
pm.Potential('action', 
            (total_value + hbar * constraint) / hbar)
```

5. **Solution Extraction**: Finds optimal solutions through posterior analysis of the quantum state space:
```python
posterior = az.extract(trace, 'inclusion')
best_idx = np.argmax(posterior.sum('sample').values)
best_solution = posterior.sel(chain=best_idx).values.astype(bool)
```

## References

[1] Research Paper (included in repository) detailing theoretical foundations
[2] PyMC documentation for MCMC implementation details
