# Applying Path Integral Optimization to the 0-1 Knapsack Problem

This document outlines a plan for applying the path integral optimization framework, as described in "From Optimization to Path Integrals," to solve the classic 0-1 Knapsack Problem.

## 1. The Core Problem: 0-1 Knapsack

First, we restate the problem we aim to solve. The "0-1 Knapsack Problem" is a combinatorial optimization problem defined as:

> "Maximize $\sum_{i=1}^{n} v_i x_i$
> Subject to $\sum_{i=1}^{n} w_i x_i \leq W$ and $x_i \in \{0, 1\}$"

Here, for a set of $n$ items, we must decide whether to include each item ($x_i=1$) or not ($x_i=0$). Each item $i$ has a value $v_i$ and a weight $w_i$. The goal is to maximize the total value of included items without exceeding the total knapsack capacity $W$.

## 2. The Framework: Path Integral Optimization

Our approach is based on the path integral formalism, which recasts an optimization problem in the language of stochastic dynamics. The core idea is that:

> "Uncertainty is introduced via a partition function:
> $$Z = \int \mathcal{D}[x(t)] \, e^{-S[x(t)] / \hbar}$$
> Here, $S[x(t)]$ is the action functional and $\hbar$ modulates fluctuation scale."

In this framework, any possible solution (a "path") has a probability determined by its "action" $S$. The MCMC sampler seeks paths with high probability, which correspond to paths with low action. Our goal is to define an action functional $S[x]$ for the knapsack problem such that its minimization yields the optimal solution.

## 3. Mapping Knapsack to a Path Integral Formulation

To apply the framework, we must translate the components of the knapsack problem into the language of path integrals. This involves defining the "path," the "action," and handling the discrete nature of the choices.

### 3.1. From a Time-Series Path to a Solution Vector

In the original resource allocation problem, a "path" was a time series of resource allocations $x(t)$. For the knapsack problem, a "path" is simply a specific configuration of choices—a binary vector $x = (x_1, x_2, \ldots, x_n)$ that represents a potential solution. The integral over all paths $\int \mathcal{D}[x(t)]$ becomes a sum over all $2^n$ possible configurations of $x$.

### 3.2. Handling Discrete Variables with Continuous Relaxation

The path integral framework and its MCMC implementation (specifically, Hamiltonian Monte Carlo) are designed for continuous variables. However, the knapsack variables $x_i$ are discrete, taking values in $\{0, 1\}$. To bridge this gap, we employ a continuous relaxation.

Instead of a binary choice $x_i$, we introduce a continuous variable $p_i \in [0, 1]$ for each item, representing the probability of its inclusion. This is a key step, as it allows us to use gradient-based samplers like HMC. The discrete solution vector $x$ is replaced by a continuous probability vector $p = (p_1, p_2, \ldots, p_n)$.

In a PyMC model, these probabilities can be represented by a suitable distribution, for example, a Beta distribution, which is naturally constrained to the $[0, 1]$ interval.

```python
import pymc as pm

n_items = 10 # Example number of items
with pm.Model() as model:
    # Use a continuous variable p_i in [0, 1] for each item x_i
    inclusion_probs = pm.Beta('p', alpha=1.0, beta=1.0, shape=n_items)
```

After the MCMC sampler generates posterior distributions for these probabilities, we can recover a final discrete solution by, for example, rounding the posterior mean probabilities to 0 or 1.

### 3.3. Defining the Action Functional $S[x]$

The action functional $S[x]$ must encode the knapsack problem's objective and constraint. A low action should correspond to a high-value, valid solution. We construct the action from two components: an objective term and a constraint penalty term.

**Objective Term:** The goal is to maximize total value, $\sum v_i x_i$. Since the path integral framework seeks to *minimize* the action, the objective term should be the negative of the total value. Using our continuous relaxation:

$$ S_{\text{objective}}[p] = - \sum_{i=1}^{n} v_i p_i $$

**Constraint Term:** The solution must respect the capacity constraint $\sum w_i x_i \leq W$. As noted in the project's approach to the knapsack problem, "Constraints are handled via a smooth penalty in the action functional." We add a penalty to the action that increases as the constraint is violated. A quadratic penalty is a common choice as it creates a smooth, differentiable function:

$$ S_{\text{constraint}}[p] = \lambda \cdot \max\left(0, \sum_{i=1}^{n} w_i p_i - W\right)^2 $$

Here, $\lambda$ is a large constant that determines how severely we penalize infeasible solutions.

**Total Action:** The total action is the sum of these two parts:

$$ S[p] = S_{\text{objective}}[p] + S_{\text{constraint}}[p] = - \sum_{i=1}^{n} v_i p_i + \lambda \cdot \max\left(0, \sum_{i=1}^{n} w_i p_i - W\right)^2 $$

Minimizing this action simultaneously maximizes the value and minimizes the constraint violation.

## 4. The Role of the Fluctuation Parameter $\hbar$

The parameter $\hbar$ is crucial for controlling the behavior of the MCMC sampler. As described in the path integral theory:

> "In our resource optimization context, $\hbar$ acts as an 'uncertainty temperature' regulating exploration strength. Large $\hbar$ values permit greater path deviations to escape local optima, while $\hbar \to 0$ recovers deterministic gradient flow."

The probability of a given solution vector $p$ is given by $P(p) \propto e^{-S[p]/\hbar}$.

*   **Large $\hbar$**: A large $\hbar$ makes the exponent $-S[p]/\hbar$ smaller in magnitude. This flattens the probability landscape, making the sampler explore a wider range of solutions, including suboptimal ones. This is useful for escaping local optima in complex problem landscapes.
*   **Small $\hbar$**: A small $\hbar$ makes the exponent larger, creating a "peaky" landscape where high-probability states are sharply distinguished from low-probability states. This encourages the sampler to focus its efforts on the most promising solutions (exploitation).

The choice of $\hbar$ is therefore a key hyperparameter for tuning the exploration-exploitation trade-off of the optimization process.

## 5. A Concrete Implementation Plan in PyMC

We can combine these elements into a concrete PyMC model. The model will define the continuous variables, the action, and use `pm.Potential` to link the action to the model's log-probability, thereby defining the probability of each path.

The log-probability of a path is $-\frac{S[p]}{\hbar}$. We add this term to the model's log-probability using `pm.Potential`.

```python
import pymc as pm
import pytensor.tensor as pt
import numpy as np

def build_knapsack_path_integral_model(values, weights, capacity, hbar=1.0, penalty_factor=1e3):
    """
    Builds a PyMC model for the knapsack problem using the path integral formulation.
    """
    n_items = len(values)
    
    # Convert inputs to numpy arrays for tensor operations
    values = np.array(values, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)

    with pm.Model() as knapsack_model:
        # 1. Continuous relaxation of discrete choices x_i
        inclusion_probs = pm.Beta('inclusion_probs', alpha=1.0, beta=1.0, shape=n_items)

        # 2. Define the action functional S[p]
        # Objective term: negative of total value
        expected_value = pt.sum(values * inclusion_probs)
        
        # Constraint term: quadratic penalty for exceeding capacity
        expected_weight = pt.sum(weights * inclusion_probs)
        weight_overage = pt.maximum(0., expected_weight - capacity)
        penalty = penalty_factor * (weight_overage ** 2)
        
        # Total action
        action = -expected_value + penalty

        # 3. Define the path probability via pm.Potential
        # The log-probability is -action / hbar
        pm.Potential("path_probability", -action / hbar)
        
    return knapsack_model

# Example Usage:
# values = [505, 352, 458, 220, 354]
# weights = [23, 26, 20, 18, 32]
# capacity = 67
#
# model = build_knapsack_path_integral_model(values, weights, capacity, hbar=0.5)
# with model:
#     trace = pm.sample()
```

This model provides a complete plan for implementation. By running an MCMC sampler like NUTS on this model, we can obtain posterior distributions for the `inclusion_probs`. Analyzing these posteriors will reveal the optimal or near-optimal item selections, thus solving the knapsack problem through the lens of path integral optimization.
