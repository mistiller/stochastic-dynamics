import sys
import multiprocessing

from loguru import logger

from path_integral_optimizer import (
    SyntheticDataset,
    PathIntegralOptimizer, 
    PathIntegralOptimizerResult
)

def main():
    hbar: float = 0.1
    num_steps: int = 1000  # Reduced for testing
    burn_in: int = 1000     # Reduced for testing
    optimization_horizon: int = 12 # Optimizer will run for this many time periods for the forecast
    total_resource: float = 100.0 # Example total resource

    # SyntheticDataset generates (t, input, cost, benefit)
    # These are considered historical data for the optimizer
    # Use the same total_resource for synthetic data generation
    # Define sensible defaults for scale_benefit and d_t_value for synthetic data
    synthetic_scale_benefit = 0.7 
    synthetic_d_t_value = 1.1 
    t_hist, input_hist, cost_hist, benefit_hist = SyntheticDataset(
        T=12,
        total_resource=total_resource,
        scale_benefit=synthetic_scale_benefit,
        d_t_value=synthetic_d_t_value
    ).generate().arrays()
        
    optimizer: PathIntegralOptimizer = PathIntegralOptimizer.from_data(
        input=input_hist,
        cost=cost_hist,
        benefit=benefit_hist,
        t=t_hist,
        total_resource=total_resource,
        hbar=hbar,
        num_steps=num_steps,
        burn_in=burn_in,
        forecast_steps=optimization_horizon # Pass the new variable name
    )

    optimizer.run_mcmc()
    optimizer.plot() # Original plot of MCMC paths
    optimizer.plot_top_paths()
    optimizer.plot_forecast() # New plot showing historical + forecast
    summary_result = optimizer.generate_summary()

    if summary_result:
        logger.info(f"\n{summary_result}") # Log the formatted summary string
    else:
        logger.warning("Summary generation failed or was skipped.")

# Set multiprocessing start method to 'spawn' to avoid JAX multithreading deadlock
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    logger.add("stochastic_dynamics.log", rotation="500 MB", level="INFO")
    logger.info("Starting the stochastic dynamics application")

    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e

    logger.info("Stochastic dynamics application finished")
