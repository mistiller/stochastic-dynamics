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
    forecast_steps: int = 5 # Number of steps to forecast beyond historical data
    total_resource: float = 100.0 # Example total resource

    # SyntheticDataset generates (t, input, cost, benefit)
    # These are considered historical data for the optimizer
    t_hist, input_hist, cost_hist, benefit_hist = SyntheticDataset(T=12).generate().arrays()
        
    optimizer: PathIntegralOptimizer = PathIntegralOptimizer.from_data(
        input=input_hist,
        cost=cost_hist,
        benefit=benefit_hist,
        t=t_hist,
        total_resource=total_resource,
        hbar=hbar,
        num_steps=num_steps,
        burn_in=burn_in,
        forecast_steps=forecast_steps
    )

    optimizer.run_mcmc()
    optimizer.plot() # Original plot of MCMC paths
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
