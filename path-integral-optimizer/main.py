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

    data:tuple[np.array]=SyntheticDataset() \
        .generate() \
        .arrays()
        
    optimizer: PathIntegralOptimizer=PathIntegralOptimizer.from_data(
        *data,
        hbar,
        num_steps,
        burn_in
    )

    optimizer.run_mcmc()
    optimizer.plot()
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
