import sys
import multiprocessing

from loguru import logger

from path_integral_optimizer import PathIntegralOptimizer

# Set multiprocessing start method to 'spawn' to avoid JAX multithreading deadlock
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    logger.add("stoch_dyn.log", rotation="500 MB", level="INFO")
    logger.info("Starting the stochastic dynamics application")

    # Parameters
    base_benefit: dict = {"dist": "Normal", "mu": 1.0, "sigma": 0.1}
    scale_benefit: dict = {"dist": "Beta", "alpha": 2.0, "beta": 2.0}
    
    # GP priors for d(t)
    gp_eta_prior: dict = {"dist": "HalfNormal", "sigma": 1}
    gp_ell_prior: dict = {"dist": "Gamma", "alpha": 5, "beta": 1}
    gp_mean_prior: dict = {"dist": "Normal", "mu": 2, "sigma": 0.5}

    base_cost: float = 0.5  # Reduced for numerical stability
    total_resource: float = 10    # Reduced for numerical stability
    T: int = 5       # Reduced for numerical stability
    hbar: float = 0.1
    num_steps: int = 1000  # Reduced for testing
    burn_in: int = 500     # Reduced for testing

    try:
        optimizer: PathIntegralOptimizer = PathIntegralOptimizer(
            base_benefit, scale_benefit,
            gp_eta_prior, gp_ell_prior, gp_mean_prior,
            base_cost, total_resource, T, hbar, num_steps, burn_in
        )
        optimizer.run_mcmc()
        optimizer.plot_top_paths()
        summary_result = optimizer.generate_summary()

        if summary_result:
            logger.info(f"\n{summary_result}") # Log the formatted summary string
        else:
            logger.warning("Summary generation failed or was skipped.")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")

    logger.info("Stochastic dynamics application finished")
