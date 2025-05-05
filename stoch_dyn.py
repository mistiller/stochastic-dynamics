from loguru import logger

from path_integral_optimizer import PathIntegralOptimizer

def main() -> None:
    """Main function to execute the stochastic dynamics application."""
    logger.add("stoch_dyn.log", rotation="500 MB", level="INFO")
    logger.info("Starting the stochastic dynamics application")

    # Parameters
    a: float = 1.0  # Reduced for numerical stability
    b: float = 0.5
    c: float = 0.5  # Reduced for numerical stability
    S: float = 10    # Reduced for numerical stability
    T: int = 5       # Reduced for numerical stability
    hbar: float = 0.1
    num_steps: int = 1000  # Reduced for testing
    burn_in: int = 500     # Reduced for testing
    proposal_stddev: float = 0.5  # Standard deviation for proposal distribution

    try:
        optimizer: PathIntegralOptimizer = PathIntegralOptimizer(a, b, c, S, T, hbar, num_steps, burn_in, proposal_stddev)
        optimizer.run_mcmc()
        optimizer.plot_top_paths()
        optimizer.generate_summary()

    except Exception as e:
        logger.exception(f"An error occurred: {e}")

    logger.info("Stochastic dynamics application finished")

if __name__ == "__main__":
    main()
