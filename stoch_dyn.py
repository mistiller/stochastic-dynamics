import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import copy
from loguru import logger
from typing import List

class PathIntegralOptimizer:
    """A class for performing path integral optimization using Markov Chain Monte Carlo (MCMC)."""
    def __init__(self, a: float, b: float, c: float, S: float, T: int, hbar: float, num_steps: int, burn_in: int, proposal_stddev: float, seed: int = 42) -> None:
        """Initializes the PathIntegralOptimizer.

        Args:
            a (float): Parameter for benefit function.
            b (float): Parameter for benefit function.
            c (float): Parameter for cost function.
            S (float): Total resource.
            T (int): Number of time steps.
            hbar (float): Noise parameter.
            num_steps (int): Number of MCMC steps.
            burn_in (int): Number of burn-in steps.
            proposal_stddev (float): Standard deviation for the proposal distribution.
            seed (int): Random seed for reproducibility.
        """
        self.a: float = a
        self.b: float = b
        self.c: float = c
        self.S: float = S
        self.T: int = T
        self.hbar: float = hbar
        self.num_steps: int = num_steps
        self.burn_in: int = burn_in
        self.proposal_stddev: float = proposal_stddev
        self.seed: int = seed
        np.random.seed(self.seed)
        self.mcmc_paths: List[np.ndarray] = []
        self.actions: List[float] = []

    def d(self, t: int) -> float:
        """Defines the time-dependent function d(t).

        Args:
            t (int): The time step.

        Returns:
            float: The value of d(t).
        """
        return 2 + 0.1 * t

    def compute_action(self, x_path: np.ndarray) -> float:
        """Computes the action for a given path.

        Args:
            x_path (np.ndarray): The allocation path.

        Returns:
            float: The action value.
        """
        total: float = 0
        for t, x in enumerate(x_path, 1):
            x_safe: float = x + 1e-9
            benefit: float = self.a * x_safe**self.b
            cost: float = self.c * x_safe**self.d(t)
            total += benefit - cost
        return -total

    def run_mcmc(self) -> None:
        """Runs the Markov Chain Monte Carlo (MCMC) simulation.

        This method generates samples of allocation paths according to the Metropolis algorithm.
        """
        logger.info("Starting MCMC sampling...")
        # Initial path (e.g., uniform allocation)
        current_path: np.ndarray = np.full(self.T, self.S / self.T)
        current_action: float = self.compute_action(current_path)

        for step in range(self.num_steps):
            # Propose a new path by transferring resources between two random time points
            proposed_path: np.ndarray = copy.deepcopy(current_path)

            # Choose two distinct time points
            t1, t2 = np.random.choice(self.T, 2, replace=False)

            # Propose amount to transfer (can be positive or negative)
            # Ensure delta is small enough not to violate non-negativity constraints easily
            max_transfer_from_t1: float = proposed_path[t1]
            transfer_amount: float = norm.rvs(loc=0, scale=self.proposal_stddev)

            # Ensure non-negativity after transfer
            # Cap the transfer amount if it makes x[t1] negative
            transfer_amount: float = max(transfer_amount, -max_transfer_from_t1 + 1e-9) # Add epsilon for safety

            # Perform transfer
            proposed_path[t1] -= transfer_amount
            proposed_path[t2] += transfer_amount

            # Check constraints: non-negativity and total sum S
            # Non-negativity at t2
            if proposed_path[t2] < 0:
                # Reject proposal if t2 becomes negative
                # (Could also reflect the transfer amount, but rejection is simpler)
                pass # Keep current_path
            else:
                # Calculate action for the proposed path
                proposed_action: float = self.compute_action(proposed_path)

                # Acceptance probability (Metropolis criterion)
                acceptance_prob: float = min(1, np.exp(-(proposed_action - current_action) / self.hbar))

                # Accept or reject the proposal
                if np.random.rand() < acceptance_prob:
                    current_path = proposed_path
                    current_action = proposed_action

            # Store path after burn-in period
            if step >= self.burn_in:
                self.mcmc_paths.append(copy.deepcopy(current_path))
                self.actions.append(current_action)

            if (step + 1) % 5000 == 0:
                logger.info(f"MCMC Step {step+1}/{self.num_steps}")

        logger.info(f"MCMC sampling finished. Collected {len(self.mcmc_paths)} samples after burn-in.")

        # Ensure we have samples
        if not self.mcmc_paths:
            logger.error("Error: No paths collected after burn-in. Increase num_steps or decrease burn_in.")
            exit()

    def plot_top_paths(self, num_paths_to_plot: int = 10) -> None:
        """Plots the top N most probable paths.

        Args:
            num_paths_to_plot (int): The number of paths to plot.
        """
        # Select top paths based on lowest action (highest probability)
        # Sort indices based on action (ascending, lower action is better)
        sorted_indices: np.ndarray = np.argsort(self.actions)
        top_indices: np.ndarray = sorted_indices[:num_paths_to_plot]

        # Plot top most probable paths from MCMC samples
        plt.figure(figsize=(10, 6))
        for i, idx in enumerate(top_indices):
            plt.plot(range(1, self.T + 1), self.mcmc_paths[idx], label=f"Path {i+1} (Action: {self.actions[idx]:.2f})", alpha=0.7)

        # Optional: Plot average path
        # average_path = np.mean(mcmc_paths, axis=0)
        # plt.plot(range(1, T + 1), average_path, label="Average Path", color='black', linewidth=2, linestyle='--')

        plt.xlabel("Time (t)")
        plt.ylabel("Allocation (x(t))")
        plt.title(f"Top {num_paths_to_plot} MCMC Sampled Allocation Paths (Lowest Action)")
        plt.legend()
        plt.grid(True)
        plt.show()


def main() -> None:
    """Main function to execute the stochastic dynamics application."""
    logger.add("stoch_dyn.log", rotation="500 MB", level="INFO")
    logger.info("Starting the stochastic dynamics application")

    # Parameters
    a: float = 10
    b: float = 0.5
    c: float = 2
    S: float = 50
    T: int = 12
    hbar: float = 0.1  # Noise parameter (smaller = less stochasticity)
    num_steps: int = 50000  # Total MCMC steps
    burn_in: int = 10000    # Steps to discard for equilibration
    proposal_stddev: float = 0.5 # Standard deviation for the resource transfer proposal

    try:
        optimizer: PathIntegralOptimizer = PathIntegralOptimizer(a, b, c, S, T, hbar, num_steps, burn_in, proposal_stddev)
        optimizer.run_mcmc()
        optimizer.plot_top_paths()

    except Exception as e:
        logger.error(f"An error occurred: {e}")

    logger.info("Stochastic dynamics application finished")

if __name__ == "__main__":
    main()