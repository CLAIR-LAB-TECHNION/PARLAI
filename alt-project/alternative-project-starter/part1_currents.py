"""Part 1. The ocean currents.

Run on lambda with
    ./py-sbatch.sh part1_currents.py --id <your ID>

This script must reproduce every number and figure of Part 1 in your report.
Save numbers to results/ and figures to figures/.
You may add functions and files. Keep the command above working.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mission import make_mission  # noqa: E402

REGIMES = ("Calm", "Strong")  # index 0 is Calm, index 1 is Strong


def transition_matrix(currents: dict) -> np.ndarray:
    """1.1 The 2 by 2 transition matrix of the regime chain, in the order (Calm, Strong)."""
    raise NotImplementedError


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """1.1 The stationary distribution of P."""
    raise NotImplementedError


def regime_distribution_after(P: np.ndarray, start: np.ndarray, n_moves: int) -> np.ndarray:
    """1.2 The regime distribution after n_moves moves, starting from the distribution start."""
    raise NotImplementedError


def simulate_currents(currents: dict, n_moves: int, seed: int) -> dict:
    """1.2 Simulate the regimes and the move outcomes, starting from Calm.

    Return the fraction of moves made in Strong and the fraction of moves that
    went in the intended direction.
    """
    raise NotImplementedError


def exact_filter(currents: dict, outcomes) -> np.ndarray:
    """1.4 Return b_t(Strong) for t = 1, ..., len(outcomes)."""
    raise NotImplementedError


def particle_filter(currents: dict, outcomes, n_particles: int, resample: bool, seed: int) -> np.ndarray:
    """1.5 Return the particle estimate of b_t(Strong) for t = 1, ..., len(outcomes)."""
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    env, info = make_mission(args.id)
    currents = info["currents"]
    log = info["current_log"]

    results = {}
    # TODO 1.1 and 1.2
    # TODO 1.4 plot b_t(Strong) with the true regimes and save it to figures/
    # TODO 1.5 compare the particle filter with the exact filter

    with open("results/part1.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
