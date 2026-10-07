"""Part 2. Planning with a known model.

Run on lambda with
    ./py-sbatch.sh part2_planning.py --id <your ID>

This script must reproduce every number and figure of Part 2 in your report.
Save numbers to results/ and figures to figures/.
You may add functions and files. Keep the command above working.

A state is the hashable tuple (cell, energy, collected), the same as
mission.obs_to_state(obs) and env.get_state() return.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import mission  # noqa: E402
from mission import ACTIONS, make_mission, evaluate, run_episode, obs_to_state  # noqa: E402,F401

DOMAIN_FILE = os.path.join("pddl", "domain.pddl")


# ----------------------------------------------------------------------------- 2.1
def write_problem_pddl(info: dict, path: str, state=None) -> None:
    """2.1(b) Write problem.pddl for the deterministic mission.

    state is None for the start of the mission. The replanning agent of 2.4 can
    pass its current state here.
    """
    raise NotImplementedError


def solve_with_fast_downward(domain_path: str, problem_path: str):
    """2.1(b) Solve optimally with Fast Downward through the Unified Planning Framework.

    Return the plan as a list of action names and the solving time in seconds.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------- 2.2
def transitions(state, action: str, info: dict):
    """2.2 Return every possible outcome as a list of (probability, next_state, reward, done).

    Use next_state = None when done is True.
    """
    raise NotImplementedError


def check_transitions(env, info: dict, n_pairs: int = 50, n_runs: int = 2000, seed: int = 0) -> float:
    """2.2 Compare transitions with env and return the largest difference in frequency."""
    raise NotImplementedError


# ----------------------------------------------------------------------------- 2.3
def value_iteration(info: dict):
    """2.3 Return (V, pi), two dictionaries keyed by state."""
    raise NotImplementedError


# ----------------------------------------------------------------------------- 2.4
class ReplanAgent:
    """2.4 Follows an optimal deterministic plan and plans again after a deviation."""

    def __init__(self, info: dict):
        self.info = info

    def reset(self):
        """Called by mission.evaluate at the start of every episode."""
        raise NotImplementedError

    def __call__(self, obs: dict, step_info: dict) -> str:
        raise NotImplementedError


# ----------------------------------------------------------------------------- 2.5
class UCTAgent:
    """2.5 Monte Carlo tree search with UCT, planning from the current state at every step."""

    def __init__(self, info: dict, n_simulations: int, c: float, rollout: str = "random", seed: int = 0):
        self.info = info
        self.n_simulations = n_simulations
        self.c = c
        self.rollout = rollout
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def __call__(self, obs: dict, step_info: dict) -> str:
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    env, info = make_mission(args.id)
    results = {}
    # TODO 2.1 to 2.5. Use mission.evaluate(policy, env, n_episodes, first_seed) for evaluation.

    with open("results/part2.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
