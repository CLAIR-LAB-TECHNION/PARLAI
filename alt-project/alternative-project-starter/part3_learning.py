"""Part 3. Learning from experience and from an expert.

Run on lambda with
    ./py-sbatch.sh part3_learning.py --id <your ID>

This script must reproduce every number and figure of Part 3 in your report.
Save numbers to results/ and figures to figures/.
You may add functions and files. Keep the command above working.

The agents of 3.1 and 3.2 may use only env.reset and env.step.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mission import ACTIONS, make_mission, evaluate, run_episode  # noqa: E402,F401

N_TRAIN_EPISODES = 30_000
EVAL_EVERY = 1_000
N_EVAL_EPISODES = 100
SEEDS = (0, 1, 2)


# ----------------------------------------------------------------------------- 3.1 and 3.2
def state_key(obs: dict):
    """3.1 Your state key. Must be hashable."""
    raise NotImplementedError


def q_learning(env, key_fn, settings: dict, seed: int):
    """3.1 Train tabular Q-learning. Return the Q table and the evaluation curve."""
    raise NotImplementedError


def mc_control(env, key_fn, settings: dict, seed: int):
    """3.1 Train on-policy first-visit Monte Carlo control. Return the Q table and the evaluation curve."""
    raise NotImplementedError


def greedy_policy(Q, key_fn):
    """Return a policy(obs, step_info) that acts greedily with respect to Q among available actions."""
    raise NotImplementedError


# ----------------------------------------------------------------------------- 3.3 and 3.4
def features(obs: dict) -> np.ndarray:
    """3.3 Your features for behavior cloning."""
    raise NotImplementedError


def collect_expert_data(env, expert, n_episodes: int, first_seed: int):
    """3.3 Run the expert and return (X, y) for supervised learning."""
    raise NotImplementedError


def train_classifier(X, y):
    """3.3 Train and return your classifier."""
    raise NotImplementedError


def dagger(env, expert, n_rounds: int = 5, episodes_per_round: int = 5):
    """3.4 Return the policy after each round and the number of expert labels used so far."""
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    env, info = make_mission(args.id)
    results = {}
    # TODO 3.1 to 3.4. The expert pi* comes from value_iteration in part2_planning.py.

    with open("results/part3.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
