"""Part 4. The summary table.

Run on lambda with
    ./py-sbatch.sh part4_summary.py --id <your ID>

Evaluate every agent of 4.1 on the same 200 episodes, with seeds 0 to 199,
and write the table to results/summary.csv.
You may load trained agents that parts 2 and 3 saved to results/.
"""

import argparse
import os

from mission import make_mission, evaluate  # noqa: F401

N_EPISODES = 200
FIRST_SEED = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)

    env, info = make_mission(args.id)
    # TODO 4.1 build the agents, evaluate them, and write results/summary.csv


if __name__ == "__main__":
    main()
