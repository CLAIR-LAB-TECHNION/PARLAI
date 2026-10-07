"""Check that your environment is ready for the project.

Run it on the lambda server through slurm, for example
    ./py-sbatch.sh check_setup.py --id 123456789

It checks the Python packages, builds your mission, runs a few random
episodes, solves a tiny planning problem with Fast Downward, and saves the
figure of your mission to figures/mission.png.
"""

import argparse
import os
import platform
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="your 9-digit ID number")
    args = parser.parse_args()

    print("Python", sys.version.split()[0], "on", platform.node())

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import gymnasium
    import sklearn
    import unified_planning
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = "not installed (needed only if you use a neural network)"
    print("numpy", np.__version__, "| gymnasium", gymnasium.__version__, "| scikit-learn", sklearn.__version__,
          "| unified-planning", unified_planning.__version__, "| torch", torch_version)

    import mission
    env, info = mission.make_mission(args.id)
    print()
    print(mission.describe_mission(info))

    def random_policy(obs, step_info, rng=np.random.default_rng(0)):
        allowed = np.flatnonzero(step_info["action_mask"])
        return int(rng.choice(allowed))

    t = time.time()
    result = mission.evaluate(random_policy, env, n_episodes=200, first_seed=0)
    print(f"\nRandom policy over 200 episodes: mean return {result['mean_return']:.1f} "
          f"(standard error {result['std_error']:.1f}), loss rate {result['loss_rate']:.2f}, "
          f"{time.time() - t:.2f} s")

    from unified_planning.io import PDDLReader
    from unified_planning.shortcuts import OneshotPlanner, get_environment
    from unified_planning.engines import PlanGenerationResultStatus

    get_environment().credits_stream = None
    domain = """
    (define (domain line)
      (:requirements :strips :typing)
      (:types spot)
      (:predicates (at ?s - spot) (next ?a - spot ?b - spot))
      (:action go :parameters (?a - spot ?b - spot)
        :precondition (and (at ?a) (next ?a ?b))
        :effect (and (at ?b) (not (at ?a)))))
    """
    problem = """
    (define (problem line3) (:domain line)
      (:objects s0 s1 s2 - spot)
      (:init (at s0) (next s0 s1) (next s1 s2))
      (:goal (at s2)))
    """
    up_problem = PDDLReader().parse_problem_string(domain, problem)
    with OneshotPlanner(problem_kind=up_problem.kind,
                        optimality_guarantee=PlanGenerationResultStatus.SOLVED_OPTIMALLY) as planner:
        plan = planner.solve(up_problem)
        planner_name = planner.name
    if plan.plan is None:
        raise RuntimeError("The planner did not find a plan for the test problem")
    print(f"\nPlanner test: {planner_name} found a plan with {len(plan.plan.actions)} actions")

    os.makedirs("figures", exist_ok=True)
    fig, _ = mission.plot_mission(info)
    fig.savefig("figures/mission.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/mission.png")
    print("\nSetup OK")


if __name__ == "__main__":
    main()
