"""Utility code for Tutorial 08 (Policy Gradient).

This package holds the non-pedagogical scaffolding for the notebook:
plotting helpers, seeding, result caching, the tabular Q-learning runner
used in the VaultEnv demo, and the DQN trainer imported from Tutorial 07
for the sample-efficiency comparison. The algorithmic code that the
tutorial actually teaches (policies, REINFORCE, the custom environments)
lives in the notebook cells themselves.
"""

from .common import set_seed, load_or_run
from .plotting import (
    rolling_mean,
    plot_seed_curves,
    plot_vault_landscape,
    plot_trajectories,
    plot_gradient_samples,
    plot_probe_histograms,
    plot_aim_policy,
)
from .tabular import tabular_q_learning, greedy_policy
from .dqn import QNet, ReplayBuffer, linear_schedule, train_dqn
