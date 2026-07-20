"""HW3 task implementation file.

This is the only file to submit. It contains the callbacks wired into
the three explorer agents:

- :class:`explorer.ALExplorer` calls :func:`select_candidate_queries` and
  :func:`get_query_score` to drive its active-learning loop.

The explorers deliberately do **not** pass a per-instance state container to
your callbacks: every call receives only ``obs``, ``info``, and
``task_info``. If your strategy needs to remember information across
callback invocations (for example, the active learner needs the history of
positions it has already sampled), use the module-level dictionary
:data:`g_task_info` as persistent scratch space. Anything you write into
``g_task_info`` will still be there on the next call. You are responsible
for detecting and resetting your state when a new episode starts (a typical
trigger is the observation reporting the agent at ``task_info["initial_position"]``
with energy equal to ``task_info["initial_energy"]``).
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np  # noqa: F401  (you will almost certainly want this)

from caldera_env import (  # noqa: F401
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_SOUTH,
    MOVE_WEST,
    NO_OP,
    SAMPLE,
)
from utils import (  # noqa: F401
    Position,
    bivariate_normal,
    generate_path,
    is_position_within_bounding_box,
    validate_bounds,
)


# ===========================================================================
# Module-level persistent state
# ===========================================================================
#: Persistent state shared across callback invocations in this module.
#:
#: The HW2 explorers do not give callbacks access to the
#: environment object, and they do not track per-callback state for you.
#: Anything you need to remember across calls (a fitted surrogate model,
#: hyperparameters, a per-task cache, the running list of positions you
#: have already sampled, etc.) goes in ``g_task_info``. The dict is cleared
#: at module import time only; values written here persist for the lifetime
#: of the Python process.
g_task_info: Dict = {}


# ==========================================================================
# Global learning parameters
# ==========================================================================
#: Tunable parameters used by the reinforcement-learning explorer.
#:
#: ``LEARNING_PARAMS`` is part of your solution and **you need to tune it**.
#: A few keys are reserved by the framework or referenced by the notebook
#: tests:
#:
#: - ``epsilon_start`` is read directly by :class:`explorer.RLExplorer` to
#:   initialize ``self.epsilon``, so this key must be present and numeric
#:   in your final submission.
#: - ``learning_rate``, ``discount_factor``, ``epsilon_min``, and
#:   ``epsilon_decay`` are consumed only by your own callbacks
#:   (:func:`epsilon_update_fn`, :func:`step_update_fn`,
#:   :func:`episode_update_fn`). If your callbacks do not read a key, you
#:   may leave it as ``None``.
#:
#: You are free to **add new keys** for any custom hyperparameters your
#: solution needs &mdash; the explorer forwards the whole dictionary to
#: your callbacks unchanged.
LEARNING_PARAMS: Dict[str, float] = {
    "learning_rate": None,
    "discount_factor": None,
    "epsilon_start": None,
    "epsilon_min": None,
    "epsilon_decay": None,
}


# ===========================================================================
# Active learning: Baseline
# ===========================================================================

def select_candidate_queries_baseline(
    obs: dict,
    info: dict,
    candidate_set_size: int,
    task_info: dict,
) -> List[Position]:
    """Select candidate sampling-grid positions to score.

    Called once per step by :class:`explorer.ALExplorer`. The explorer scores
    each returned candidate via :func:`get_query_score`, teleports to the
    best-scoring one, and either samples there (positive score) or issues a
    ``NO_OP`` (non-positive score). Returning an empty list signals to the
    explorer that no candidate is worth visiting and ends the episode.

    Args:
        obs (dict): Current observation. Contains the agent's current
            ``position`` (``np.ndarray`` of shape ``(2,)``), ``energy``,
            ``sampled_before``, ``value``, ``max_value_observed``, and
            ``min_value_observed`` (see :class:`caldera_env.CalderaEnv`).
        info (dict): Auxiliary info dict from the environment. May contain
            an ``action_mask`` array of length ``len(ACTION_NAMES)``.
        candidate_set_size (int): Number of candidate positions to return.
            The explorer uses this as an upper bound &mdash; returning fewer
            is allowed.
        task_info (dict): Static task description provided by the notebook
            for the active-learning task. Contains:

            - ``"grid_dimensions"``: ``(dim_x, dim_y)`` of the map,
            - ``"sampling_res"``: side length of one sampling cell,
            - ``"initial_position"``: ``(x, y)`` agent start position,
            - ``"initial_energy"``: starting energy budget.

    Returns:
        list[Position]: Up to ``candidate_set_size`` candidate positions on
        the sampling grid (each coordinate must be in ``[0, dim_x]`` /
        ``[0, dim_y]`` and a multiple of ``task_info["sampling_res"]``).
        Use :class:`utils.Position` or plain ``(x, y)`` 2-tuples.
    """
    #TODO Task 
    candidate_positions = ...

    return candidate_positions


def get_query_score_baseline(
    position: Position,
    obs: dict,
    info: dict,
    task_info: dict,
) -> float:
    """Score a candidate position for sampling.

    Called once per candidate returned by :func:`select_candidate_queries`.
    The explorer teleports to the highest-scoring candidate and issues a
    ``SAMPLE`` action when the best score is strictly positive, or a
    ``NO_OP`` otherwise. The scale of the score does not matter &mdash;
    only the relative ordering of candidates and the sign of the best
    score.

    Args:
        position (Position): Candidate position to score.
        obs (dict): Current observation (same schema as in
            :func:`select_candidate_queries`).
        info (dict): Current info dict from the environment.
        task_info (dict): Static task description (see
            :func:`select_candidate_queries`).

    Returns:
        float: Higher is better. Strictly positive scores trigger a
        ``SAMPLE`` action at ``position``; non-positive scores trigger
        ``NO_OP``.
    """
    #TODO Task 
    score = ...

    return score



# ===========================================================================
# Active learning: Learning
# ===========================================================================

def select_candidate_queries_learning(
    obs: dict,
    info: dict,
    candidate_set_size: int,
    task_info: dict,
) -> List[Position]:
    """Select candidate sampling-grid positions to score.

    Called once per step by :class:`explorer.ALExplorer`. The explorer scores
    each returned candidate via :func:`get_query_score`, teleports to the
    best-scoring one, and either samples there (positive score) or issues a
    ``NO_OP`` (non-positive score). Returning an empty list signals to the
    explorer that no candidate is worth visiting and ends the episode.

    Args:
        obs (dict): Current observation. Contains the agent's current
            ``position`` (``np.ndarray`` of shape ``(2,)``), ``energy``,
            ``sampled_before``, ``value``, ``max_value_observed``, and
            ``min_value_observed`` (see :class:`caldera_env.CalderaEnv`).
        info (dict): Auxiliary info dict from the environment. May contain
            an ``action_mask`` array of length ``len(ACTION_NAMES)``.
        candidate_set_size (int): Number of candidate positions to return.
            The explorer uses this as an upper bound &mdash; returning fewer
            is allowed.
        task_info (dict): Static task description provided by the notebook
            for the active-learning task. Contains:

            - ``"grid_dimensions"``: ``(dim_x, dim_y)`` of the map,
            - ``"sampling_res"``: side length of one sampling cell,
            - ``"initial_position"``: ``(x, y)`` agent start position,
            - ``"initial_energy"``: starting energy budget.

    Returns:
        list[Position]: Up to ``candidate_set_size`` candidate positions on
        the sampling grid (each coordinate must be in ``[0, dim_x]`` /
        ``[0, dim_y]`` and a multiple of ``task_info["sampling_res"]``).
        Use :class:`utils.Position` or plain ``(x, y)`` 2-tuples.
    """
    #TODO Task 
    candidate_positions = ...

    return candidate_positions


def get_query_score_learning(
    position: Position,
    obs: dict,
    info: dict,
    task_info: dict,
) -> float:
    """Score a candidate position for sampling.

    Called once per candidate returned by :func:`select_candidate_queries`.
    The explorer teleports to the highest-scoring candidate and issues a
    ``SAMPLE`` action when the best score is strictly positive, or a
    ``NO_OP`` otherwise. The scale of the score does not matter &mdash;
    only the relative ordering of candidates and the sign of the best
    score.

    Args:
        position (Position): Candidate position to score.
        obs (dict): Current observation (same schema as in
            :func:`select_candidate_queries`).
        info (dict): Current info dict from the environment.
        task_info (dict): Static task description (see
            :func:`select_candidate_queries`).

    Returns:
        float: Higher is better. Strictly positive scores trigger a
        ``SAMPLE`` action at ``position``; non-positive scores trigger
        ``NO_OP``.
    """
    #TODO Task 
    score = ...

    return score
