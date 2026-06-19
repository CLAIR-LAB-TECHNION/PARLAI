"""HW2 task implementation file.

This is the only file to submit. It contains the callbacks wired into
the three explorer agents:

- :class:`explorer.ALExplorer` calls :func:`select_candidate_queries` and
  :func:`get_query_score` to drive its active-learning loop.
- :class:`explorer.PlanningExplorer` calls :func:`init_policy_fn` once before
  the first episode step and :func:`update_policy_fn` after every step.
- :class:`explorer.RLExplorer` reads :data:`LEARNING_PARAMS`, calls
  :func:`epsilon_update_fn` before training actions, calls
  :func:`step_update_fn` after every step and :func:`episode_update_fn` at
  the end of every episode, and uses the environment's reward signal which
  is provided here by :func:`reward_function`.

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
from value_and_policy import Policy, QTable  # noqa: F401
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
# Task 1: Active learning
# ===========================================================================

def select_candidate_queries(
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
    #TODO Task 1
    candidate_positions = ...

    return candidate_positions


def get_query_score(
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
    #TODO Task 1
    score = ...

    return score


# ===========================================================================
# Task 2: Planning
# ===========================================================================

def init_policy_fn(
    policy: Policy,
    resources,
    objective,
    task_info: dict,
    state_key_fn: Callable,
) -> None:
    """Plan once before the episode and install the actions on ``policy``.

    Called exactly once, from inside ``PlanningExplorer.__init__``, before
    the agent takes its first step. Use this hook to do all of the offline
    planning work and write the resulting actions into ``policy`` via
    ``policy.set_action(state_key, action)`` where ``state_key`` is what
    :func:`state_key_fn` returns for the observation at the corresponding
    map position.

    .. warning::
        During internal grading this function is capped at **5 minutes of
        wall-clock time per environment**. Solutions that exceed this
        limit will be killed and receive no credit for that environment.

    Args:
        policy (Policy): Mutable policy container to populate. Set actions
            via ``policy.set_action(state_key, action)``. The default
            :class:`explorer.PlanningExplorer` state-key function maps the
            observation ``{"position": (x, y)}`` to the string ``"(x, y)"``.
        resources: Free-form planning-budget container forwarded by the
            explorer. May be ``None``.
        objective: Optimization direction (entry from
            :data:`explorer.OBJECTIVES`). For HW2 you may assume the
            ``"deepest"`` objective.
        task_info (dict): Static task description provided by the notebook
            for the planning task. Contains:

            - ``"grid_dimensions"``: ``(dim_x, dim_y)`` of the map,
            - ``"sampling_res"``: side length of one sampling cell,
            - ``"movement_size"``: distance moved by one movement action,
            - ``"initial_position"``: ``(x, y)`` agent start position,
            - ``"initial_energy"``: starting energy budget,
            - ``"energy_per_move"``, ``"energy_per_sample"``,
              ``"energy_per_no_op"``: action energy costs,
            - ``"other_vehicles"``: mapping ``{(x, y): size}`` of obstacle
              vehicles (or an iterable of ``((x, y), size)`` entries),
            - ``"end_episode_on_collision"``: whether collisions are
              terminal,
            - either ``"external_depth_map"`` (a matrix) or both
              ``"pit_params"`` and ``"pit_weights"`` (Gaussian-pit
              parameters). Exactly one of these is supplied per
              environment.

        state_key_fn (Callable): Function ``(obs, info) -> state_key`` used
            by the explorer to derive a policy key from observations. To
            install an action for the agent being at position ``cell``,
            call ``policy.set_action(state_key_fn({"position": cell}, {}), action)``.
    """
    #TODO Task 2
    return None


def update_policy_fn(
    policy: Policy,
    resources,
    objective,
    task_info: dict,
    state_key_fn: Callable,
    step_info: dict,
) -> None:
    """Update the policy after one environment step.

    Called after every environment step while
    :class:`explorer.PlanningExplorer` is in training mode (i.e., between
    every pair of consecutive ``select_action`` calls). Use it to refine the
    plan online &mdash; e.g., to react to a collision, to mark a planned
    path complete, or to switch to a new sub-goal.

    Args:
        policy (Policy): Policy container to mutate.
        resources: Free-form planning-budget container forwarded by the
            explorer. May be ``None``.
        objective: Optimization direction (see :func:`init_policy_fn`).
        task_info (dict): Static task description (see
            :func:`init_policy_fn`).
        state_key_fn (Callable): ``(obs, info) -> state_key``.
        step_info (dict): Single-transition data with keys ``obs``,
            ``action``, ``reward``, ``next_obs``, ``terminated``,
            ``truncated``, ``info``. See
            :meth:`agent.Agent.step_update` for the full schema.
    """
    #TODO Task 2
    return None


# ===========================================================================
# Task 3: Reinforcement learning
# ===========================================================================

def epsilon_update_fn(
    current_epsilon: float,
    episode_index: int,
    step_index: int,
    total_steps: int,
    learning_params: dict,
) -> float:
    """Return the epsilon value to use for the next RL action.

    :class:`explorer.RLExplorer` calls this function automatically before
    each action while the agent is in training mode. The returned value is
    used by the epsilon-greedy action selector for the upcoming action.

    Args:
        current_epsilon (float): Epsilon used for the previous action.
        episode_index (int): Zero-based number of training episodes started
            so far in this training loop.
        step_index (int): Zero-based step counter within the current
            episode (reset to 0 at the start of every training episode).
        total_steps (int): Number of training transitions completed across
            all previous episodes and previous steps in the current
            episode.
        learning_params (dict): Hyperparameter dictionary, normally
            :data:`LEARNING_PARAMS`.

    Returns:
        float: Epsilon value for the upcoming action. Should be clipped to
        ``[0, 1]``.
    """
    #TODO Task 3
    epsilon = ...

    return epsilon


def state_key_fn(obs: dict, info: dict) -> str:
    """Convert an observation to the state key used by the policy / Q-table.

    Two observations that should share their Q-value estimates and policy
    entries must map to the *same* state key. Two observations that should
    learn different decisions must map to *different* keys. Choose
    carefully which fields of ``obs`` matter for the decision at hand &mdash;
    including too many fields explodes the tabular state space; including
    too few makes the Q-table average over decisions that ought to be
    distinct.

    Args:
        obs (dict): Current observation. Contains ``position``, ``energy``,
            ``sampled_before``, ``value``, ``max_value_observed``,
            ``min_value_observed``.
        info (dict): Auxiliary info dict. May contain ``action_mask``.

    Returns:
        str: A stable string suitable as a Q-table state key.
    """
    #TODO Task 3
    key = ...

    return key


def step_update_fn(
    policy: Policy,
    q_table: QTable,
    resources,
    objective,
    task_info: dict,
    state_key_fn: Callable,
    step_info: dict,
) -> None:
    """Per-transition (TD-style) learning update.

    Called by :class:`explorer.RLExplorer` after every transition while the
    agent is in training mode. Use this hook for one-step TD updates such
    as the Q-learning rule

    .. math::
        Q(s, a) \\leftarrow Q(s, a) + \\alpha \\cdot
        (r + \\gamma \\cdot \\max_{a'} Q(s', a') - Q(s, a))

    The mutable :class:`QTable` is provided so you can read and write Q-values
    directly via ``q_table.get_value``, ``q_table.set_value``, and
    ``q_table.update_value``. The :class:`Policy` is also mutable but is
    only needed if your solution maintains an explicit policy
    representation (e.g., a stochastic / softmax policy); a vanilla
    Q-learning solution can ignore ``policy``.

    Args:
        policy (Policy): Mutable policy container (optional scratch).
        q_table (QTable): Mutable Q-table to update.
        resources: Free-form learning-budget container. May be ``None``.
        objective: Optimization direction (see :func:`init_policy_fn`).
        task_info (dict): Static task description provided by the notebook
            for the RL task. Contains all the keys from the planning
            ``task_info`` except the depth-map fields, plus
            ``"action_names"`` (the tuple of action strings).
        state_key_fn (Callable): ``(obs, info) -> state_key``.
        step_info (dict): Single-transition data with keys ``obs``,
            ``action``, ``reward``, ``next_obs``, ``terminated``,
            ``truncated``, ``info`` (may contain ``action_mask``). See
            :meth:`agent.Agent.step_update` for the full schema.
    """
    #TODO Task 3
    return None


def episode_update_fn(
    policy: Policy,
    q_table: QTable,
    resources,
    objective,
    task_info: dict,
    state_key_fn: Callable,
    episode_info: dict,
) -> None:
    """End-of-episode (Monte-Carlo-style) learning update.

    Called by :class:`explorer.RLExplorer` once per training episode, after
    the rollout has terminated. Use this hook for updates that need the
    full trajectory (Monte-Carlo returns, eligibility-trace backups,
    per-episode bookkeeping, etc.).

    Args:
        policy (Policy): Mutable policy container (optional scratch).
        q_table (QTable): Mutable Q-table to update.
        resources: Free-form learning-budget container. May be ``None``.
        objective: Optimization direction (see :func:`init_policy_fn`).
        task_info (dict): Static task description (see :func:`step_update_fn`).
        state_key_fn (Callable): ``(obs, info) -> state_key``.
        episode_info (dict): Whole-episode data with keys:

            - ``"trajectory"``: list of ``step_info`` dicts for every
              transition, in order,
            - ``"total_reward"``: sum of rewards over the episode,
            - ``"num_steps"``: number of transitions in the episode.

    Implement either this function, :func:`step_update_fn`, or both. If
    your design only uses one, return ``None`` from the other.
    """
    #TODO Task 3
    return None


def reward_function(action: str, obs: dict) -> float:
    """Custom reward used by the RL explorer.

    The environment calls this function after every action with the action
    that was executed and the post-action observation, and uses the
    returned scalar as the per-step reward. Designing this function is
    where you encode "find the deepest cell" as a learning signal.

    Args:
        action (str): Action executed at this step. One of ``MOVE_NORTH``,
            ``MOVE_SOUTH``, ``MOVE_EAST``, ``MOVE_WEST``, ``SAMPLE``,
            ``NO_OP``.
        obs (dict): Observation produced after the action. Fields:

            - ``position``: agent's ``(x, y)`` after the action,
            - ``energy``: remaining energy,
            - ``sampled_before``: ``1`` if the cell at the new position
              was already sampled earlier this episode, else ``0``,
            - ``value``: sampled depth value when this observation
              followed a successful ``SAMPLE``; otherwise the environment's
              default sentinel (infinite),
            - ``max_value_observed``, ``min_value_observed``: running
              extrema of all sampled values so far in the episode.
              The running deepest value lives in ``min_value_observed``
              because deeper terrain is encoded as more-negative values.

    Returns:
        float: Scalar reward for this transition.
    """
    #TODO Task 3
    reward = ...

    return reward
