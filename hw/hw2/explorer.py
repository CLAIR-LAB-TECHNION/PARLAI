"""Caldera explorer agents.

This module defines three explorer agents used in HW2:

- :class:`ALExplorer` -- active-learning style sampling. The agent teleports
  to a chosen sampling-grid position and takes a sample (no physical movement
  dynamics).
- :class:`PlanningExplorer` -- look-ahead planning. The agent is constrained
  by the environment's movement and energy dynamics and follows a policy that
  is precomputed (and optionally refined per step).
- :class:`RLExplorer` -- tabular reinforcement learning (Q-learning style).
  The agent learns from rewards while acting in the environment.

All three subclass :class:`agent.Agent`, share the same ``step_info`` /
``episode_info`` schema (see :mod:`agent`), and avoid exposing the underlying
environment object to user-supplied callbacks. Callbacks receive only:

- ``obs`` / ``info`` (current observation and info from the env),
- ``task_info`` (a static description of the task), and
- the persistent containers the explorer maintains (a :class:`Policy` or a
  :class:`QTable`, depending on the explorer).

This separation keeps the callback-facing code realistic: a callback cannot
peek at the depth map, vehicle layout, or other environment internals.
"""

from __future__ import annotations

from abc import ABC
from typing import Callable, Optional

import numpy as np

from agent import Agent
from caldera_env import NO_OP, SAMPLE
from value_and_policy import Policy, QTable
from utils import Position


#: Optimization directions supported by :class:`Explorer`. ``is_better(a, b)``
#: returns ``True`` when ``a`` is preferred over ``b``.
OBJECTIVES = {
    "deepest": {
        "is_better": lambda value, incumbent: value < incumbent,
    },
    "shallowest": {
        "is_better": lambda value, incumbent: value > incumbent,
    },
}


def get_planning_state(obs: dict, info: dict) -> str:
    """Default state-key function: serialize the agent position as a string.

    Args:
        obs (dict): Observation dict containing a ``"position"`` array-like.
        info (dict): Unused; accepted for API symmetry with other state-key
            functions that may consume info.

    Returns:
        str: ``str((x, y))`` for the integer agent position.
    """
    return str(tuple(map(int, obs["position"])))


class Explorer(Agent, ABC):
    """Abstract base class for agents that explore a Caldera environment.

    Adds two pieces of bookkeeping shared by all explorers:

    - ``best_value_observed`` / ``best_position_observed`` -- the best sampled
      value seen so far (in the direction defined by ``objective``) and the
      position where it was observed. These accumulate across rollouts and
      are never auto-cleared.
    - ``task_info`` -- a static dict describing the task. Concrete explorers
      forward this dict to user-supplied callbacks.

    Args:
        env: A Caldera environment (any subclass of
            :class:`caldera_env.BaseCalderaEnv`).
        objective (dict): An entry from :data:`OBJECTIVES`. Defaults to
            ``OBJECTIVES["deepest"]``.
        task_info (dict, optional): Static task description forwarded to
            user-supplied callbacks. Defaults to ``None``.
    """

    def __init__(self, env, objective=OBJECTIVES["deepest"], task_info: dict = None):
        super().__init__(env=env)
        self.objective = objective
        self.best_value_observed: Optional[float] = None
        self.best_position_observed: Optional[Position] = None
        self.task_info = task_info if task_info is not None else {}

    def get_best_value(self):
        """Return ``[best_position_observed, best_value_observed]`` as a list."""
        return [self.best_position_observed, self.best_value_observed]

    def update_best_value(self, step_info: dict) -> None:
        """Update ``best_value_observed`` from a transition.

        Reads the post-action observation (``step_info["next_obs"]``) and
        updates the running best whenever the new sampled value beats the
        incumbent according to ``self.objective``. Observations whose
        ``value`` field equals the environment's ``default_value`` (no fresh
        sample took place) are ignored.

        Args:
            step_info (dict): A step-info dict as documented in
                :meth:`agent.Agent.step_update`.
        """
        next_obs = step_info["next_obs"]
        position = next_obs["position"]
        value = next_obs["value"]
        if value is None:
            return
        value = float(np.asarray(value).item())
        if value == self.env.get_default_value():
            return
        if (
            self.best_value_observed is None
            or self.objective["is_better"](value, self.best_value_observed)
        ):
            self.best_value_observed = value
            self.best_position_observed = tuple(map(int, position))


class ALExplorer(Explorer):
    """Active-learning explorer.

    Unlike the planning and RL explorers, this agent is not constrained by
    physical movement dynamics. At each step it asks the callbacks for a set
    of candidate sampling-grid positions, scores them, and teleports to the
    best-scoring candidate before issuing a ``SAMPLE`` action. If the best
    score is ``<= 0`` the agent issues a ``NO_OP`` instead (interpreted as "no
    candidate is worth sampling").

    The explorer does not pass the environment object to the callbacks;
    callbacks receive only the current observation, info dictionary, and static
    task information.

    The explorer **does** count successful samples in
    :attr:`samples_collected` so it can enforce the sample budget via
    :meth:`is_done`. This counter is never auto-reset; create a new
    :class:`ALExplorer` instance if you want a fresh budget.

    Args:
        env: The Caldera environment.
        objective (dict): Optimization direction. Defaults to
            ``OBJECTIVES["deepest"]``.
        task_info (dict, optional): Static task description forwarded to
            callbacks.
        max_samples (int): Sample budget. The explorer terminates the
            episode (via :meth:`is_done`) once this many successful
            ``SAMPLE`` actions have been taken. Must be non-negative.
        candidate_set_size (int): Size of the candidate set requested from
            ``select_candidate_positions_fn`` at every step. Must be
            positive.
        select_candidate_positions_fn (Callable): Callback with
            signature
            ``(obs, info, candidate_set_size, task_info) -> list[Position]``.
            Returns the candidate positions to score.
        get_query_score_fn (Callable): Callback with signature
            ``(position, obs, info, task_info) -> float``. Returns a score
            indicating how desirable it is to sample at ``position``.
    """

    def __init__(
        self,
        env,
        objective=OBJECTIVES["deepest"],
        task_info: dict = None,
        max_samples: int = 5,
        candidate_set_size: int = 5,
        select_candidate_positions_fn: Callable = None,
        get_query_score_fn: Callable = None,
    ):
        super().__init__(env=env, objective=objective, task_info=task_info)
        if max_samples < 0:
            raise ValueError("max_samples must be non-negative")
        if candidate_set_size <= 0:
            raise ValueError("candidate_set_size must be positive")

        self.max_samples = int(max_samples)
        self.candidate_set_size = int(candidate_set_size)
        self.select_candidate_positions_fn = select_candidate_positions_fn
        self.get_query_score_fn = get_query_score_fn
        self.samples_collected: int = 0
        self._exhausted_candidates: bool = False

    def is_done(self) -> bool:
        """Return ``True`` once the budget or candidate space is exhausted."""
        return self.samples_collected >= self.max_samples or self._exhausted_candidates

    def select_action(self, obs: dict, info: dict) -> str:
        """Pick the next action by scoring candidate sampling positions.

        Steps:

        1. Call ``select_candidate_positions_fn`` to obtain a candidate set.
        2. Score each candidate via ``get_query_score_fn``.
        3. Teleport the agent to the best-scoring candidate.
        4. Return ``SAMPLE`` if the best score is positive, else ``NO_OP``.

        Args:
            obs (dict): Current observation.
            info (dict): Current info dict from the environment.

        Returns:
            str: One of ``SAMPLE`` or ``NO_OP``.
        """
        candidate_positions = self.select_candidate_positions_fn(
            obs,
            info,
            self.candidate_set_size,
            self.task_info,
        )
        if not candidate_positions:
            self._exhausted_candidates = True
            return NO_OP

        scored_candidates = [
            (
                self.get_query_score_fn(
                    position=position,
                    obs=obs,
                    info=info,
                    task_info=self.task_info,
                ),
                position,
            )
            for position in candidate_positions
        ]

        best_score, best_position = max(scored_candidates, key=lambda item: item[0])
        self.env.set_position(best_position)
        if best_score > 0:
            return SAMPLE
        return NO_OP

    def step_update(self, step_info: dict) -> None:
        """Update the running best value and count successful samples.

        A step counts as a successful sample when the action was ``SAMPLE``
        and the post-step observation reports ``sampled_before == 0`` (the
        cell had not been sampled before). Only successful samples increment
        :attr:`samples_collected` (and therefore consume the sample budget).
        """
        super().update_best_value(step_info)

        action = step_info["action"]
        next_obs = step_info["next_obs"]
        if action == SAMPLE and int(next_obs.get("sampled_before", 1)) == 0:
            self.samples_collected += 1

    def episode_update(self, episode_info: dict) -> None:
        """No-op. The AL explorer has no end-of-episode update."""
        return None


class PlanningExplorer(Explorer):
    """Planning-based explorer.

    The agent is given a budget for an offline planning phase that runs once,
    inside ``__init__``, via the ``init_policy_fn`` callback. After
    that, the agent simply looks up its precomputed policy at every step.

    .. important::
       During internal grading the planning phase (``init_policy_fn``) is
       capped at **5 minutes of wall-clock time**. Solutions that exceed this
       limit will be killed.

    A ``update_policy_fn`` callback is also exposed for users who want to
    do on-line replanning (one call per environment step). Its use is
    optional; the default no-op implementation provided in
    :mod:`to_implement` simply leaves the policy unchanged.

    The planner does not maintain a Q-table or a value table -- value
    storage, if any, is up to the callback implementation.

    Args:
        env: The Caldera environment.
        objective (dict): Optimization direction. Defaults to
            ``OBJECTIVES["deepest"]``.
        task_info (dict, optional): Static task description forwarded to
            callbacks.
        resources: Free-form planning-budget container forwarded to
            callbacks. May be ``None``.
        init_policy_fn (Callable): User callback with signature
            ``(policy, resources, objective, task_info, state_key_fn) ->
            None``. Mutates ``policy`` to install the planned actions.
        update_policy_fn (Callable): User callback with signature
            ``(policy, resources, objective, task_info, state_key_fn,
            step_info) -> None``. Called after every environment step when
            ``self.training`` is ``True``.
        default_action (str): Action to take when the planned policy has no
            entry for the current state. Defaults to ``NO_OP``.
    """

    def __init__(
        self,
        env,
        objective=OBJECTIVES["deepest"],
        task_info: dict = None,
        resources=None,
        init_policy_fn: Callable = None,
        update_policy_fn: Callable = None,
        default_action: str = NO_OP,
    ):
        super().__init__(env=env, objective=objective, task_info=task_info)
        self.resources = resources
        self.state_key_fn = get_planning_state
        self.init_policy_fn = init_policy_fn
        self.update_policy_fn = update_policy_fn
        self.default_action = default_action

        self.policy = Policy()
        self._done: bool = False
        self._run_initial_planning()

    def _run_initial_planning(self) -> None:
        """Call ``init_policy_fn`` once to populate the policy."""
        if self.init_policy_fn is not None:
            self.init_policy_fn(
                self.policy,
                self.resources,
                self.objective,
                self.task_info,
                self.state_key_fn,
            )

    def is_done(self) -> bool:
        """Return whether the planned policy has no remaining useful action."""
        return self._done

    def select_action(self, obs: dict, info: dict) -> str:
        """Look up the planned action for the current state.

        Falls back to :attr:`default_action` when the policy has no entry
        for the current state key. A missing action marks the planner as done
        after the fallback action is returned, preventing long runs of repeated
        ``NO_OP`` actions.
        """
        planning_state = self.state_key_fn(obs, info)
        action = self.policy.get_action(planning_state)
        if action is None:
            self._done = True
            return self.default_action
        return action

    def step_update(self, step_info: dict) -> None:
        """Update bookkeeping and (in training mode) replan once per step."""
        super().update_best_value(step_info)
        if self.training and self.update_policy_fn is not None:
            self.update_policy_fn(
                self.policy,
                self.resources,
                self.objective,
                self.task_info,
                self.state_key_fn,
                step_info,
            )

        next_state = self.state_key_fn(
            step_info["next_obs"],
            step_info.get("info", {}),
        )
        next_action = self.policy.get_action(next_state)
        if next_action is None or next_action == NO_OP:
            self._done = True

    def episode_update(self, episode_info: dict) -> None:
        """No-op. The planning explorer has no end-of-episode update."""
        return None


class RLExplorer(Explorer):
    """Tabular Q-learning style reinforcement-learning explorer.

    The agent maintains a :class:`QTable` that persists across episodes (so
    that learning accumulates over a training loop). Per-step updates are
    dispatched to ``step_update_fn`` (TD-style learning); end-of-episode
    updates are dispatched to ``episode_update_fn`` (Monte-Carlo-style
    learning). Both callbacks are skipped when ``self.training`` is
    ``False``, which lets the user disable learning for visualization or
    evaluation rollouts via :meth:`agent.Agent.eval`.

    A :class:`Policy` instance is also maintained on :attr:`policy` and
    forwarded to both update callbacks. The default action-selection logic
    is Q-greedy over the Q-table, so a vanilla Q-learning solution does not
    need to write to ``policy`` at all. ``policy`` is offered as an optional
    scratch container for solutions that want to maintain an explicit on-policy representation (e.g., stochastic / soft policies, or
    bookkeeping shared between ``step_update_fn`` and ``episode_update_fn``).

    Action selection uses epsilon-greedy: with probability ``epsilon`` the
    agent selects a uniformly random currently-available action; otherwise it
    picks the Q-greedy available action for the current state. In evaluation
    mode (``self.training == False``), epsilon-exploration is disabled and the
    agent always picks the Q-greedy action.

    During training, the explorer can call a user-supplied
    ``epsilon_update_fn`` before each action to update :attr:`epsilon` from the
    current counters and ``learning_params``. This lets users choose any
    exploration schedule they want while keeping the runner automatic.

    Args:
        env: The Caldera environment.
        objective (dict): Optimization direction. Defaults to
            ``OBJECTIVES["deepest"]``.
        task_info (dict, optional): Static task description forwarded to
            callbacks. Must contain ``"action_names"`` (the tuple of action
            strings the agent is allowed to choose from).
        resources: Free-form learning-budget container forwarded to
            callbacks. May be ``None``.
        discount_factor (float): Discount factor ``gamma``. Stored on the
            explorer for convenience; the user callback is free to
            reference it via ``task_info`` or to ignore it.
        learning_params (dict, optional): Optional dict of hyperparameters.
            Passed through to ``epsilon_update_fn`` and user update
            callbacks. For backwards compatibility, ``"epsilon"`` is accepted
            as the initial epsilon when ``"epsilon_start"`` is missing.
        state_key_fn (Callable): State-key function with signature
            ``(obs, info) -> str``. Defaults to :func:`get_planning_state`.
        epsilon_update_fn (Callable): Optional callback with signature
            ``(current_epsilon, episode_index, step_index, total_steps,
            learning_params) -> float``. Called before each action in training
            mode.
        step_update_fn (Callable): User callback with signature
            ``(policy, q_table, resources, objective, task_info,
            state_key_fn, step_info) -> None``. Called after every step when
            ``self.training`` is ``True``.
        episode_update_fn (Callable): User callback with signature
            ``(policy, q_table, resources, objective, task_info,
            state_key_fn, episode_info) -> None``. Called at the end of every
            episode when ``self.training`` is ``True``.
        default_action (str): Action to fall back to when the action space is
            empty (defensive default; not normally reachable). Defaults to
            ``NO_OP``.
    """

    def __init__(
        self,
        env,
        objective=OBJECTIVES["deepest"],
        task_info: dict = None,
        resources=None,
        discount_factor: float = 0.95,
        learning_params: dict = None,
        state_key_fn=get_planning_state,
        epsilon_update_fn: Callable = None,
        step_update_fn: Callable = None,
        episode_update_fn: Callable = None,
        default_action: str = NO_OP,
    ):
        super().__init__(env=env, objective=objective, task_info=task_info)

        self.discount_factor = float(discount_factor)
        self.learning_params = dict(learning_params) if learning_params is not None else {}
        self.state_key_fn = state_key_fn
        self.epsilon_update_fn = epsilon_update_fn
        self.resources = resources

        self.q_table = QTable()
        self.policy = Policy()

        self.step_update_fn = step_update_fn
        self.episode_update_fn = episode_update_fn
        self.default_action = default_action

        epsilon_start = self.learning_params.get(
            "epsilon_start",
            self.learning_params.get("epsilon", 0.1),
        )
        self.epsilon_start: float = float(epsilon_start)
        self.training_episode_index: int = 0
        self.training_step_index: int = 0
        self.total_training_steps: int = 0
        self.epsilon: float = self.epsilon_start

    def run_episode(self, env=None, render: bool = False) -> dict:
        """Run one episode while maintaining training counters."""
        if self.training:
            self.training_step_index = 0
        episode_info = super().run_episode(env=env, render=render)
        if self.training:
            self.training_episode_index += 1
        return episode_info

    def _update_epsilon_for_action(self) -> None:
        """Call ``epsilon_update_fn`` before a training action, if provided."""
        if not self.training or self.epsilon_update_fn is None:
            return
        self.epsilon = float(
            self.epsilon_update_fn(
                current_epsilon=self.epsilon,
                episode_index=self.training_episode_index,
                step_index=self.training_step_index,
                total_steps=self.total_training_steps,
                learning_params=self.learning_params,
            )
        )

    def select_action(self, obs: dict, info: dict) -> str:
        """Choose an action using epsilon-greedy over ``task_info['action_names']``.

        In training mode the agent explores with probability :attr:`epsilon`;
        in eval mode the agent always picks the Q-greedy action.
        """
        self._update_epsilon_for_action()
        state_key = self.state_key_fn(obs, info)
        action_names = tuple(self.task_info["action_names"])
        action_mask = info.get("action_mask")
        if action_mask is None:
            available_actions = action_names
        else:
            available_actions = tuple(
                action
                for action, is_available in zip(action_names, action_mask)
                if int(is_available)
            )
        if not available_actions:
            return self.default_action

        if self.training and np.random.random() < self.epsilon:
            action = np.random.choice(list(available_actions))
        else:
            action = self.q_table.get_best_action(state_key, available_actions)
        return str(action)

    def step_update(self, step_info: dict) -> None:
        """Apply a TD-style update (when training) and refresh best-value tracking."""
        super().update_best_value(step_info)
        if self.training and self.step_update_fn is not None:
            self.step_update_fn(
                self.policy,
                self.q_table,
                self.resources,
                self.objective,
                self.task_info,
                self.state_key_fn,
                step_info,
            )
        if self.training:
            self.training_step_index += 1
            self.total_training_steps += 1

    def episode_update(self, episode_info: dict) -> None:
        """Apply a policy update at the end of an episode (when training)."""
        if self.training and self.episode_update_fn is not None:
            self.episode_update_fn(
                self.policy,
                self.q_table,
                self.resources,
                self.objective,
                self.task_info,
                self.state_key_fn,
                episode_info,
            )
