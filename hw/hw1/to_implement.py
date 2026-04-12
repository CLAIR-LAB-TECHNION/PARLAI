"""Homework 1 Caldera environments and helper functions.

This module provides:
- Stochastic transition-effect helpers.
- Reward functions used during training.
- Environment variants for deterministic, stochastic, and partially observable setups.
"""

from typing import Callable, Optional, Tuple, Union

from gymnasium import spaces
import numpy as np

from caldera_env import (
    BaseCalderaEnv,
    DEFAULT_VALUE,
    DIRECTION_STEPS,
    MOVEMENT_ACTIONS,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_SOUTH,
    MOVE_WEST,
    SAMPLE,
    validate_bounds,
)
from utils import generate_path, position_to_indices


def stochastic_effet_none(env, action):
    """Return the intended action unchanged.

    Args:
        env (BaseCalderaEnv): Environment instance. Included for API compatibility
            with other stochastic-effect functions. It is not used.
        action (str): Action selected by the agent.

    Returns:
        str: The same action that was provided in ``action``.
    """
    return action


def stochastic_effet_wrong_turn(env, action, success_probability=0.8):
    """Apply a right-turn transition error to movement actions.

    With probability ``1 - success_probability``, a movement action is replaced by
    its right-turn alternative using ``env.WRONG_TURN_ACTION``. Non-movement
    actions are never changed.

    Args:
        env (BaseCalderaEnv): Environment instance that defines
            ``WRONG_TURN_ACTION``.
        action (str): Intended action selected by the agent.
        success_probability (float, optional): Probability that the intended
            movement action is executed as-is. Defaults to ``0.8``.

    Returns:
        str: The effective action after stochastic transition effects.
    """
    #TODO Task 2
    effective_action = ...

    return effective_action


def reward_function_explore(env, action, obs) -> float:
    """Reward exploration of new cells and penalize repeated sampling.

    Reward rules:
    - Non-sampling action: ``-0.1``
    - Sampling an already sampled cell: ``-0.5``
    - Sampling a new cell: ``+1.0``

    Args:
        env (BaseCalderaEnv): Environment instance. Included for API
            consistency. It is not used.
        action (str): Action executed at the current step.
        obs (dict): Observation dictionary. Must include:
            - ``sampled_before`` (int or bool): Whether the current cell was
              sampled previously.

    Returns:
        float: Scalar reward for the step.
    """
    #TODO Task 1
    reward = ...

    return reward


def reward_function_gap_to_max(env, action, obs) -> float:
    """Reward samples based on proximity to the best sampled value so far.

    Reward rules:
    - Non-sampling action: ``-0.1``
    - Sampling an already sampled cell: ``-0.5``
    - Sampling a new cell: ``-(env.max_value_observed - obs['value'])``

    This gives larger rewards to values closer to the running maximum observed
    so far.

    Args:
        env (BaseCalderaEnv): Environment instance. Must expose
            ``max_value_observed``.
        action (str): Action executed at the current step.
        obs (dict): Observation dictionary. Must include:
            - ``sampled_before`` (int or bool)
            - ``value`` (float): Sampled value at current cell.

    Returns:
        float: Scalar reward for the step.
    """
    #TODO Task 1
    reward = ...

    return reward


class CalderaEnv(BaseCalderaEnv):
    """Base Caldera environment with full observability."""

    def _get_observation_space(self):
        """Build the Gymnasium observation space.

        Returns:
            spaces.Dict: Observation space with fields:
                - ``position``: ``Box(shape=(2,), dtype=int64)``
                - ``energy``: ``Discrete(initial_energy + 1)``
                - ``sampled_before``: ``Discrete(2)``
                - ``value``: scalar ``Box(dtype=float64)``
        """
        #TODO Task 1
        observation_space = ...
        
        return spaces.Dict(observation_space)

    def _get_observation(
        self,
        action_result: Optional[Union[np.ndarray, Tuple[int, float]]],
    ):
        """Create the observation dictionary for the current state.

        If ``action_result`` is a tuple ``(sampled_before, value)``, those values
        are used directly (typically after a sample action). Otherwise,
        ``sampled_before`` is inferred from ``self.sampled_cells`` and ``value`` is
        set to ``DEFAULT_VALUE``.

        Args:
            action_result (Optional[Union[np.ndarray, Tuple[int, float]]]):
                Result returned by the previously executed action.

        Returns:
            dict: Observation with keys ``position``, ``energy``,
            ``sampled_before``, and ``value``.
        """
        #TODO Task 1
        observation = ...

        return observation

    def _get_sample(self) -> Tuple[int, float]:
        """Sample the current cell and update tracked value statistics.

        Returns:
            Tuple[int, float]: ``(sampled_before, sampled_value)`` where:
                - ``sampled_before`` is ``1`` if the cell was sampled previously,
                  else ``0``.
                - ``sampled_value`` is the cell value (retrieved from cache or map).
        """
        #TODO Task 1
        sampled_before = ...
        sampled_value = ...

        self.max_value_observed = ...
        self.min_value_observed = ...

        return sampled_before, sampled_value

    def _perform_move(self, action: str) -> Tuple[np.ndarray, bool]:
        """Execute movement step-by-step until destination or collision.

        The path is generated from the current position to the proposed
        destination. Movement stops at the first out-of-bounds or occupied cell.

        Args:
            action (str): Movement action (e.g., north/east/south/west).

        Returns:
            Tuple[np.ndarray, bool]: ``(position, collision_occurred)`` where:
                - ``position`` is the final valid position reached.
                - ``collision_occurred`` is ``True`` iff movement was interrupted
                  by obstacle or boundary collision.
        """
        #TODO Task 1
        position = ...
        collision_occurred = ...

        return position, collision_occurred


class SCalderaEnv(CalderaEnv):
    """Caldera environment with stochastic transition effects."""

    WRONG_TURN_ACTION = {
        MOVE_NORTH: MOVE_EAST,
        MOVE_EAST: MOVE_SOUTH,
        MOVE_SOUTH: MOVE_WEST,
        MOVE_WEST: MOVE_NORTH,
    }

    def __init__(
        self,
        *args,
        stochastic_effet_function: Callable[..., float] = stochastic_effet_wrong_turn,
        **kwargs,
    ):
        """Initialize stochastic Caldera environment.

        Args:
            *args: Positional arguments forwarded to ``CalderaEnv``.
            stochastic_effet_function (Callable[..., float], optional): Function
                that maps intended action to effective action. Defaults to
                ``stochastic_effet_wrong_turn``.
            **kwargs: Keyword arguments forwarded to ``CalderaEnv``.
        """
        super().__init__(*args, **kwargs)
        self.stochastic_effet_function = stochastic_effet_function

    def _get_effective_action(self, action: str) -> str:
        """Convert intended action to effective action under stochasticity.

        Args:
            action (str): Intended action.

        Returns:
            str: Effective action after applying ``stochastic_effet_function``.
        """
        if self.stochastic_effet_function is None:
            return action
        return self.stochastic_effet_function(self, action)


class POCalderaEnv(CalderaEnv):
    """Caldera environment configured for partial observability."""

    def __init__(
        self,
        *args,
        full_observability: bool = False,
        observability_distance: int = 1,
        **kwargs,
    ):
        """Initialize partially observable Caldera environment.

        Args:
            *args: Positional arguments forwarded to ``CalderaEnv``.
            full_observability (bool, optional): If ``True``, external code may
                choose to treat this environment as fully observable. Defaults to
                ``False``.
            observability_distance (int, optional): Maximum sensing distance
                (inclusive, in map cells) for obstacle detection in each of the 8
                directions. Must be non-negative. Defaults to ``1``.
            **kwargs: Keyword arguments forwarded to ``CalderaEnv``.

        Raises:
            ValueError: If ``observability_distance < 0``.
        """
        self.full_observability = full_observability
        if observability_distance < 0:
            raise ValueError("observability_distance must be non-negative")
        self.observability_distance = observability_distance
        super().__init__(*args, **kwargs)

    def get_invariant_information(self):
        """Disallow access to invariant/full-map information.

        Raises:
            AttributeError: Always raised because this environment is partially
                observable.
        """
        # DO NO CHANGE THIS!
        raise AttributeError(
            "get_invariant_information is not available in POCalderaEnv because the environment is partially observable."
        )

    def _get_observation_space(self):
        """Extend base observation space with local obstacle indicators.

        Returns:
            spaces.Dict: Observation space from ``CalderaEnv`` plus
            ``surrounding_obstacles`` as ``MultiBinary(8)``.
        """
        #TODO Task 3
        observation_space = ...

        return observation_space

    def _get_observation(
        self,
        action_result: Optional[Union[np.ndarray, Tuple[int, float]]],
    ):
        """Return current observation with local obstacle visibility.

        Args:
            action_result (Optional[Union[np.ndarray, Tuple[int, float]]]):
                Result returned by the previously executed action.

        Returns:
            dict: Base observation fields plus ``surrounding_obstacles``
            (``np.ndarray`` of 8 booleans).
        """
        #TODO Task 3
        observation = ...

        return observation

    def _get_surrounding_obstacles(
        self,
        position: Tuple[int, int],
    ) -> np.ndarray:
        """Detect nearby obstacles in 8 directions from a given position.

        For each direction in ``DIRECTION_STEPS`` (cardinal and intercardinal),
        this method scans outward from distance ``1`` to
        ``self.observability_distance`` and marks the direction as occupied when
        the first obstacle is found.

        Args:
            position (Tuple[int, int]): Agent position ``(x, y)`` in map
                coordinates.

        Returns:
            np.ndarray: Boolean array of shape ``(8,)`` where each entry is
            ``True`` if an obstacle is detectable in the corresponding direction,
            otherwise ``False``.

        Raises:
            ValueError: If ``position`` is outside map bounds.
        """
        #TODO Task 3
        occupied_directions = ...

        return occupied_directions
