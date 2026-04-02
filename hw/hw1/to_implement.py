from typing import Callable, Optional, Tuple, Union

from gymnasium import spaces
import numpy as np

from caldera_env import (
    CalderaEnv,
    DIRECTION_STEPS,
    MOVEMENT_ACTIONS,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_SOUTH,
    MOVE_WEST,
    validate_bounds,
)


def stochastic_effet_none(env, action):
    return action


def stochastic_effet_wrong_turn(env, action, success_probability=0.8):
    effective_action = action
    if action in MOVEMENT_ACTIONS:
        if np.random.random() >= success_probability:
            effective_action = env.WRONG_TURN_ACTION[action]
    return effective_action


class SCalderaEnv(CalderaEnv):
    """Caldera environment with stochastic transition effects enabled."""

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
        super().__init__(*args, **kwargs)
        self.stochastic_effet_function = stochastic_effet_function

    def _get_effective_action(self, action: str) -> str:
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
        self.full_observability = full_observability
        if observability_distance < 0:
            raise ValueError("observability_distance must be non-negative")
        self.observability_distance = observability_distance
        super().__init__(*args, **kwargs)

    def _get_observation_space(self):
        observation_space = super()._get_observation_space()
        observation_space.spaces["surrounding_obstacles"] = spaces.MultiBinary(8)
        return observation_space

    def _get_observation(
        self,
        action_result: Optional[Union[np.ndarray, Tuple[int, float]]],
    ):
        observation = super()._get_observation(action_result)
        observation["surrounding_obstacles"] = np.asarray(
            self._get_surrounding_obstacles(tuple(map(int, self.position))),
            dtype=np.bool_,
        )
        return observation

    # for each of the 8 cardinal and intercardinal directions,
    # check if there is an obstacle within the observability distance
    def _get_surrounding_obstacles(
        self,
        position: Tuple[int, int],
    ) -> np.ndarray:
        # validate the input position
        if not validate_bounds(position, self.max_position):
            raise ValueError(
                f"map_position must be between (0, 0) and {tuple(self.max_position)}"
            )

        # convert the position to integers
        # and prepare a fixed-size array to store the occupancy status of each direction
        default_obstacle_value = False
        occupied_directions = np.full(
            len(DIRECTION_STEPS),
            default_obstacle_value,
            dtype=bool,
        )

        # go over the 8 directions and check if there is an obstacle within the observability distance
        x_pos, y_pos = map(int, position)
        for direction_index, (step_x, step_y) in enumerate(DIRECTION_STEPS.values()):
            # for each direction, check the path from position to the maximum observability distance in that direction
            # cap the path at the map boundaries
            # check if there are any obstacles on that path
            # if there is an obstacle on the path, mark that direction as occupied (otherwise leave it as not occupied)
            for distance in range(1, self.observability_distance + 1):
                candidate_position = (
                    x_pos + (step_x * distance),
                    y_pos + (step_y * distance),
                )
                if not validate_bounds(candidate_position, self.max_position):
                    break

                if self.is_occupied(candidate_position, include_agent=False):
                    occupied_directions[direction_index] = True
                    break

        return occupied_directions
