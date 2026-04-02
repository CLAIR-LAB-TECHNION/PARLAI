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
    return action

# example stochastic effect function that simulates a wrong turn with a certain probability
def stochastic_effet_wrong_turn(env, action, success_probability=0.8):
    effective_action = action
    if action in MOVEMENT_ACTIONS:
        if np.random.random() >= success_probability:
            effective_action = env.WRONG_TURN_ACTION[action]
    return effective_action


def reward_function_explore(env, action, obs) -> float:
    reward = 0
    if action != SAMPLE:
        reward = -0.1

    if action == SAMPLE:
        if obs["sampled_before"]:
            reward = -0.5
        else:
            reward = 1.0
    return reward


class CalderaEnv(BaseCalderaEnv):
    def _get_observation_space(self):
        observation_space = {
            "position": spaces.Box(
                low=np.array([0, 0], dtype=np.int64),
                high=self.max_position.copy(),
                shape=(2,),
                dtype=np.int64,
            ),
            "energy": spaces.Discrete(self.initial_energy + 1),
            "sampled_before": spaces.Discrete(2),
            "value": spaces.Box(
                low=np.array(-np.inf, dtype=np.float64),
                high=np.array(np.inf, dtype=np.float64),
                shape=(),
                dtype=np.float64,
            ),
        }

        return spaces.Dict(observation_space)

    def _get_observation(
        self,
        action_result: Optional[Union[np.ndarray, Tuple[int, float]]],
    ):
        if isinstance(action_result, tuple):
            sampled_before, value = action_result
        else:
            sampled_before = int(
                position_to_indices(self.position, self.sampling_res) in self.sampled_cells
            )
            value = DEFAULT_VALUE

        observation = {
            "position": np.asarray(self.position, dtype=np.int64),
            "energy": int(self.energy),
            "sampled_before": int(sampled_before),
            "value": np.float64(value),
        }

        return observation

    def _get_sample(self) -> Tuple[int, float]:
        sampling_grid_cell = position_to_indices(self.position, self.sampling_res)
        sampled_before = int(sampling_grid_cell in self.sampled_cells)
        if not sampled_before:
            sampled_value = float(self._get_value(sampling_grid_cell))
            self.sampled_cells[sampling_grid_cell] = sampled_value
        else:
            sampled_value = float(self.sampled_cells[sampling_grid_cell])

        self.max_value_observed = max(self.max_value_observed, sampled_value)
        self.min_value_observed = min(self.min_value_observed, sampled_value)

        return sampled_before, sampled_value

    def _perform_move(self, action: str) -> Tuple[np.ndarray, bool]:
        if action not in MOVEMENT_ACTIONS:
            raise ValueError(f"_perform_move only supports movement actions {MOVEMENT_ACTIONS}")

        x_pos, y_pos = self.position

        if action == MOVE_NORTH:
            y_pos += self.movement_size
        elif action == MOVE_SOUTH:
            y_pos -= self.movement_size
        elif action == MOVE_WEST:
            x_pos -= self.movement_size
        elif action == MOVE_EAST:
            x_pos += self.movement_size

        proposed_position = (int(x_pos), int(y_pos))

        if not validate_bounds(proposed_position, self.max_position):
            print(f"Movement out of bounds for {tuple(proposed_position)}. Staying at {tuple(self.position)}")
            return self.position.copy(), False

        path_positions = generate_path(tuple(map(int, self.position)), proposed_position)

        collision_occurred = False
        for path_cell in path_positions:
            if self.is_occupied(path_cell, include_agent=False):
                if self.end_episode_on_collision:
                    collision_occurred = True
                    print(f"Movement collided with vehicle at {path_cell}. Episode terminated.")
                else:
                    print(f"Movement blocked by vehicle at {path_cell}. Staying at {tuple(self.position)}")
                return self.position.copy(), collision_occurred

        next_position = np.array(proposed_position, dtype=np.int64)

        return next_position, collision_occurred


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

    def get_vehicle_locations(self, include_agent: bool = False):
        raise AttributeError(
            "get_vehicle_locations is not available in POCalderaEnv because the environment is partially observable."
        )

    # for each of the 8 cardinal and intercardinal directions,
    # check if there is an obstacle within the observability distance
    def _get_surrounding_obstacles(
        self,
        position: Tuple[int, int]
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
