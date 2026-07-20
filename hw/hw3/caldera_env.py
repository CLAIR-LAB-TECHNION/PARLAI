
from dataclasses import dataclass, replace
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils import (
    BivariateNormalStruct,
    Position,
    bivariate_normal,
    generate_path,
    is_position_within_bounding_box,
    validate_bounds,
    validate_depth_map,
)


# Default parameters for the three Gaussian pits that define terrain depth.
DEFAULT_PIT_PARAMS = (
    BivariateNormalStruct(x=0.0, y=0.0, sigmax=0.16, sigmay=0.15, mux=0.2, muy=0.2),
    BivariateNormalStruct(x=0.0, y=0.0, sigmax=0.18, sigmay=0.2, mux=0.5, muy=0.7),
    BivariateNormalStruct(x=0.0, y=0.0, sigmax=0.17, sigmay=0.15, mux=0.8, muy=0.3),
)

# Relative influence (depth) of each pit above.
DEFAULT_PIT_WEIGHTS = (16000.0, 22000.0, 18000.0)

# Step vectors for cardinal and intercardinal directions.
DIRECTION_STEPS = {
    "NORTH": (0, 1),
    "NORTHEAST": (1, 1),
    "EAST": (1, 0),
    "SOUTHEAST": (1, -1),
    "SOUTH": (0, -1),
    "SOUTHWEST": (-1, -1),
    "WEST": (-1, 0),
    "NORTHWEST": (-1, 1),
}

# Action names and mappings used by all Caldera environments.
MOVE_NORTH = "MOVE_NORTH"
MOVE_SOUTH = "MOVE_SOUTH"
MOVE_EAST = "MOVE_EAST"
MOVE_WEST = "MOVE_WEST"
SAMPLE = "SAMPLE"
NO_OP = "NO_OP"
ACTION_NAMES = (MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST, SAMPLE, NO_OP)
ACTION_TO_INDEX = {action_name: index for index, action_name in enumerate(ACTION_NAMES)}
MOVEMENT_ACTIONS = (MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST)


@dataclass(frozen=True)
class State:
    """Minimal Caldera state used for action availability checks."""

    position: Union[Position, Tuple[int, int], np.ndarray]
    energy: int

    def __post_init__(self):
        position = tuple(map(int, self.position))
        object.__setattr__(self, "position", Position(position[0], position[1]))
        object.__setattr__(self, "energy", int(self.energy))

    @classmethod
    def from_observation(cls, obs: dict) -> "State":
        return cls(position=obs["position"], energy=obs["energy"])

    def as_dict(self) -> dict:
        return {
            "position": self.position,
            "energy": self.energy,
        }

    def __getitem__(self, key: str):
        return getattr(self, key)


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
    effective_action = action
    if action in MOVEMENT_ACTIONS:
        if np.random.random() >= success_probability:
            effective_action = env.WRONG_TURN_ACTION[action]
    return effective_action


def reward_function_default(action, obs) -> float:
    """Return the baseline reward used when no custom reward is provided.

    Args:
        action (str): Action executed at the current step.
        obs (dict): Observation after applying the action.

    Returns:
        float: Constant per-step penalty ``-0.1``.
    """
    return -0.1


class BaseCalderaEnv(gym.Env):
    """Shared Caldera environment dynamics.

    This abstract base class implements map generation (or external map
    loading), energy bookkeeping, action normalization, stepping logic,
    obstacle handling, action-mask construction, and visualization.

    Concrete subclasses (:class:`CalderaEnv`, :class:`SCalderaEnv`,
    :class:`POCalderaEnv`) implement the four template methods
    :meth:`_get_observation_space`, :meth:`_get_observation`,
    :meth:`_get_sample`, and :meth:`_perform_move`.
    """

    def __init__(
        self,
        id: str = None,
        dim_x: int = 100,
        dim_y: int = 100,
        pit_params: Sequence[BivariateNormalStruct] = DEFAULT_PIT_PARAMS,
        pit_weights: Sequence[float] = DEFAULT_PIT_WEIGHTS,
        external_depth_map: Optional[np.ndarray] = None,
        sampling_res: int = 10,
        initial_position: Position = (0, 0),
        movement_size: int = 1,
        initial_energy: int = 200,
        energy_per_move: int = 1,
        energy_per_sample: int = 1,
        energy_per_no_op: int = 1,
        other_vehicles: Optional[Sequence[Tuple[Position, int]]] = None,
        end_episode_on_collision: bool = False,
        reward_function: Callable[..., float] = reward_function_default,
        default_value: float = np.inf,
    ):
        """Initialize Caldera environment state and configuration.

        Args:
            id (str, optional): Human-readable identifier displayed in plot
                titles. Defaults to ``None``.
            dim_x (int, optional): Maximum x-coordinate (inclusive) of the
                generated map. Ignored when ``external_depth_map`` is provided.
                Defaults to ``100``.
            dim_y (int, optional): Maximum y-coordinate (inclusive) of the
                generated map. Ignored when ``external_depth_map`` is provided.
                Defaults to ``100``.
            pit_params (Sequence[BivariateNormalStruct], optional): Gaussian
                pit definitions used to generate terrain depth. Defaults to
                :data:`DEFAULT_PIT_PARAMS` when ``external_depth_map`` is
                ``None``.
            pit_weights (Sequence[float], optional): Weight per entry in
                ``pit_params``. Must have the same length as ``pit_params``.
                Defaults to :data:`DEFAULT_PIT_WEIGHTS` when
                ``external_depth_map`` is ``None``.
            external_depth_map (Optional[np.ndarray], optional): Pre-built
                depth map of shape ``(dim_y + 1, dim_x + 1)``. When provided,
                ``pit_params`` and ``pit_weights`` are ignored and the map
                dimensions are inferred from the matrix shape.
            sampling_res (int, optional): Sampling-grid cell size in map units.
                Defaults to ``10``.
            initial_position (Position, optional): Agent starting ``(x, y)``
                position. Defaults to ``(0, 0)``.
            movement_size (int, optional): Distance moved by one movement
                action, in map units. Defaults to ``1``.
            initial_energy (int, optional): Starting energy budget. Defaults
                to ``200``.
            energy_per_move (int, optional): Energy consumed by one movement
                action. Must be positive. Defaults to ``1``.
            energy_per_sample (int, optional): Energy consumed by a sample
                action. Must be positive. Defaults to ``1``.
            energy_per_no_op (int, optional): Energy consumed by a NO_OP
                action. Must be positive. Defaults to ``1``.
            other_vehicles (Optional[Sequence[Tuple[Position, int]]],
                optional): Obstacle vehicles as
                ``(bottom_right_position, size)`` entries. Defaults to
                ``None``.
            end_episode_on_collision (bool, optional): Whether a collision
                ends the episode. Defaults to ``False``.
            reward_function (Callable[[str, dict], float], optional): Reward
                callback with signature ``(action, obs) -> float``. Defaults
                to :func:`reward_function_default`.
            default_value (float, optional): Sentinel value used in the
                ``value`` field of an observation when the current cell has
                not just been sampled. Defaults to ``np.inf``.

        Raises:
            ValueError: If validation of any input fails (invalid pit
                parameters, non-positive energy costs, out-of-bounds initial
                position, etc.).
        """
        self.id = id

        # If a map is provided, use it directly. The validated matrix is
        # stored on both ``self.depth_map`` (the canonical source of truth for
        # sampling) and ``self.external_depth_map`` (so callers can tell the
        # map came from outside vs. being generated from pits). When the map
        # is generated from pits, ``self.external_depth_map`` is set to None.
        if external_depth_map is not None:
            external_depth_map = validate_depth_map(external_depth_map)
            if external_depth_map is None:
                raise ValueError("external_depth_map is not valid")
            self.dim_y = external_depth_map.shape[0] - 1
            self.dim_x = external_depth_map.shape[1] - 1
            self.depth_map = external_depth_map
            self.external_depth_map = external_depth_map
            self.pit_params = None
            self.pit_weights = None

        # otherwise, generate a map based on the provided pit parameters and weights (or defaults if not provided)
        else:
            if pit_params is None:
                pit_params = DEFAULT_PIT_PARAMS
            if pit_weights is None:
                pit_weights = DEFAULT_PIT_WEIGHTS
            self.dim_x = dim_x
            self.dim_y = dim_y
            # Validate the pit parameters and weights
            self.pit_params = list(pit_params)
            self.pit_weights = list(pit_weights)
            if len(self.pit_params) != len(self.pit_weights):
                raise ValueError("pit_params and pit_weights must have the same length")
            for params in self.pit_params:
                if not isinstance(params, BivariateNormalStruct):
                    raise ValueError(
                        "Each pit entry must be a BivariateNormalStruct with x, y, sigmax, sigmay, mux, and muy"
                    )
            _, _, self.depth_map = self._generate_caldera_map()
            self.external_depth_map = None
            
            
        self.max_position = np.array([self.dim_x, self.dim_y], dtype=np.int64)

        # Initilize the agent parameters and state variables
        self.initial_energy = initial_energy
        self.energy_per_move = energy_per_move
        self.energy_per_sample = energy_per_sample
        self.energy_per_no_op = energy_per_no_op
        if self.energy_per_move <= 0:
            raise ValueError("energy_per_move must be positive")
        if self.energy_per_sample <= 0:
            raise ValueError("energy_per_sample must be positive")
        if self.energy_per_no_op <= 0:
            raise ValueError("energy_per_no_op must be positive")
        self.energy = self.initial_energy

        self.movement_size = movement_size
        self.end_episode_on_collision = end_episode_on_collision
        self.default_value = float(default_value)

        # The positions of the agent and vehicles are in the dimensions of the map
        # whenever reset is applied, the agent returns to the initial position and energy level, and all sampled cells are cleared.
        if not validate_bounds(initial_position, self.max_position):
            raise ValueError(
                f"initial_position must be between (0, 0) and {tuple(self.max_position)}"
            )
        self.initial_position = np.array(
            list(map(int, initial_position)), dtype=np.int64
        )
        self.position = self.initial_position.copy()

        # The sampling process is based on a grid defined by the sampling resolution,
        # and the agent can only sample at the bottom left corner of these grid cells.
        self.sampling_res = (
            sampling_res  # defining the resultion of the sampling process
        )
        self.x_grid_coords = np.arange(0, self.dim_x + 1, self.sampling_res)
        self.y_grid_coords = np.arange(0, self.dim_y + 1, self.sampling_res)
        # to keep track of which grid cells have been sampled and their values
        self.sampled_cells: Dict[Position, float] = {}
        # setting the obstacles on the surface, which are represented as vehicles with a certain size.
        # The environment provides methods to add and remove vehicles and check for occupancy,
        # which are used to determine if the agent can move to a certain cell or if it is blocked by an obstacle.
        self.other_vehicles: Dict[Position, int] = {}
        self.add_vehicles(other_vehicles or [])
        self.agent_path = [tuple(map(int, self.position))]

        # initilize the reward and transition dynamics functions
        self.set_reward_function(reward_function)

        # The environment exposes a discrete action space for Gym compatibility,
        # but the implementation uses named actions internally.
        self.action_space = self._get_action_space()

        # The observation space is a dictionary containing the agent's position, remaining energy, whether the current cell has been sampled before, and the value of the current cell.
        self.observation_space = self._get_observation_space()

    def get_default_value(self) -> float:
        """Return the default observation value used before sampling."""
        return self.default_value

    def get_reward_function(self) -> Callable[..., float]:
        """Return the reward function used by the environment."""
        return self.reward_function

    def set_reward_function(self, reward_function: Callable[..., float]) -> None:
        """Set the reward function used by the environment."""
        if not callable(reward_function):
            raise ValueError("reward_function must be callable")
        self.reward_function = reward_function

    def get_max_value_observed(self) -> float:
        """Return the maximum sampled value observed so far.

        Returns:
            float: Running maximum of sampled cell values.
        """
        if not self.sampled_cells:
            return float("-inf")
        return max(self.sampled_cells.values())

    def get_min_value_observed(self) -> float:
        """Return the minimum sampled value observed so far.

        Returns:
            float: Running minimum of sampled cell values.
        """
        if not self.sampled_cells:
            return float("inf")
        return min(self.sampled_cells.values())

    def add_vehicles(
        self,
        other_vehicles: Sequence[Tuple[Position, int]],
    ) -> None:
        """Add obstacle vehicles to the map.

        Args:
            other_vehicles (Sequence[Tuple[Tuple[int, int], int]]): Iterable of
                ``(bottom_right_position, vehicle_size)`` entries.

        Raises:
            ValueError: If a vehicle corner is out of map bounds.
        """
        for bottom_right_position, vehicle_size in other_vehicles:
            if not validate_bounds(bottom_right_position, self.max_position):
                raise ValueError(f"Vehicle position must be within the map dimensions")
            top_right_position = (
                int(bottom_right_position[0]) + vehicle_size,
                int(bottom_right_position[1]) + vehicle_size,
            )
            if not validate_bounds(top_right_position, self.max_position):
                raise ValueError(f"Vehicle position must be within the map dimensions")

            self.other_vehicles[tuple(map(int, bottom_right_position))] = vehicle_size

    def remove_vehicle(
        self,
        bottom_right_position: Position,
    ) -> None:
        """Remove one obstacle vehicle by its anchor position.

        Args:
            bottom_right_position (Tuple[int, int]): Vehicle anchor coordinates.

        Raises:
            ValueError: If ``bottom_right_position`` is out of bounds.
        """
        if not validate_bounds(bottom_right_position, self.max_position):
            raise ValueError("Vehicle position must be within the map dimensions")

        validated_position = tuple(map(int, bottom_right_position))
        if validated_position not in self.other_vehicles:
            print(f"No vehicle found at position {validated_position}")
            return

        del self.other_vehicles[validated_position]

    def _is_occupied(
        self,
        cell: Position,
        include_agent: bool = False,
    ) -> bool:
        """Check whether a map cell is occupied.

        Args:
            cell (Tuple[int, int]): Cell coordinates ``(x, y)``.
            include_agent (bool, optional): If ``True``, the agent's current
                position counts as occupied. Defaults to ``False``.

        Returns:
            bool: ``True`` if out of bounds, occupied by a vehicle, or occupied
            by the agent (when ``include_agent=True``).
        """
        if not validate_bounds(cell, self.max_position):
            return True
        validated_cell = tuple(map(int, cell))

        if include_agent and validated_cell == tuple(map(int, self.position)):
            return True

        return any(
            is_position_within_bounding_box(
                validated_cell, vehicle_position, vehicle_size
            )
            for vehicle_position, vehicle_size in self.other_vehicles.items()
        )

    def _get_vehicle_locations(
        self,
        include_agent: bool = False,
    ) -> Tuple[Position, ...]:
        """Return obstacle anchor positions, optionally including the agent.

        Args:
            include_agent (bool, optional): If ``True``, include current agent
                position in returned locations. Defaults to ``False``.

        Returns:
            Tuple[Tuple[int, int], ...]: Sorted tuple of coordinate pairs.
        """
        agent_position = tuple(map(int, self.position))
        vehicle_locations = {
            vehicle_position
            for vehicle_position in self.other_vehicles
            if include_agent or vehicle_position != agent_position
        }
        if include_agent:
            vehicle_locations.add(agent_position)

        return tuple(sorted(vehicle_locations))

    def get_invariant_information(self):
        """Return environment information that does not depend on observations.

        Returns:
            Tuple[Tuple[int, int], ...]: Vehicle locations.
        """
        return self._get_vehicle_locations()

    def set_position(self, position: Position) -> None:
        """Set the agent position to a valid sampling-grid position.

        Args:
            position (Tuple[int, int]): Target position ``(x, y)``.

        Raises:
            ValueError: If ``position`` is outside the map bounds or is not on
                the sampling grid.
        """
        validated_position = tuple(map(int, position))
        x_pos, y_pos = validated_position

        if not validate_bounds(validated_position, self.max_position):
            raise ValueError(
                f"position must be between (0, 0) and {tuple(self.max_position)}"
            )
        if x_pos % self.sampling_res != 0 or y_pos % self.sampling_res != 0:
            raise ValueError(
                f"position must lie on the sampling grid with resolution {self.sampling_res}"
            )

        self.position = np.asarray(validated_position, dtype=np.int64)
        if validated_position != self.agent_path[-1]:
            self.agent_path.append(validated_position)

    def reset(self, *, seed=None, options=None):
        """Reset environment state to the initial episode configuration.

        Args:
            seed (Optional[int], optional): Random seed passed to Gym. Defaults
                to ``None``.
            options (Optional[dict], optional): Unused reset options for API
                compatibility. Defaults to ``None``.

        Returns:
            Tuple[dict, dict]: ``(observation, info)``.

        Raises:
            ValueError: If generated observation is outside
                ``self.observation_space``.
        """
        super().reset(seed=seed)
        self.position = self.initial_position.copy()
        self.energy = self.initial_energy
        self.sampled_cells = {}
        self.agent_path = [tuple(map(int, self.position))]

        obs = self._get_observation(None)
        if not self.observation_space.contains(obs):
            raise ValueError(
                "Observation is outside observation_space. "
                f"obs={obs}, observation_space={self.observation_space}"
            )
        info = {"action_mask": self._get_action_mask(self._get_current_state())}
        return obs, info

    def step(self, action: Union[int, str]):
        """Advance the environment by one action.

        Args:
            action (Union[int, str]): Action as either index in
                ``ACTION_NAMES`` or action string.

        Returns:
            Tuple[dict, float, bool, bool, dict]:
                ``(observation, reward, terminated, truncated, info)``.

        Raises:
            ValueError: If action is invalid (via ``_normalize_action``).
            ValueError: If generated observation is outside
                ``self.observation_space``.
        """
        info = {}

        normalized_action = self._normalize_action(action)
        # get the relevant energy cost
        action_energy_cost = self._get_action_energy_cost(normalized_action)
        # if the agent does not have enough energy to perform the action,
        # the episode is terminated and a reward of 0 is given,
        # with the info dictionary containing details about the energy depletion.
        if self.energy < action_energy_cost:
            obs = self._get_observation(None)
            if not self.observation_space.contains(obs):
                raise ValueError(
                    "Observation is outside observation_space. "
                    f"obs={obs}, observation_space={self.observation_space}"
                )
            info = {
                "reason": "insufficient_energy",
                "required_energy": int(action_energy_cost),
                "remaining_energy": int(self.energy),
                "action_mask": self._get_action_mask(self._get_current_state()),
            }
            # return observation, reward, terminated, truncated, and info
            return obs, 0.0, True, False, info

        # the action to perform (can be overriden by subclasses to implement stochasticity or other effects)
        effective_action = self._get_effective_action(normalized_action)

        # perform the action
        action_result = None
        collision_occurred = False

        # move
        if effective_action in MOVEMENT_ACTIONS:
            next_position, collision_occurred = self._perform_move(effective_action)
            self.position = next_position

        # sample
        if effective_action == SAMPLE:
            action_result = self._get_sample()
            sampled_before, _ = action_result
            if sampled_before:
                info.update(
                    {
                        "message": "The cell the agent was trying to sample was already sampled."
                    }
                )

        # update energy based on the energy consumed by the action
        self.energy -= action_energy_cost

        # record the agent's path
        current_position = tuple(map(int, self.position))
        if current_position != self.agent_path[-1]:
            self.agent_path.append(current_position)

        # get the observation after performing the action and consuming energy
        obs = self._get_observation(action_result)
        if not self.observation_space.contains(obs):
            raise ValueError(
                "Observation is outside observation_space. "
                f"obs={obs}, observation_space={self.observation_space}"
            )

        # compute the reward based on the action taken and the resulting observation using the provided reward function.
        reward = float(self.reward_function(normalized_action, obs))

        # check if the episode is terminated due to collision or energy depletion
        terminated = (
            collision_occurred and self.end_episode_on_collision
        ) or self.energy == 0
        truncated = False
        info["action_mask"] = self._get_action_mask(self._get_current_state())

        # return the observation, reward, terminated, truncated, and info as expected by Gym environments
        return obs, reward, terminated, truncated, info

    # checking if the current cell has been sampled before and returning the appropriate value.
    def _get_sample(self) -> Tuple[int, float]:
        """Sample the current cell.

        Returns:
            Tuple[int, float]: ``(sampled_before, sampled_value)``.

        Raises:
            NotImplementedError: Always. Implemented in
                ``to_implement.CalderaEnv``.
        """
        raise NotImplementedError(
            "_get_sample must be implemented in to_implement.CalderaEnv"
        )

    def visualize(
        self,
        agent_path: Optional[Sequence[Tuple[int, int]]] = None,
        show_gaussian_centers: bool = False,
        show_grid_lines: bool = False,
        show_agent_path: bool = True,
    ):
        """Render a contour view of the terrain and optional overlays.

        Args:
            agent_path (Optional[Sequence[Tuple[int, int]]], optional): Path to
                draw. If ``None``, use ``self.agent_path``. Defaults to ``None``.
            show_gaussian_centers (bool, optional): Whether to draw pit centers.
                Defaults to ``False``.
            show_grid_lines (bool, optional): Whether to draw sampling grid
                lines. Defaults to ``False``.
            show_agent_path (bool, optional): Whether to draw path polyline.
                Defaults to ``True``.

        Returns:
            Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The created
            figure and axes.
        """
        x, y, z = self._generate_caldera_map()
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(x, y, z, levels=20, cmap="viridis")

        if show_grid_lines:
            ax.vlines(
                self.x_grid_coords,
                ymin=self.y_grid_coords[0],
                ymax=self.y_grid_coords[-1],
                color="white",
                linewidth=0.5,
                alpha=0.2,
                zorder=5,
            )
            ax.hlines(
                self.y_grid_coords,
                xmin=self.x_grid_coords[0],
                xmax=self.x_grid_coords[-1],
                color="white",
                linewidth=0.5,
                alpha=0.2,
                zorder=5,
            )

        if show_gaussian_centers and self.pit_params is not None:
            pit_center_x = [params.mux * self.dim_x for params in self.pit_params]
            pit_center_y = [params.muy * self.dim_y for params in self.pit_params]
            ax.scatter(
                pit_center_x,
                pit_center_y,
                marker="x",
                s=50,
                c="white",
                linewidths=1.5,
                zorder=11,
            )

        path_to_plot = self.agent_path if agent_path is None else agent_path

        other_vehicle_locations = self._get_vehicle_locations()
        if other_vehicle_locations:
            for vehicle_position in other_vehicle_locations:
                vehicle_size = self.other_vehicles[vehicle_position]
                ax.add_patch(
                    Rectangle(
                        vehicle_position,
                        vehicle_size,
                        vehicle_size,
                        facecolor="black",
                        edgecolor="black",
                        linewidth=1.5,
                        alpha=0.85,
                        zorder=8,
                    )
                )

        row = int(self.position[1])
        col = int(self.position[0])
        if 0 <= row < z.shape[0] and 0 <= col < z.shape[1]:
            vehicle_value = z[row, col]
            if not np.isneginf(vehicle_value):
                ax.scatter(
                    self.position[0],
                    self.position[1],
                    marker="o",
                    s=20,
                    c="black",
                    zorder=9,
                )

        if show_agent_path and path_to_plot:
            path_x, path_y = zip(*path_to_plot)
            ax.plot(
                path_x,
                path_y,
                color="white",
                linewidth=2,
                linestyle="-",
                alpha=0.9,
                zorder=10,
            )

        fig.colorbar(contour, ax=ax, label="Relative depth")
        ax.set_title("Caldera Depth Map")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        return fig, ax

    def render(self, pause_time: float = 1.0, close: bool = True, **visualize_kwargs):
        """Render the current environment state using ``visualize``."""
        visualize_kwargs.setdefault("show_grid_lines", True)
        fig, ax = self.visualize(**visualize_kwargs)
        plt.show(block=False)
        plt.pause(pause_time)
        if close:
            plt.close(fig)
        return fig, ax

    def _get_value(self, cell: Position) -> float:
        """Return terrain value at one cell.

        Args:
            cell (Tuple[int, int]): Map coordinates ``(x, y)``.

        Returns:
            float: Depth-map value at ``cell``.

        Raises:
            ValueError: If ``cell`` is out of bounds.
        """
        if not validate_bounds(cell, self.max_position):
            raise ValueError(
                f"cell must be between (0, 0) and {tuple(self.max_position)}"
            )
        x_pos, y_pos = map(int, cell)
        return float(self.depth_map[y_pos, x_pos])

    def _caldera_sim_function(self, x, y):
        """Evaluate terrain depth function on coordinate arrays.

        Args:
            x (np.ndarray): X-coordinate grid.
            y (np.ndarray): Y-coordinate grid.

        Returns:
            np.ndarray: Negative weighted sum of bivariate normal components.
        """
        # Normalize coordinates to [0, 1] for Gaussian parameterization.
        x = x / self.dim_x
        y = y / self.dim_y

        z = np.zeros_like(x, dtype=float)
        for weight, params in zip(self.pit_weights, self.pit_params):
            z += weight * bivariate_normal(replace(params, x=x, y=y))

        # Negative sign turns Gaussian peaks into terrain pits.
        return -z

    def set_depth_map(self, depth_map: np.ndarray) -> None:
        """Replace the terrain depth map with an external matrix.

        Matrix rows map to y-coordinates and columns map to x-coordinates, so
        ``depth_map[y, x]`` is the value sampled at position ``(x, y)``. The
        map dimensions, sampling grid, and observation space are recomputed
        accordingly, and ``self.external_depth_map`` is updated so callers
        can still tell that the map came from outside.

        Args:
            depth_map (np.ndarray): 2D matrix of shape ``(dim_y + 1, dim_x + 1)``.

        Raises:
            ValueError: If ``depth_map`` cannot accommodate the current
                ``initial_position`` or ``position``.
        """
        validated_depth_map = validate_depth_map(depth_map)
        dim_y = validated_depth_map.shape[0] - 1
        dim_x = validated_depth_map.shape[1] - 1
        max_position = np.array([dim_x, dim_y], dtype=np.int64)

        if not validate_bounds(self.initial_position, max_position):
            raise ValueError("depth_map is too small for the initial_position")
        if not validate_bounds(self.position, max_position):
            raise ValueError("depth_map is too small for the current position")

        self.depth_map = validated_depth_map
        self.external_depth_map = validated_depth_map
        self.dim_y = dim_y
        self.dim_x = dim_x
        self.max_position = max_position
        self.x_grid_coords = np.arange(0, self.dim_x + 1, self.sampling_res)
        self.y_grid_coords = np.arange(0, self.dim_y + 1, self.sampling_res)
        self.observation_space = self._get_observation_space()

    def _generate_caldera_map(self):
        """Generate full terrain grids for the current map dimensions.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: ``(x_grid, y_grid, z)``.
        """
        x_coords = np.arange(0, self.dim_x + 1, dtype=np.int64)
        y_coords = np.arange(0, self.dim_y + 1, dtype=np.int64)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        if hasattr(self, "depth_map"):
            z = self.depth_map
        else:
            z = self._caldera_sim_function(x_grid, y_grid)
        return x_grid, y_grid, z

    def _get_action_space(self):
        """Return Gym action-space specification.

        Returns:
            spaces.Discrete: Discrete action space over ``ACTION_NAMES``.
        """
        return spaces.Discrete(n=len(ACTION_NAMES))

    def _get_current_state(self) -> State:
        """Return the current state fields needed for action availability."""
        return State(position=self.position, energy=self.energy)

    def _normalize_state(self, state: Union[State, dict]) -> State:
        """Normalize a state-like object into a ``State`` instance."""
        if isinstance(state, State):
            return state
        return State(position=state["position"], energy=state["energy"])

    def _get_action_mask(self, state: Union[State, dict]) -> np.ndarray:
        """Return a binary mask of currently available actions.

        The mask follows ``ACTION_NAMES`` order. A value of ``1`` means the
        action is currently available, and ``0`` means it is not.

        Returns:
            np.ndarray: Binary action mask with shape ``(len(ACTION_NAMES),)``.
        """
        return np.asarray(
            [self._is_action_available(action, state) for action in ACTION_NAMES],
            dtype=np.int8,
        )

    def get_available_actions(self, state: Optional[Union[State, dict]] = None) -> Tuple[str, ...]:
        """Return the currently available action names.

        Returns:
            Tuple[str, ...]: Available actions in ``ACTION_NAMES`` order.
        """
        if state is None:
            state = self._get_current_state()
        return tuple(
            action
            for action, is_available in zip(ACTION_NAMES, self._get_action_mask(state))
            if is_available
        )


    def _is_action_available(self, action: Union[int, str], state: Union[State, dict]) -> bool:
        """Return whether an action can currently be performed.

        Args:
            action (Union[int, str]): Action index or action name.
            state (dict): State to evaluate. Must include ``position`` and
                ``energy``.

        Returns:
            bool: ``True`` when the action is currently available.
        """
        normalized_action = self._normalize_action(action)
        state = self._normalize_state(state)
        position = state.position
        energy = state.energy

        if energy < self._get_action_energy_cost(normalized_action):
            return False

        if normalized_action in MOVEMENT_ACTIONS:
            proposed_destination = self._get_proposed_destination(
                normalized_action,
                position,
            )
            path_positions = generate_path(
                position,
                proposed_destination,
            )
            return all(
                validate_bounds(path_cell, self.max_position)
                and not self._is_occupied(path_cell, include_agent=False)
                for path_cell in path_positions
            )

        if normalized_action == SAMPLE:
            return True

        if normalized_action == NO_OP:
            return True

        return False

    def _get_observation_space(self):
        """Return Gym observation-space specification.

        Raises:
            NotImplementedError: Always. Implemented in
                ``to_implement.CalderaEnv``.
        """
        raise NotImplementedError(
            "_get_observation_space must be implemented in to_implement.CalderaEnv"
        )

    def _get_observation(
        self,
        action_result: Optional[Union[np.ndarray, Tuple[int, float]]],
    ):
        """Create observation for the current state.

        Args:
            action_result (Optional[Union[np.ndarray, Tuple[int, float]]]):
                Action-side data used to construct observation.

        Raises:
            NotImplementedError: Always. Implemented in
                ``to_implement.CalderaEnv``.
        """
        raise NotImplementedError(
            "_get_observation must be implemented in to_implement.CalderaEnv"
        )

    def _get_action_energy_cost(self, action: str) -> int:
        """Return energy required to execute an action.

        Args:
            action (str): Action name.

        Returns:
            int: Energy cost for ``action``.

        Raises:
            ValueError: If ``action`` is unsupported.
        """
        if action in MOVEMENT_ACTIONS:
            return self.energy_per_move
        if action == SAMPLE:
            return self.energy_per_sample
        if action == NO_OP:
            return self.energy_per_no_op
        raise ValueError(f"Unsupported action for energy cost: {action}")

    def _normalize_action(self, action: Union[int, str]) -> str:
        """Normalize an action from integer or string form.

        Args:
            action (Union[int, str]): Action index or action name.

        Returns:
            str: Canonical upper-case action string.

        Raises:
            ValueError: If action value is not valid.
        """
        if isinstance(action, str):
            normalized_action = action.upper()
            if normalized_action in ACTION_TO_INDEX:
                return normalized_action
            raise ValueError(f"Invalid action: {action}")

        if isinstance(action, (int, np.integer)) and self.action_space.contains(
            int(action)
        ):
            return ACTION_NAMES[int(action)]

        raise ValueError(f"Invalid action: {action}")

    def _get_effective_action(self, action: str) -> str:
        """Map intended action to executed action.

        Subclasses can override this to model stochastic transitions.

        Args:
            action (str): Intended action.

        Returns:
            str: Effective action to execute.
        """
        return action

    def _get_proposed_destination(
        self,
        action: str,
        position: Optional[Position] = None,
    ) -> Position:
        """Compute target position after one movement action.

        Args:
            action (str): One of ``MOVEMENT_ACTIONS``.

        Returns:
            Tuple[int, int]: Proposed destination ``(x, y)``.

        Raises:
            ValueError: If ``action`` is not a movement action.
        """
        if action not in MOVEMENT_ACTIONS:
            raise ValueError(
                f"_get_proposed_destination only supports movement actions {MOVEMENT_ACTIONS}"
            )

        if position is None:
            position = self.position
        x_pos, y_pos = map(int, position)
        if action == MOVE_NORTH:
            y_pos += self.movement_size
        elif action == MOVE_SOUTH:
            y_pos -= self.movement_size
        elif action == MOVE_WEST:
            x_pos -= self.movement_size
        elif action == MOVE_EAST:
            x_pos += self.movement_size

        return int(x_pos), int(y_pos)

    def _perform_move(self, action: str) -> Tuple[np.ndarray, bool]:
        """Execute movement with collision handling.

        Args:
            action (str): One of ``MOVEMENT_ACTIONS``.

        Returns:
            Tuple[np.ndarray, bool]: ``(new_position, collision_occurred)``.

        Raises:
            NotImplementedError: Always. Implemented in
                ``to_implement.CalderaEnv``.
        """
        raise NotImplementedError(
            "_perform_move must be implemented in to_implement.CalderaEnv"
        )


class CalderaEnv(BaseCalderaEnv):
    """Base Caldera environment with full observability."""

    def _get_observation_space(self):
        """Build the Gymnasium observation space.

        Returns:
            spaces.Dict: Observation space with fields:
                - ``position``: ``Box(shape=(2,), dtype=int64)`` -- current
                  agent position ``(x, y)`` on the map.
                - ``energy``: ``Discrete(initial_energy + 1)`` -- remaining
                  energy budget.
                - ``sampled_before``: ``Discrete(2)`` -- whether the cell at
                  the current position was sampled previously in this episode.
                - ``value``: scalar ``Box(dtype=float64)`` -- value of the
                  current cell when this observation followed a successful
                  ``SAMPLE`` action; otherwise the environment's default value.
                - ``max_value_observed``: scalar ``Box(dtype=float64)`` --
                  running maximum sampled value observed so far in this
                  episode. ``-inf`` until at least one sample has been taken.
                - ``min_value_observed``: scalar ``Box(dtype=float64)`` --
                  running minimum sampled value observed so far in this
                  episode. ``inf`` until at least one sample has been taken.
        """
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
            "max_value_observed": spaces.Box(
                low=np.array(-np.inf, dtype=np.float64),
                high=np.array(np.inf, dtype=np.float64),
                shape=(),
                dtype=np.float64,
            ),
            "min_value_observed": spaces.Box(
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
        """Create the observation dictionary for the current state.

        If ``action_result`` is a tuple ``(sampled_before, value)``, those
        values are used directly (typically after a successful ``SAMPLE``
        action). Otherwise, ``sampled_before`` is inferred from
        ``self.sampled_cells`` and ``value`` is set to ``self.default_value``.

        Args:
            action_result (Optional[Union[np.ndarray, Tuple[int, float]]]):
                Result returned by the previously executed action.

        Returns:
            dict: Observation with keys ``position``, ``energy``,
            ``sampled_before``, ``value``, ``max_value_observed``, and
            ``min_value_observed`` (see :meth:`_get_observation_space` for
            field semantics).
        """
        if isinstance(action_result, tuple):
            sampled_before, value = action_result
        else:
            sampled_before = int(self.was_sampled_before(self.position))
            value = self.default_value

        observation = {
            "position": np.asarray(self.position, dtype=np.int64),
            "energy": int(self.energy),
            "sampled_before": int(sampled_before),
            "value": np.asarray(value, dtype=np.float64),
            "max_value_observed": np.asarray(
                self.get_max_value_observed(), dtype=np.float64
            ),
            "min_value_observed": np.asarray(
                self.get_min_value_observed(), dtype=np.float64
            ),
        }

        return observation

    def _position_to_sampling_cell(self, position) -> Tuple[int, int]:
        """Return the ``(cell_x, cell_y)`` sampling-grid key for ``position``.

        A sampling cell groups all map positions whose bottom-left corner is
        ``(cell_x * sampling_res, cell_y * sampling_res)``. All positions
        within the same cell share the same sampled value.
        """
        x_pos, y_pos = map(int, position)
        return (x_pos // self.sampling_res, y_pos // self.sampling_res)

    def _sampling_cell_corner(self, cell_key: Tuple[int, int]) -> Tuple[int, int]:
        """Return the bottom-left map position of a sampling cell.

        Args:
            cell_key (Tuple[int, int]): ``(cell_x, cell_y)`` as returned by
                :meth:`_position_to_sampling_cell`.

        Returns:
            Tuple[int, int]: Bottom-left ``(x, y)`` map position of the cell.
        """
        cell_x, cell_y = cell_key
        return (cell_x * self.sampling_res, cell_y * self.sampling_res)

    def was_sampled_before(self, position: Position) -> bool:
        """Return whether the sampling cell containing ``position`` was sampled."""
        return self._position_to_sampling_cell(position) in self.sampled_cells

    def _get_sample(self) -> Tuple[int, float]:
        """Sample the current cell and update tracked value statistics.

        The sampled value for the current agent position is the depth-map
        value at the bottom-left corner of the sampling cell that contains
        the agent (so all positions within the same sampling cell share the
        same value). Sampled values are cached so that repeat samples are
        deduplicated.

        Returns:
            Tuple[int, float]: ``(sampled_before, sampled_value)`` where
                ``sampled_before`` is ``1`` iff the sampling cell containing
                the agent's current position had been sampled previously in
                this episode.
        """
        cell_key = self._position_to_sampling_cell(self.position)
        sampled_before = int(cell_key in self.sampled_cells)
        if not sampled_before:
            sample_position = self._sampling_cell_corner(cell_key)
            sampled_value = float(self._get_value(sample_position))
            self.sampled_cells[cell_key] = sampled_value
        else:
            sampled_value = float(self.sampled_cells[cell_key])

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
        # get the proposed destination based on the current position and the action
        proposed_destination = self._get_proposed_destination(action)

        # generate the path the agent needs to take to get to the proposed destination
        path_positions = generate_path(
            tuple(map(int, self.position)), proposed_destination
        )

        collision_occurred = False
        position = np.asarray(self.position, dtype=np.int64).copy()
        for path_cell in path_positions:
            # check if the path cell is occupied by an obstacle or is out of bounds
            if self._is_occupied(path_cell, include_agent=False) or not validate_bounds(
                path_cell, self.max_position
            ):
                collision_occurred = True
                #print(f"Collision occurred at position {path_cell} when trying to move from {self.position} to {path_cell}.")
                break
            else:
                position = np.asarray(path_cell, dtype=np.int64)

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
        observation_space = super()._get_observation_space()
        observation_space.spaces["surrounding_obstacles"] = spaces.MultiBinary(8)
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
        observation = super()._get_observation(action_result)
        observation["surrounding_obstacles"] = np.asarray(
            self._get_surrounding_obstacles(tuple(map(int, self.position))),
            dtype=np.bool_,
        )
        return observation

    def _get_surrounding_obstacles(
        self,
        position: Position,
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

                if self._is_occupied(candidate_position, include_agent=False):
                    occupied_directions[direction_index] = True
                    break

        return occupied_directions
