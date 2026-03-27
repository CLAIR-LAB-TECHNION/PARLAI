from dataclasses import replace
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    BivariateNormalStruct,
    _bivariate_normal,
    is_cell_within_bounding_box,
    valiedate_bounds,
)


# Default parameters for the three pits in the caldera, along with their weights that control how deep they are.
DEFAULT_PIT_PARAMS = (
    BivariateNormalStruct(x=0.0, y=0.0, sigmax=0.16, sigmay=0.15, mux=0.2, muy=0.2),
    BivariateNormalStruct(x=0.0, y=0.0, sigmax=0.18, sigmay=0.2, mux=0.5, muy=0.7),
    BivariateNormalStruct(x=0.0, y=0.0, sigmax=0.17, sigmay=0.15, mux=0.8, muy=0.3),
)

# The weights for each pit, controlling how deep they are relative to each other.
DEFAULT_PIT_WEIGHTS = (16000.0, 22000.0, 18000.0)

# Action names and mappings for better readability in the environment implementation.
NORTH = "NORTH"
SOUTH = "SOUTH"
EAST = "EAST"
WEST = "WEST"
SAMPLE = "SAMPLE"
ACTION_NAMES = (NORTH, SOUTH, EAST, WEST, SAMPLE)
ACTION_TO_INDEX = {action_name: index for index, action_name in enumerate(ACTION_NAMES)}
MOVEMENT_ACTIONS = (NORTH, SOUTH, EAST, WEST)
WRONG_TURN_ACTION = {
    NORTH: EAST,
    EAST: SOUTH,
    SOUTH: WEST,
    WEST: NORTH,
}

# A simple reward function that encourages exploration by giving a positive reward for sampling new locations and a negative reward for sampling the same location or taking movement actions.
def reward_function_exploration(
    env,
    action,
    obs
) -> float:
    if action != SAMPLE:
        return -0.1

    if action == SAMPLE and obs["sampled_here"]:
        return -0.5
    else:    
        return 1.0

# The main environment class that simulates the caldera and the agent's interactions with it.
class CalderaEnv(gym.Env):
    def __init__(
        self,
        dim_x: int = 100,
        dim_y: int = 100,
        delta: int = 10, # cell size in the grid
        pit_params: Sequence[BivariateNormalStruct] = DEFAULT_PIT_PARAMS,
        pit_weights: Sequence[float] = DEFAULT_PIT_WEIGHTS,
        initial_position: Tuple[int, int] = (0, 0),
        vehicle_size: Optional[int] = None, # default to 1 cell, can be larger to represent bigger vehicles
        max_energy: int = 200,
        initial_energy: Optional[int] = None,
        reward_function: Callable[..., float] = reward_function_exploration,
        stochastic: bool = False,
        success_probabilities: Optional[Sequence[float]] = None,
    ):
        
        # Initialize the environment with the provided parameters, validating them as needed.        
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.delta = delta

        # set the coordinates for the grid based on the dimensions and delta
        self.x_coords = np.arange(0, self.dim_x + 1, self.delta)
        self.y_coords = np.arange(0, self.dim_y + 1, self.delta)
        self.num_cols = len(self.x_coords)
        self.num_rows = len(self.y_coords)
        self.max_position = np.array([self.x_coords[-1], self.y_coords[-1]], dtype=np.int64)
        
        # Validae the pit parameters and weights
        self.pit_params = list(pit_params)
        self.pit_weights = list(pit_weights)
        if len(self.pit_params) != len(self.pit_weights):
            raise ValueError("pit_params and pit_weights must have the same length")
        for params in self.pit_params:
            if not isinstance(params, BivariateNormalStruct):
                raise ValueError(
                    "Each pit entry must be a BivariateNormalStruct with x, y, sigmax, sigmay, mux, and muy"
                )
     
        # Generate the depth map for the caldera using the provided pit parameters and weights.
        _, _, self.depth_map = self.generate_caldera_map()
        self.value_map = -self.depth_map
 
        # initilize the agent parameters and state variables
        self.max_energy = max_energy
        self.initial_energy = max_energy if initial_energy is None else initial_energy
        self.vehicle_size = (
            max(1, int(self.dim_x / self.delta))
            if vehicle_size is None
            else vehicle_size
        )
        # Position is stored as (x, y) in the same coordinate scale as the plotted grid.
        # Whenever reset is applied, the agent returns to the initial position and energy level, and all sampled cells are cleared.
        self.initial_position = valiedate_bounds(
            initial_position,
            self.max_position,
            self.delta,
        )
        self.position = self.initial_position.copy()
        self.energy = self.initial_energy
        self.sampled_cells: Set[Tuple[int, int]] = set()
        self.surface_vehicles: Dict[Tuple[int, int], int] = {}
        self.agent_path = [tuple(map(int, self.position))]
        
        # initilize the reward and transition dynamics functions      
        if not callable(reward_function):
            raise ValueError("reward_function must be callable")
        self.reward_function = reward_function        
        self.stochastic = stochastic
        if self.stochastic:
            self.success_probabilities = (
                tuple(1.0 for _ in MOVEMENT_ACTIONS)
                if success_probabilities is None
                else success_probabilities
            )
 
        # The environment exposes a discrete action space for Gym compatibility,
        # but the implementation uses named actions internally.
        self.action_space = self._get_action_space()

        # The observation space includes the agent's current position, remaining energy, and whether the current cell has been sampled before.        
        self.observation_space = self._get_observation_space()
    
    # The function that computes the depth of the caldera at any given (x, y) coordinate by summing the contributions from all the pits, which are modeled as bivariate normal distributions. 
    # The depth is negative because we want deeper areas to have lower values.    
    def caldera_sim_function(self, x, y):
        x = x / self.dim_x
        y = y / self.dim_y
        z = np.zeros_like(x, dtype=float)
        for weight, params in zip(self.pit_weights, self.pit_params):
            z += weight * _bivariate_normal(replace(params, x=x, y=y))
        return -z

    # Generate the depth map for the entire caldera based on the simulation function. 
    # This is used for visualization and to look up depth values at specific locations.
    def generate_caldera_map(self):
        x_grid, y_grid = np.meshgrid(self.x_coords, self.y_coords)
        z = self.caldera_sim_function(x_grid, y_grid)
        return x_grid, y_grid, z
    # Helper method to convert a position in the grid to the corresponding indices in the depth map.    
    def _position_to_indices(self, position: np.ndarray) -> Tuple[int, int]:
        x_pos, y_pos = map(int, position)
        return y_pos // self.delta, x_pos // self.delta
  
    # Method to get the depth value at a specific cell, validating the input position first.
    def get_depth_value(self, cell: Tuple[int, int]) -> float:
        validated_cell = valiedate_bounds(cell, self.max_position, self.delta)
        row, col = self._position_to_indices(validated_cell)
        return float(self.depth_map[row, col])
    
    # Method to add a vehicle to the surface at a specified position and size, validating the input position and ensuring it is on the grid.    
    def add_vehicle(
        self,
        vehicle_position: Tuple[int, int],
        vehicle_size: Optional[int] = None,
    ) -> None:
        validated_position = tuple(
            map(int, valiedate_bounds(vehicle_position, self.max_position, self.delta))
        )
        resolved_vehicle_size = self.vehicle_size if vehicle_size is None else vehicle_size
        self.surface_vehicles[validated_position] = resolved_vehicle_size

    # checks if the cell is occupied by another vehicle, with an option to include the agent's own position in the check. This is used to determine if a movement action would be blocked by an obstacle.   
    def is_occupied(
        self,
        cell: Tuple[int, int],
        include_agent: bool = False,
    ) -> bool:
        
        validated_cell = tuple(
            map(int, valiedate_bounds(cell, self.max_position, self.delta))
        )

        if include_agent and is_cell_within_bounding_box(
            validated_cell,
            tuple(map(int, self.position)),
            self.vehicle_size,
        ):
            return True

        return any(
            is_cell_within_bounding_box(validated_cell, vehicle_position, vehicle_size)
            for vehicle_position, vehicle_size in self.surface_vehicles.items()
        )

    def get_other_vehicle_locations(self) -> Tuple[Tuple[int, int], ...]:
        agent_position = tuple(map(int, self.position))
        return tuple(
            sorted(
                vehicle_position
                for vehicle_position in self.surface_vehicles
                if vehicle_position != agent_position
            )
        )

    # return the structure of the action space, according to the Gym API.    
    def _get_action_space(self):
        return spaces.Discrete(n=len(ACTION_NAMES))
    # return the structure of the observation space, according to the Gym API.
    def _get_observation_space(self):
        return spaces.Dict(
            {
                "position": spaces.Box(
                    low=np.array([0, 0], dtype=np.int64),
                    high=self.max_position.copy(),
                    shape=(2,),
                    dtype=np.int64,
                ),
                "energy": spaces.Discrete(self.max_energy + 1),
                "sampled_here": spaces.Discrete(2),
            }
        )

    # get the current observation
    def _get_obs(self):
        sampled_here = int(tuple(self.position) in self.sampled_cells)
        return {
            "position": self.position.copy(),
            "energy": self.energy,
            "sampled_here": sampled_here,
        }

    # reset the enviornment including the agent's state
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = self.initial_position.copy()
        self.energy = self.initial_energy
        self.sampled_cells = set()
        self.agent_path = [tuple(map(int, self.position))]
        info = {}
        return self._get_obs(), info
    
    def _normalize_action(self, action: Union[int, str]) -> str:
        if isinstance(action, str):
            normalized_action = action.upper()
            if normalized_action in ACTION_TO_INDEX:
                return normalized_action
            raise ValueError(f"Invalid action: {action}")

        if isinstance(action, (int, np.integer)) and self.action_space.contains(int(action)):
            return ACTION_NAMES[int(action)]

        raise ValueError(f"Invalid action: {action}")

    def perform_move(self, action: str) -> np.ndarray:
        if action not in MOVEMENT_ACTIONS:
            raise ValueError(f"perform_move only supports movement actions {MOVEMENT_ACTIONS}")

        effective_action = action
        if self.stochastic:           
            action_index = ACTION_TO_INDEX[action]
            if np.random.random() >= self.success_probabilities[action_index]:
                effective_action = WRONG_TURN_ACTION[action]

        x_pos, y_pos = self.position

        if effective_action == NORTH:
            y_pos = max(0, y_pos - self.delta)
        elif effective_action == SOUTH:
            y_pos = min(int(self.max_position[1]), y_pos + self.delta)
        elif effective_action == WEST:
            x_pos = max(0, x_pos - self.delta)
        elif effective_action == EAST:
            x_pos = min(int(self.max_position[0]), x_pos + self.delta)

        next_position = np.array([x_pos, y_pos], dtype=np.int64)
        proposed_position = tuple(map(int, next_position))
        print(f"Attempting to move {effective_action} to {proposed_position}")

        if self.is_occupied(proposed_position, include_agent=False):
            print(f"Movement blocked by vehicle at {proposed_position}. Staying at {tuple(self.position)}")
            return self.position.copy()

        print(f"Movement successful to {proposed_position}")
        return next_position

    def step(self, action: Union[int, str]):
        normalized_action = self._normalize_action(action)

        if normalized_action in MOVEMENT_ACTIONS:
            self.position = self.perform_move(normalized_action)

        is_new_sample = False
        if normalized_action == SAMPLE:
            print(f"Sampling at position {tuple(self.position)}")
            cell = tuple(self.position)
            is_new_sample = cell not in self.sampled_cells
            if is_new_sample:
                self.sampled_cells.add(cell)

        reward = float(
            self.reward_function(
                self,
                normalized_action,
                self._get_obs()
            )
        )

        # record the agent's path
        current_position = tuple(map(int, self.position))
        if current_position != self.agent_path[-1]:
            self.agent_path.append(current_position)

        # update energy usage    
        self.energy -= 1

        terminated = self.energy <= 0
        truncated = False
        info = {}

        obs = self._get_obs()

        assert self.observation_space.contains(obs)
        assert isinstance(reward, (float, int))

        # return the observation, reward, terminated, truncated, and info as expected by Gym environments 
        return obs, reward, terminated, truncated, info

   
    def visualize_caldera(
        self,
        agent_path: Optional[Sequence[Tuple[int, int]]] = None,
    ):
        x, y, z = self.generate_caldera_map()
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(x, y, z, levels=20, cmap="viridis")

        path_to_plot = self.agent_path if agent_path is None else agent_path

        other_vehicle_locations = self.get_other_vehicle_locations()
        if other_vehicle_locations:
            vehicle_x, vehicle_y = zip(*other_vehicle_locations)
            vehicle_sizes = [self.surface_vehicles[vehicle_position] for vehicle_position in other_vehicle_locations]
            ax.scatter(
                vehicle_x,
                vehicle_y,
                marker="s",
                s=vehicle_sizes,
                c="red",
                edgecolors="white",
                linewidths=1.5,
            )

        row, col = self._position_to_indices(self.position)
        if 0 <= row < z.shape[0] and 0 <= col < z.shape[1]:
            vehicle_value = z[row, col]
            if not np.isneginf(vehicle_value):
                ax.scatter(
                    self.position[0],
                    self.position[1],
                    marker="s",
                    s=self.vehicle_size,
                    c="black",
                    linewidths=2,
                )

        if path_to_plot:
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
    
