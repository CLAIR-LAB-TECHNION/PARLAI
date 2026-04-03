from dataclasses import replace
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


from utils import (
    BivariateNormalStruct,
    bivariate_normal,
    is_position_within_bounding_box,
    validate_bounds,
)


# Default parameters for the three pits in the caldera, along with their weights that control how deep they are.
DEFAULT_PIT_PARAMS = (
    BivariateNormalStruct(x=0.0, y=0.0, sigmax=0.16, sigmay=0.15, mux=0.2, muy=0.2),
    BivariateNormalStruct(x=0.0, y=0.0, sigmax=0.18, sigmay=0.2, mux=0.5, muy=0.7),
    BivariateNormalStruct(x=0.0, y=0.0, sigmax=0.17, sigmay=0.15, mux=0.8, muy=0.3),
)

# The weights for each pit, controlling how deep they are 
DEFAULT_PIT_WEIGHTS = (16000.0, 22000.0, 18000.0)

# A default value to use in observations when certain information is not applicable
DEFAULT_VALUE = -1
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

# Action names and mappings for better readability in the environment implementation.
MOVE_NORTH = "MOVE_NORTH"
MOVE_SOUTH = "MOVE_SOUTH"
MOVE_EAST = "MOVE_EAST"
MOVE_WEST = "MOVE_WEST"
SAMPLE = "SAMPLE"
ACTION_NAMES = (MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST, SAMPLE)
ACTION_TO_INDEX = {action_name: index for index, action_name in enumerate(ACTION_NAMES)}
MOVEMENT_ACTIONS = (MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST)


def default_reward_function(
    env,
    action,
    obs
) -> float:
    return -0.1

# Base environment class that implements the transition dynamics and shared helpers.
class BaseCalderaEnv(gym.Env):
    def __init__(
        self,
        # the map dimensions
        dim_x: int = 100, 
        dim_y: int = 100, 
        # the model for the geological features of the caldera, which determine the depth at each location
        pit_params: Sequence[BivariateNormalStruct] = DEFAULT_PIT_PARAMS, 
        pit_weights: Sequence[float] = DEFAULT_PIT_WEIGHTS,
        # the resolution of the sampling grid, which determines how the map is discretized for sampling.
        sampling_res: int = 10, # cell size in the sampling grid
        # the agent features
        initial_position: Tuple[int, int] = (0, 0),
        movement_size: int = 1,               
        initial_energy: int = 200,
        energy_per_move: int = 1,
        energy_per_sample: int = 1,
        other_vehicles: Optional[Sequence[Tuple[Tuple[int, int], int]]] = None,
        end_episode_on_collision: bool = False, # does the episode end if the agent collides with a vehicle, or is it just blocked from moving into that cell?
        reward_function: Callable[..., float] = default_reward_function,
    ):
        
        # Initialize the environment with the provided parameters, validating them as needed.        
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.max_position = np.array([self.dim_x, self.dim_y], dtype=np.int64)
        
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
        _, _, self.depth_map = self._generate_caldera_map()
 
        # Initilize the agent parameters and state variables
        self.initial_energy = initial_energy 
        self.energy_per_move = energy_per_move
        self.energy_per_sample = energy_per_sample
        if self.energy_per_move <= 0:
            raise ValueError("energy_per_move must be positive")
        if self.energy_per_sample <= 0:
            raise ValueError("energy_per_sample must be positive")
        self.energy = self.initial_energy
        self.movement_size = movement_size
        self.end_episode_on_collision = end_episode_on_collision
        
        
        # The positions of the agent and vehicles are in the dimensions of the map
        # whenever reset is applied, the agent returns to the initial position and energy level, and all sampled cells are cleared.
        if not validate_bounds(initial_position, self.max_position):
            raise ValueError(
                f"initial_position must be between (0, 0) and {tuple(self.max_position)}"
            )
        self.initial_position = np.array(list(map(int, initial_position)), dtype=np.int64)
        self.position = self.initial_position.copy()

        
        # The sampling process is based on a grid defined by the sampling resolution, 
        # and the agent can only sample at the bottom left corner of these grid cells.
        self.sampling_res = sampling_res  # defining the resultion of the sampling process    
        self.x_grid_coords = np.arange(0, self.dim_x + 1, self.sampling_res)
        self.y_grid_coords = np.arange(0, self.dim_y + 1, self.sampling_res)
        # to keep track of which grid cells have been sampled and their values        
        self.sampled_cells: Dict[Tuple[int, int], float] = {}
        self.max_value_observed = float("-inf")
        self.min_value_observed = float("inf")
        

        # setting the obstacles on the surface, which are represented as vehicles with a certain size.
        # The environment provides methods to add and remove vehicles and check for occupancy,
        # which are used to determine if the agent can move to a certain cell or if it is blocked by an obstacle.    
        self.surface_vehicles: Dict[Tuple[int, int], int] = {}
        self.add_vehicles(other_vehicles or [])
        self.agent_path = [tuple(map(int, self.position))]
                
        # initilize the reward and transition dynamics functions      
        if not callable(reward_function):
            raise ValueError("reward_function must be callable")
        self.reward_function = reward_function
 
        # The environment exposes a discrete action space for Gym compatibility,
        # but the implementation uses named actions internally.
        self.action_space = self._get_action_space()

        # The observation space is a dictionary containing the agent's position, remaining energy, whether the current cell has been sampled before, and the value of the current cell.
        self.observation_space = self._get_observation_space()

    def get_max_value_observed(self) -> float:
        return self.max_value_observed

    def get_min_value_observed(self) -> float:
        return self.min_value_observed
    
    # Method to add vehicles to the surface, validating each position and size.
    def add_vehicles(
        self,
        other_vehicles: Sequence[Tuple[Tuple[int, int], int]],
    ) -> None:
        for bottom_right_position, vehicle_size in other_vehicles:
            if not validate_bounds(bottom_right_position, self.max_position):
                raise ValueError(
                    f"Vehicle position must be within the map dimensions"
                )
            top_right_position = (
                int(bottom_right_position[0]) + vehicle_size,
                int(bottom_right_position[1]) + vehicle_size,
            )
            if not validate_bounds(top_right_position, self.max_position):
                raise ValueError(
                    f"Vehicle position must be within the map dimensions"
                )

            self.surface_vehicles[tuple(map(int, bottom_right_position))] = vehicle_size

    # Method to remove a vehicle from the surface by its bottom-right position.
    def remove_vehicle(
        self,
        bottom_right_position: Tuple[int, int],
    ) -> None:
        if not validate_bounds(bottom_right_position, self.max_position):
            raise ValueError("Vehicle position must be within the map dimensions")

        validated_position = tuple(map(int, bottom_right_position))
        if validated_position not in self.surface_vehicles:
            print(f"No vehicle found at position {validated_position}")
            return

        del self.surface_vehicles[validated_position]

    # Checks if the position is occupied by another vehicle,
    # with an option to include the agent's own position in the check.
    # Cells outside the map are treated as occupied.
    def is_occupied(
        self,
        cell: Tuple[int, int],
        include_agent: bool = False,
    ) -> bool:
        if not validate_bounds(cell, self.max_position):
            return True
        validated_cell = tuple(map(int, cell))

        if include_agent and validated_cell == tuple(map(int, self.position)):
            return True

        return any(
            is_position_within_bounding_box(validated_cell, vehicle_position, vehicle_size)
            for vehicle_position, vehicle_size in self.surface_vehicles.items()
        )

    # Method to get the locations of all vehicles on the surface, with an option to include the agent's own position. 
    # This is useful for visualization and for checking occupancy.    
    def get_vehicle_locations(
        self,
        include_agent: bool = False,
    ) -> Tuple[Tuple[int, int], ...]:
        agent_position = tuple(map(int, self.position))
        vehicle_locations = {
            vehicle_position
            for vehicle_position in self.surface_vehicles
            if include_agent or vehicle_position != agent_position
        }
        if include_agent:
            vehicle_locations.add(agent_position)

        return tuple(sorted(vehicle_locations))

    def get_invariant_information(self):
        return self.get_vehicle_locations()
    
    # reset the enviornment including the agent's state
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = self.initial_position.copy()
        self.energy = self.initial_energy
        self.sampled_cells = {}
        self.max_value_observed = float("-inf")
        self.min_value_observed = float("inf")
        self.agent_path = [tuple(map(int, self.position))]
        info = {}
        obs = self._get_observation(None)
        if not self.observation_space.contains(obs):
            raise ValueError(
                "Observation is outside observation_space. "
                f"obs={obs}, observation_space={self.observation_space}"
            )
        return obs, info

    # The main method that processes the agent's action, updates the environment state, 
    # computes the reward, and returns the new observation, reward, and done flags according to the Gym API. 
    # step returns the observation, reward, terminated, truncated, and info as expected by Gym environments.
    def step(self, action: Union[int, str]):

        # prepare the info dictionary for any information to return.
        info = {}    
        # Check if the input action is a valid string or integer and converts it to the corresponding string action name.
        normalized_action = self._normalize_action(action)
        # get the relevant energy cost
        action_energy_cost = self._get_action_energy_cost(normalized_action)
        # if the agent does not have enough energy to perform the action, 
        # the episode is terminated and a reward of 0 is given, 
        # with the info dictionary containing details about the energy depletion.
        if self.energy < action_energy_cost:
            obs = self._get_observation(None)
            if not self.observation_space.contains(obs):
                raise ValueError("Observation is outside observation_space. "f"obs={obs}, observation_space={self.observation_space}")
            info = {
                "reason": "insufficient_energy",
                "required_energy": int(action_energy_cost),
                "remaining_energy": int(self.energy),
            }
            #return observation, reward, terminated, truncated, and info
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
            # update energy based on the energy consumed by the action
            self.energy -= action_energy_cost

        # sample
        if effective_action == SAMPLE:
            action_result = self._get_sample()
            sampled_before, _ = action_result
            # update energy based on the energy consumed by the action
            # only if the cell has not been sampled before, we consume energy for the sampling action 
            # (note that the agent should be penalized for trying to sample the same cell repeatedly,
            #  but we don't want to consume energy for an action that has no effect)
            if not sampled_before:
                self.energy -= action_energy_cost
            else:
                info.update(
                    {"message": "The cell the agent was trying to sample was already sampled."}
                )

        # record the agent's path
        current_position = tuple(map(int, self.position))
        if current_position != self.agent_path[-1]:
            self.agent_path.append(current_position)


        # get the observation after performing the action and consuming energy
        obs = self._get_observation(action_result)
        if not self.observation_space.contains(obs):
            raise ValueError("Observation is outside observation_space. " f"obs={obs}, observation_space={self.observation_space}")

        # compute the reward based on the action taken and the resulting observation using the provided reward function.
        reward = float(self.reward_function(self, normalized_action, obs))
       
        # check if the episode is terminated due to collision or energy depletion
        terminated = (
            (collision_occurred and self.end_episode_on_collision)
            or self.energy == 0
        )
        truncated = False
     
        # return the observation, reward, terminated, truncated, and info as expected by Gym environments 
        return obs, reward, terminated, truncated, info
    
    # checking if the current cell has been sampled before and returning the appropriate value. 
    def _get_sample(self) -> Tuple[int, float]:
        raise NotImplementedError(
            "_get_sample must be implemented in to_implement.CalderaEnv"
        )
    
    # Visualization method to plot the depth map of the caldera,
    # the agent's path, the locations of vehicles, 
    # and optionally the centers of the pits and grid lines.
    def visualize(
        self,
        agent_path: Optional[Sequence[Tuple[int, int]]] = None,
        show_gaussian_centers: bool = False,
        show_grid_lines: bool = False,
        show_agent_path: bool = True,
    ):
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
        if show_gaussian_centers:
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

        other_vehicle_locations = self.get_vehicle_locations()
        if other_vehicle_locations:
            for vehicle_position in other_vehicle_locations:
                vehicle_size = self.surface_vehicles[vehicle_position]
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

    # Method to get the terrain value at a specific cell, validating the input position first.
    def _get_value(self, cell: Tuple[int, int]) -> float:
        if not validate_bounds(cell, self.max_position):
            raise ValueError(
                f"cell must be between (0, 0) and {tuple(self.max_position)}"
            )
        x_pos, y_pos = map(int, cell)
        return float(self.depth_map[y_pos, x_pos])

    # The function that computes the depth of the caldera at any given (x, y) coordinate by summing the contributions from all the pits, which are modeled as bivariate normal distributions. 
    # The depth is negative because we want deeper areas to have lower values.    
    def _caldera_sim_function(self, x, y):
        x = x / self.dim_x
        y = y / self.dim_y
        z = np.zeros_like(x, dtype=float)
        for weight, params in zip(self.pit_weights, self.pit_params):
            z += weight * bivariate_normal(replace(params, x=x, y=y))
        return -z

    # Generate the depth map for the entire caldera based on the simulation function. 
    # This is used for visualization and to look up depth values at specific locations.
    def _generate_caldera_map(self):
        x_coords = np.arange(0, self.dim_x + 1, dtype=np.int64)
        y_coords = np.arange(0, self.dim_y + 1, dtype=np.int64)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        z = self._caldera_sim_function(x_grid, y_grid)
        return x_grid, y_grid, z

    # return the structure of the action space, according to the Gym API.    
    def _get_action_space(self):
        return spaces.Discrete(n=len(ACTION_NAMES))

    # return the structure of the observation space, according to the Gym API.
    # This is intentionally implemented in to_implement.CalderaEnv.
    def _get_observation_space(self):
        raise NotImplementedError(
            "_get_observation_space must be implemented in to_implement.CalderaEnv"
        )

    # This is intentionally implemented in to_implement.CalderaEnv.
    def _get_observation(
        self,
        action_result: Optional[Union[np.ndarray, Tuple[int, float]]],
    ):
        raise NotImplementedError(
            "_get_observation must be implemented in to_implement.CalderaEnv"
        )

    def _get_action_energy_cost(self, action: str) -> int:
        if action in MOVEMENT_ACTIONS:
            return self.energy_per_move
        if action == SAMPLE:
            return self.energy_per_sample
        raise ValueError(f"Unsupported action for energy cost: {action}")

    # Helper method to normalize the action input, allowing for both string and integer representations of actions. 
    # This ensures that the environment can handle different formats of action inputs gracefully.
    def _normalize_action(self, action: Union[int, str]) -> str:
        if isinstance(action, str):
            normalized_action = action.upper()
            if normalized_action in ACTION_TO_INDEX:
                return normalized_action
            raise ValueError(f"Invalid action: {action}")

        if isinstance(action, (int, np.integer)) and self.action_space.contains(int(action)):
            return ACTION_NAMES[int(action)]

        raise ValueError(f"Invalid action: {action}")

    # This method can be overridden by subclasses to implement stochastic effects or other modifications to the action before it is executed.
    def _get_effective_action(self, action: str) -> str:
        return action

    # Compute the destination reached by applying a movement action once.
    def _get_proposed_destination(self, action: str) -> Tuple[int, int]:
        if action not in MOVEMENT_ACTIONS:
            raise ValueError(
                f"_get_proposed_destination only supports movement actions {MOVEMENT_ACTIONS}"
            )

        x_pos, y_pos = map(int, self.position)
        if action == MOVE_NORTH:
            y_pos += self.movement_size
        elif action == MOVE_SOUTH:
            y_pos -= self.movement_size
        elif action == MOVE_WEST:
            x_pos -= self.movement_size
        elif action == MOVE_EAST:
            x_pos += self.movement_size

        return int(x_pos), int(y_pos)

    # Performs the movement action by calculating the proposed new position based on the action and movement size,
    # checking for collisions with vehicles along the path, and updating the agent's position if the 
    def _perform_move(self, action: str) -> Tuple[np.ndarray, bool]:
        raise NotImplementedError(
            "_perform_move must be implemented in to_implement.CalderaEnv"
        )

