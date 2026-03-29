from dataclasses import replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


from utils import (
    BivariateNormalStruct,
    bivariate_normal,
    generate_path,
    is_position_within_bounding_box,
    position_to_indices,
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
WRONG_TURN_ACTION = {
    MOVE_NORTH: MOVE_EAST,
    MOVE_EAST: MOVE_SOUTH,
    MOVE_SOUTH: MOVE_WEST,
    MOVE_WEST: MOVE_NORTH,
}


# A simple reward function that encourages exploration by giving a positive reward for sampling new locations and a negative reward for sampling the same location or taking movement actions.
def reward_function_explore(
    env,
    action,
    obs
) -> float:
    reward = 0
    if action != SAMPLE:
        reward = -0.1

    if action == SAMPLE:
        if obs["sampled_before"]:
            reward = -0.5
        else:    
            reward = 1.0
    return reward    
    
def stochastic_effet_none(env,action):
    return action

def stochastic_effet_wrong_turn(env,action,success_probability=0.8):
    effective_action = action
    if action in MOVEMENT_ACTIONS:
        if np.random.random() >= success_probability:
            effective_action = WRONG_TURN_ACTION[action]
    return effective_action    

# The main environment class that simulates the caldera and the agent's interactions with it.
class CalderaEnv(gym.Env):
    def __init__(
        self,
        # the map dimensions
        dim_x: int = 100, 
        dim_y: int = 100, 
        # the model for the geolofical features of the caldera, which determine the depth at each location
        pit_params: Sequence[BivariateNormalStruct] = DEFAULT_PIT_PARAMS, 
        pit_weights: Sequence[float] = DEFAULT_PIT_WEIGHTS,
        # the resolution of the sampling grid, which determines how the map is discretized for sampling.
        sampling_res: int = 10, # cell size in the sampling grid
        # the agent features
        initial_position: Tuple[int, int] = (0, 0),
        movement_size: int = 1,               
        max_energy: int = 200,
        initial_energy: Optional[int] = None,
        energy_per_step: int = 1,
        # setting what the agent observes
        full_observability: bool = False,
        observability_distance: int = 1,
        # setting the transition dynamics and reward function
        reward_function: Callable[..., float] = reward_function_explore,
        stochastic_effet_function: Callable[..., float] = None,        
        stochastic: bool = False,
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
        self.max_energy = max_energy
        self.initial_energy = max_energy if initial_energy is None else initial_energy
        self.energy_per_step = energy_per_step
        self.energy = self.initial_energy
        self.movement_size = movement_size
        
        
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
        

        # setting the obstacles on the surface, which are represented as vehicles with a certain size. The environment provides methods to add vehicles and check for occupancy,
        #  which are used to determine if the agent can move to a certain cell or if it is blocked by an obstacle.    
        self.surface_vehicles: Dict[Tuple[int, int], int] = {}
        self.agent_path = [tuple(map(int, self.position))]
        
        # defining the agent's observability with regard to the other vehicles
        # under full_obseravability - the agent known where the other vehicles are located
        # under partial_observability - the agent only has information about the current cell
        # and can sense if there are obstacles within the surrponding 8 cardinal and intercardinal directions at a distance defined by observability_distance
        self.full_observability = full_observability
        if observability_distance < 0:
            raise ValueError("observability_distance must be non-negative")
        self.observability_distance = observability_distance
                
        # initilize the reward and transition dynamics functions      
        if not callable(reward_function):
            raise ValueError("reward_function must be callable")
        self.reward_function = reward_function        
        self.stochastic = stochastic
        if self.stochastic:
            self.stochastic_effet_function = stochastic_effet_function
 
        # The environment exposes a discrete action space for Gym compatibility,
        # but the implementation uses named actions internally.
        self.action_space = self._get_action_space()

        # The observation space includes the agent's current position, remaining energy, and whether the current cell has been sampled before.        
        self.observation_space = self._get_observation_space()
    
    # Method to add a vehicle to the surface at a specified position and size, 
    # validating the input position and ensuring its bottom left corner is on the grid.    
    def add_vehicle(
        self,
        bottom_right_position: Tuple[int, int],
        vehicle_size: Optional[int],
    ) -> None:
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

    # Checks if the position is occupied by another vehicle, 
    # with an option to include the agent's own position in the check.
    # This is used to determine if a movement action would be blocked by an obstacle.   
    def is_occupied(
        self,
        cell: Tuple[int, int],
        include_agent: bool = False,
    ) -> bool:
        if not validate_bounds(cell, self.max_position):
            raise ValueError(
                f"cell must be between (0, 0) and {tuple(self.max_position)}"
            )
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

    # check if the neigbouring positions around the agent are occupied by vehicles
    def get_surrounding_obstacles(
        self,
        position: Tuple[int, int],
    ) -> np.ndarray:
        
        # validate the input position    
        if not validate_bounds(position, self.max_position):
            raise ValueError(
                f"map_position must be between (0, 0) and {tuple(self.max_position)}"
            )

        # convert the position to integers 
        # and prepare a list to store the occupancy status of each direction
        x_pos, y_pos = map(int, position)
        occupied_directions = []

        # for each of the 8 cardinal and intercardinal directions, 
        # check if there is an obstacle within the observability distance 
        # and return a binary array indicating the presence of obstacles in those directions.
        for step_x, step_y in DIRECTION_STEPS.values():
            # get the path of cells in the current direction up to the observability distance
            path_cells = []
            for distance in range(1, self.observability_distance + 1):
                candidate_position = (
                    x_pos + (step_x * distance),
                    y_pos + (step_y * distance),
                )
                if not validate_bounds(candidate_position, self.max_position):
                    break
                path_cells.append(candidate_position)

            # check if any of the cells in the path are occupied by vehicles,
            #  and if so, mark that direction as having an obstacle
            # otherwise, mark it as clear.
            occupied_directions.append(
                any(
                    self.is_occupied(path_cell, include_agent=False)
                    for path_cell in path_cells
                )
            )

        return np.array(occupied_directions, dtype=bool)

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
        obs = self._get_observation(
            position=self.position,
            energy=self.energy,
            sampled_before=0,
            value=DEFAULT_VALUE,
            surrounding_obstacles=self.get_surrounding_obstacles(self.position),
        )
        return obs, info

    # The main method that processes the agent's action, updates the environment state, 
    # computes the reward, and returns the new observation, reward, and done flags according to the Gym API. 
    def step(self, action: Union[int, str]):

        # perform the perscribed action
 
        # Check if the input action is a valid string or integer and converts it to the corresponding string action name.
        normalized_action = self._normalize_action(action)

        # if the environment is stochastic, apply the stochastic effect function
        if self.stochastic and self.stochastic_effet_function is not None:
            action = self.stochastic_effet_function(self, action)

        # perform a move action
        if normalized_action in MOVEMENT_ACTIONS:
            self.position = self._perform_move(normalized_action)

        # perform a sample action
        if normalized_action == SAMPLE:
            sampled_before, sampled_value = self.perform_sample()
              
        # get the observation after performing the action
        obs = self._get_observation(
            position=self.position,
            energy=self.energy,
            sampled_before=(
                sampled_before
                if normalized_action == SAMPLE
                else int(
                    position_to_indices(self.position, self.sampling_res)
                    in self.sampled_cells
                )
            ),
            value=(
                sampled_value
                if normalized_action == SAMPLE
                else DEFAULT_VALUE
            ),
            surrounding_obstacles=self.get_surrounding_obstacles(self.position),
        )
        assert self.observation_space.contains(obs)


        # compute the reward based on the action taken and the resulting observation using the provided reward function.         
        reward = float(self.reward_function(self, normalized_action, obs))
        assert isinstance(reward, (float, int))


        # record the agent's path
        current_position = tuple(map(int, self.position))
        if current_position != self.agent_path[-1]:
            self.agent_path.append(current_position)

        # update energy usage    
        self.energy -= self.energy_per_step
        # check if the episode is terminated due to energy depletion or other conditions, 
        # and prepare the info dictionary for any additional information to return.
        terminated = self.energy <= 0
        truncated = False
        info = {}


      
        # return the observation, reward, terminated, truncated, and info as expected by Gym environments 
        return obs, reward, terminated, truncated, info
    
    # The function that performs the sampling action, 
    # checking if the current cell has been sampled before and returning the appropriate value. 
    # It also updates the sampled_cells dictionary to keep track of which cells have been sampled and their values.
    def perform_sample(self) -> Tuple[int, float]:
        grid_cell = position_to_indices(self.position, self.sampling_res)
        sampled_before = int(grid_cell in self.sampled_cells)
        if not sampled_before:
            sampled_value = float(self._get_value(grid_cell))
            # the value of the grid cell is stored 
            self.sampled_cells[grid_cell] = sampled_value
        else:
            sampled_value = float(self.sampled_cells[grid_cell])

        self.max_value_observed = max(self.max_value_observed, sampled_value)
        self.min_value_observed = min(self.min_value_observed, sampled_value)

        return sampled_before, sampled_value
    def visualize_caldera(
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
    def _get_observation_space(self):
        observation_space = {
            "position": spaces.Box(
                low=np.array([0, 0], dtype=np.int64),
                high=self.max_position.copy(),
                shape=(2,),
                dtype=np.int64,
            ),
            "energy": spaces.Discrete(self.max_energy + 1),
            "sampled_before": spaces.Discrete(2),
            "value": spaces.Box(
                low=np.array(-np.inf, dtype=np.float64),
                high=np.array(np.inf, dtype=np.float64),
                shape=(),
                dtype=np.float64,
            ),
            "surrounding_obstacles": spaces.MultiBinary(8),
        }    

        return spaces.Dict(observation_space)

    def _get_observation(
        self,
        position: Sequence[int],
        energy: int,
        sampled_before: int,
        value: float,
        surrounding_obstacles: Sequence[bool],
    ):
        observation = {
            "position": np.asarray(position, dtype=np.int64),
            "energy": int(energy),
            "sampled_before": int(sampled_before),
            "value": np.float64(value),
            "surrounding_obstacles": np.asarray(surrounding_obstacles, dtype=np.bool_),
        }     

        return observation

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

    def _perform_move(self, action: str) -> np.ndarray:
        
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
            return self.position.copy()

        path_positions = generate_path(tuple(map(int, self.position)), proposed_position)

        for path_cell in path_positions:
            if self.is_occupied(path_cell, include_agent=False):
                print(f"Movement blocked by vehicle at {path_cell}. Staying at {tuple(self.position)}")
                return self.position.copy()

        next_position = np.array(proposed_position, dtype=np.int64)

        return next_position
    
