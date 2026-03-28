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
    is_cell_within_bounding_box,
    validate_bounds,
    position_to_indices
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
    print(f"Reward computed: {reward} for action: {action} with observation: {obs}")    
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
        dim_x: int = 100,
        dim_y: int = 100,
        delta: int = 10, # cell size in the grid
        movement_size: Optional[int] = None,
        pit_params: Sequence[BivariateNormalStruct] = DEFAULT_PIT_PARAMS,
        pit_weights: Sequence[float] = DEFAULT_PIT_WEIGHTS,
        initial_position: Tuple[int, int] = (0, 0),
        max_energy: int = 200,
        initial_energy: Optional[int] = None,
        energy_per_step: int = 1,
        reward_function: Callable[..., float] = reward_function_explore,
        stochastic_effet_function: Callable[..., float] = None,        
        stochastic: bool = False,
    ):
        
        # Initialize the environment with the provided parameters, validating them as needed.        
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.delta = delta
        self.x_grid_coords = np.arange(0, self.dim_x + 1, self.delta)
        self.y_grid_coords = np.arange(0, self.dim_y + 1, self.delta)
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
        self.value_map = -self.depth_map
 
        # initilize the agent parameters and state variables
        self.max_energy = max_energy
        self.initial_energy = max_energy if initial_energy is None else initial_energy
        self.energy_per_step = energy_per_step
        self.movement_size = self.delta if movement_size is None else movement_size
        
        # position is stored as (x, y) in the same coordinate scale as the plotted grid.
        # Whenever reset is applied, the agent returns to the initial position and energy level, and all sampled cells are cleared.
        self.initial_position = validate_bounds(
            initial_position,
            self.max_position,
        )
        # the positions of the agent and vehicles are in the dimensions of the caldera, not the grid indices,
        # so they can be anywhere within the bounds defined by dim_x and dim_y,
        # as long as they're validated by validate_bounds.
        self.position = self.initial_position.copy()
        self.energy = self.initial_energy
        self.sampled_cells: Dict[Tuple[int, int], float] = {}
        self.surface_vehicles: Dict[Tuple[int, int], int] = {}
        self.agent_path = [tuple(map(int, self.position))]
                
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
    
    # Method to get the terrain value at a specific cell, validating the input position first.
    def _get_value(self, cell: Tuple[int, int]) -> float:
        validated_cell = validate_bounds(cell, self.max_position)
        x_pos, y_pos = map(int, validated_cell)
        return float(self.depth_map[y_pos, x_pos])
    
    # Method to add a vehicle to the surface at a specified position and size, validating the input position and ensuring it is on the grid.    
    def add_vehicle(
        self,
        vehicle_position: Tuple[int, int],
        vehicle_size: Optional[int] = None,
    ) -> None:
        validated_position = tuple(
            map(int, validate_bounds(vehicle_position, self.max_position))
        )
        resolved_vehicle_size = self.delta if vehicle_size is None else vehicle_size
        self.surface_vehicles[validated_position] = resolved_vehicle_size

    # checks if the cell is occupied by another vehicle, with an option to include the agent's own position in the check. This is used to determine if a movement action would be blocked by an obstacle.   
    def is_occupied(
        self,
        cell: Tuple[int, int],
        include_agent: bool = False,
    ) -> bool:
        
        validated_cell = tuple(
            map(int, validate_bounds(cell, self.max_position))
        )

        if include_agent and validated_cell == tuple(map(int, self.position)):
            return True

        return any(
            is_cell_within_bounding_box(validated_cell, vehicle_position, vehicle_size)
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

    # reset the enviornment including the agent's state
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = self.initial_position.copy()
        self.energy = self.initial_energy
        self.sampled_cells = {}
        self.agent_path = [tuple(map(int, self.position))]
        info = {}
        obs = self._create_observation(
            position=self.position,
            energy=self.energy,
            sampled_before=0,
            value=DEFAULT_VALUE,
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
        obs = self._create_observation(
            position=self.position,
            energy=self.energy,
            sampled_before=(
                sampled_before
                if normalized_action == SAMPLE
                else int(tuple(self.position) in self.sampled_cells)
            ),
            value=(
                sampled_value
                if normalized_action == SAMPLE
                else DEFAULT_VALUE
            ),
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
        return spaces.Dict(
            {
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
            }
        )

    def _create_observation(
        self,
        position: np.ndarray,
        energy: int,
        sampled_before: int,
        value: float,
    ):
        return {
            "position": position.copy(),
            "energy": energy,
            "sampled_before": sampled_before,
            "value": np.float64(value),
        }
    
    # The function that performs the sampling action, 
    # checking if the current cell has been sampled before and returning the appropriate value. 
    # It also updates the sampled_cells dictionary to keep track of which cells have been sampled and their values.
    def perform_sample(self) -> Tuple[int, np.float64]:
        grid_cell = position_to_indices(self.position,self.delta)
        sampled_before = int(grid_cell in self.sampled_cells)
        if not sampled_before:
            sampled_value = np.float64(self._get_value(grid_cell))
            # the value of the grid cell is stored 
            self.sampled_cells[grid_cell] = float(sampled_value)
        else:
            sampled_value = np.float64(self.sampled_cells[grid_cell])

        return sampled_before, sampled_value

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

        if action == NORTH:
            y_pos -= self.movement_size
        elif action == SOUTH:
            y_pos += self.movement_size
        elif action == WEST:
            x_pos -= self.movement_size
        elif action == EAST:
            x_pos += self.movement_size

        proposed_position = (int(x_pos), int(y_pos))
        print(f"Attempting to move {action} to {proposed_position}")

        if not (
            0 <= proposed_position[0] <= int(self.max_position[0])
            and 0 <= proposed_position[1] <= int(self.max_position[1])
        ):
            print(f"Movement out of bounds for {proposed_position}. Staying at {tuple(self.position)}")
            return self.position.copy()

        current_x, current_y = map(int, self.position)
        if current_x != proposed_position[0]:
            step_direction = 1 if proposed_position[0] > current_x else -1
            path_cells = (
                (x_pos, current_y)
                for x_pos in range(current_x + step_direction, proposed_position[0] + step_direction, step_direction)
            )
        else:
            step_direction = 1 if proposed_position[1] > current_y else -1
            path_cells = (
                (current_x, y_pos)
                for y_pos in range(current_y + step_direction, proposed_position[1] + step_direction, step_direction)
            )

        for path_cell in path_cells:
            if self.is_occupied(path_cell, include_agent=False):
                print(f"Movement blocked by vehicle at {path_cell}. Staying at {tuple(self.position)}")
                return self.position.copy()

        next_position = np.array(proposed_position, dtype=np.int64)

        print(f"Movement successful to {proposed_position}")
        return next_position
    
