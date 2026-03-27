from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class BivariateNormalStruct:
    x: Any
    y: Any
    sigmax: float
    sigmay: float
    mux: float
    muy: float


# A terrain-shaping building block:
# The function computes the value of a 2D Gaussian (bivariate normal) density.
def _bivariate_normal(params: BivariateNormalStruct):
    # x_term measures how far x is from the center mux, scaled by sigmax
    x_term = ((params.x - params.mux) ** 2) / (2 * params.sigmax ** 2)
    # y_term does the same for y
    y_term = ((params.y - params.muy) ** 2) / (2 * params.sigmay ** 2)
    #normalization constant so the whole distribution integrates to 1
    norm = 1.0 / (2 * np.pi * params.sigmax * params.sigmay)
    #makes the value highest near (mux, muy) and decay smoothly away from it
    return norm * np.exp(-(x_term + y_term))


DEFAULT_PIT_PARAMS = (
    BivariateNormalStruct(x=0.0, y=0.0, sigmax=1.6, sigmay=1.5, mux=2.0, muy=2.0),
    BivariateNormalStruct(x=0.0, y=0.0, sigmax=1.8, sigmay=2.0, mux=5.0, muy=7.0),
    BivariateNormalStruct(x=0.0, y=0.0, sigmax=1.7, sigmay=1.5, mux=8.0, muy=3.0),
)

DEFAULT_PIT_WEIGHTS = (16000.0, 22000.0, 18000.0)

NORTH = "NORTH"
SOUTH = "SOUTH"
WEST = "WEST"
EAST = "EAST"
SAMPLE = "SAMPLE"

ACTION_NAMES = (NORTH, SOUTH, WEST, EAST, SAMPLE)
ACTION_TO_INDEX = {action_name: index for index, action_name in enumerate(ACTION_NAMES)}
MOVEMENT_ACTIONS = ACTION_NAMES[:-1]
WRONG_TURN_ACTION = {
    NORTH: EAST,
    EAST: SOUTH,
    SOUTH: WEST,
    WEST: NORTH,
}


def exploration_reward_function(
    env,
    action,
    obs
) -> float:
    if action != SAMPLE:
        return -0.1

    if obs["sampled_here"]:
        return -0.5
    
    return 1.0

class CalderaEnv(gym.Env):
    def __init__(
        self,
        pit_params: Sequence[BivariateNormalStruct] = DEFAULT_PIT_PARAMS,
        pit_weights: Sequence[float] = DEFAULT_PIT_WEIGHTS,
        dim_x: int = 100,
        dim_y: int = 100,
        delta: int = 10,
        initial_position: Tuple[int, int] = (0, 0),
        max_energy: int = 200,
        reward_function: Callable[..., float] = exploration_reward_function,
        vehicle_size: Optional[int] = None,
        stochastic_movement: bool = False,
        move_success_probabilities: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    ):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.delta = delta
        self.max_energy = max_energy
        self.reward_function = reward_function
        default_vehicle_size = max(1, int(self.dim_x / self.delta))
        resolved_vehicle_size = (
            default_vehicle_size
            if vehicle_size is None
            else vehicle_size
        )
        self.vehicle_size = self._validate_vehicle_size(resolved_vehicle_size)
        self.stochastic_movement = stochastic_movement
        self.move_success_probabilities = self._validate_success_probabilities(
            move_success_probabilities
        )

        self.pit_params = list(pit_params)
        self.pit_weights = list(pit_weights)

        if len(self.pit_params) != len(self.pit_weights):
            raise ValueError("pit_params and pit_weights must have the same length")

        for params in self.pit_params:
            if not isinstance(params, BivariateNormalStruct):
                raise ValueError(
                    "Each pit entry must be a BivariateNormalStruct with x, y, sigmax, sigmay, mux, and muy"
                )
        
        self.x_coords = np.arange(0, self.dim_x + 1, self.delta)
        self.y_coords = np.arange(0, self.dim_y + 1, self.delta)
        self.num_cols = len(self.x_coords)
        self.num_rows = len(self.y_coords)
        self.max_position = np.array([self.x_coords[-1], self.y_coords[-1]], dtype=np.int64)

        _, _, self.depth_map = self.generate_caldera_map()
        self.value_map = -self.depth_map
       
        # The environment still exposes a discrete action space for Gym compatibility,
        # but the implementation uses named actions internally.
        self.action_space = spaces.Discrete(5)

        # Position is stored as (x, y) in the same coordinate scale as the plotted grid.
        self.initial_position = self._validate_position(initial_position)
        self.position = self.initial_position.copy()
        self.energy = self.max_energy
        self.sampled_cells: Set[Tuple[int, int]] = set()
        self.surface_vehicles: Dict[Tuple[int, int], int] = {}
        self.agent_path = [tuple(map(int, self.position))]

        self.observation_space = spaces.Dict(
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

        if not callable(self.reward_function):
            raise ValueError("reward_function must be callable")

    def _validate_position(self, position: Tuple[int, int]) -> np.ndarray:
        x_pos, y_pos = map(int, position)

        if x_pos < 0 or x_pos > self.max_position[0] or x_pos % self.delta != 0:
            raise ValueError(
                f"x position must be on the grid between 0 and {self.max_position[0]}"
            )
        if y_pos < 0 or y_pos > self.max_position[1] or y_pos % self.delta != 0:
            raise ValueError(
                f"y position must be on the grid between 0 and {self.max_position[1]}"
            )

        return np.array([x_pos, y_pos], dtype=np.int64)

    def _position_to_indices(self, position: np.ndarray) -> Tuple[int, int]:
        x_pos, y_pos = map(int, position)
        return y_pos // self.delta, x_pos // self.delta

    def _validate_vehicle_size(self, vehicle_size: int) -> int:
        vehicle_size = int(vehicle_size)
        if vehicle_size <= 0:
            raise ValueError("vehicle size must be positive")
        return vehicle_size

    def get_depth_value(self, cell: Tuple[int, int]) -> float:
        validated_cell = self._validate_position(cell)
        row, col = self._position_to_indices(validated_cell)
        return float(self.depth_map[row, col])

    def add_vehicle(
        self,
        vehicle_position: Tuple[int, int],
        vehicle_size: Optional[int] = None,
    ) -> None:
        validated_position = tuple(map(int, self._validate_position(vehicle_position)))
        resolved_vehicle_size = self.vehicle_size if vehicle_size is None else vehicle_size
        self.surface_vehicles[validated_position] = self._validate_vehicle_size(resolved_vehicle_size)

    def _cell_is_inside_vehicle(
        self,
        cell: Tuple[int, int],
        vehicle_position: Tuple[int, int],
        vehicle_size: float,
    ) -> bool:
        cell_x, cell_y = cell
        vehicle_x, vehicle_y = vehicle_position
        half_size = vehicle_size / 2.0

        x_min = vehicle_x - half_size
        x_max = vehicle_x + half_size
        y_min = vehicle_y - half_size
        y_max = vehicle_y + half_size

        return x_min <= cell_x < x_max and y_min <= cell_y < y_max

    def is_cell_occupied_by_vehicle(
        self,
        cell: Tuple[int, int],
        include_agent: bool = False,
    ) -> bool:
        validated_cell = tuple(map(int, self._validate_position(cell)))

        if include_agent and self._cell_is_inside_vehicle(
            validated_cell,
            tuple(map(int, self.position)),
            self.vehicle_size,
        ):
            return True

        return any(
            self._cell_is_inside_vehicle(validated_cell, vehicle_position, vehicle_size)
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

    def _get_obs(self):
        sampled_here = int(tuple(self.position) in self.sampled_cells)
        return {
            "position": self.position.copy(),
            "energy": self.energy,
            "sampled_here": sampled_here,
        }

    def _record_agent_position(self) -> None:
        current_position = tuple(map(int, self.position))
        if current_position != self.agent_path[-1]:
            self.agent_path.append(current_position)

    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = self.initial_position.copy()
        self.energy = self.max_energy
        self.sampled_cells = set()
        self.agent_path = [tuple(map(int, self.position))]
        info = {}
        return self._get_obs(), info

    def _validate_success_probabilities(
        self,
        success_probabilities: Sequence[float],
    ) -> Tuple[float, float, float, float]:
        if len(success_probabilities) != 4:
            raise ValueError("success_probabilities must contain exactly 4 values")

        validated_probabilities = tuple(float(probability) for probability in success_probabilities)
        if any(probability < 0.0 or probability > 1.0 for probability in validated_probabilities):
            raise ValueError("each success probability must be between 0 and 1")

        return validated_probabilities

    def avoid_obstacle(self, next_position: np.ndarray) -> np.ndarray:
        proposed_position = tuple(map(int, next_position))
        if self.is_cell_occupied_by_vehicle(proposed_position, include_agent=False):
            print(f"Movement blocked by vehicle at {proposed_position}. Staying at {tuple(self.position)}")
            return self.position.copy()
        print(f"Movement successful to {proposed_position}")
        return next_position

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
        x_pos, y_pos = self.position

        if action == NORTH:
            y_pos = max(0, y_pos - self.delta)
        elif action == SOUTH:
            y_pos = min(int(self.max_position[1]), y_pos + self.delta)
        elif action == WEST:
            x_pos = max(0, x_pos - self.delta)
        elif action == EAST:
            x_pos = min(int(self.max_position[0]), x_pos + self.delta)

        next_position = np.array([x_pos, y_pos], dtype=np.int64)
        print(f"Attempting to move {action} to {tuple(next_position)}")
        return self.avoid_obstacle(next_position)

    def perform_stochastic_move(
        self,
        action: str,
        success_probabilities: Sequence[float],
    ) -> np.ndarray:
        if action not in MOVEMENT_ACTIONS:
            raise ValueError(
                "perform_stochastic_move only supports movement actions "
                f"{MOVEMENT_ACTIONS}"
            )

        validated_probabilities = self._validate_success_probabilities(success_probabilities)
        action_index = ACTION_TO_INDEX[action]

        if np.random.random() < validated_probabilities[action_index]:
            return self.perform_move(action)

        return self.perform_move(WRONG_TURN_ACTION[action])

    def step(self, action: Union[int, str]):
        normalized_action = self._normalize_action(action)

        if normalized_action != SAMPLE:
            if self.stochastic_movement:
                self.position = self.perform_stochastic_move(
                    normalized_action,
                    self.move_success_probabilities,
                )
            else:
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

        self._record_agent_position()
        self.energy -= 1

        terminated = self.energy <= 0
        truncated = False
        info = {}
        return self._get_obs(), reward, terminated, truncated, info

    def caldera_sim_function(self, x, y):
        x = x / (self.dim_x / 10.0)
        y = y / (self.dim_y / 10.0)
        z = np.zeros_like(x, dtype=float)
        for weight, params in zip(self.pit_weights, self.pit_params):
            z += weight * _bivariate_normal(replace(params, x=x, y=y))
        return -z


    def generate_caldera_map(self):
        x_grid, y_grid = np.meshgrid(self.x_coords, self.y_coords)
        z = self.caldera_sim_function(x_grid, y_grid)
        return x_grid, y_grid, z


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
    

def build_lawnmower_actions(env: CalderaEnv) -> List[str]:
    actions: List[str] = []

    for row_index in range(env.num_rows):
        horizontal_action = EAST if row_index % 2 == 0 else WEST

        for col_index in range(env.num_cols):
            actions.append(SAMPLE)
            if col_index < env.num_cols - 1:
                actions.append(horizontal_action)

        if row_index < env.num_rows - 1:
            actions.append(SOUTH)

    return actions


def compute_lawnmower_energy(env: CalderaEnv) -> int:
    samples = env.num_rows * env.num_cols
    horizontal_moves = env.num_rows * (env.num_cols - 1)
    vertical_moves = env.num_rows - 1
    return samples + horizontal_moves + vertical_moves


def main_lawnmower():
    dim_x = 100
    dim_y = 100
    delta = 10
    initial_position = (0, 0)

    probe_env = CalderaEnv(
        pit_params=DEFAULT_PIT_PARAMS,
        pit_weights=DEFAULT_PIT_WEIGHTS,
        dim_x=dim_x,
        dim_y=dim_y,
        delta=delta,
        initial_position=initial_position,
    )
    max_energy = compute_lawnmower_energy(probe_env)

    caldera_env = CalderaEnv(
        pit_params=DEFAULT_PIT_PARAMS,
        pit_weights=DEFAULT_PIT_WEIGHTS,
        dim_x=dim_x,
        dim_y=dim_y,
        delta=delta,
        initial_position=initial_position,
        max_energy=max_energy,
    )

    actions = build_lawnmower_actions(caldera_env)
    total_reward = 0.0
    counter =0
    for step_index, action in enumerate(actions, start=1):
        counter += 1
        if counter<100:            
            obs, reward, terminated, truncated, _ = caldera_env.step(action)
            total_reward += reward

        print(
            f"Step {step_index}: action={action}, position={tuple(obs['position'])}, "
            f"reward={reward:.2f}, energy={obs['energy']}"
        )

        if terminated or truncated:
            print("Episode finished before the full lawn-mowing pattern completed.")
            break

    print(f"Visited cells: {len(caldera_env.sampled_cells)}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final position: {tuple(caldera_env.position)}")

    fig, _ = caldera_env.visualize_caldera()
    plt.show()




def main():
    delta = 10
    # the x and y dimensions of our caldera
    dim_x = 1000
    dim_y = 1000
    initial_position = (600, 200)  

    caldera_env = CalderaEnv(
        pit_params=DEFAULT_PIT_PARAMS,
        pit_weights=DEFAULT_PIT_WEIGHTS,
        dim_x=dim_x,
        dim_y=dim_y,
        delta=delta,
        initial_position=initial_position,
    )

    current_depth = caldera_env.get_depth_value(tuple(caldera_env.position))
    caldera_env.add_vehicle((10, 10))
    caldera_env.add_vehicle((10, 20), vehicle_size=2 * delta)
    caldera_env.add_vehicle((10, 40), vehicle_size=4 * delta)

    print(f"Current depth: {current_depth}")
    print(f"Is (10, 10) occupied? {caldera_env.is_cell_occupied_by_vehicle((10, 10))}")
    print(f"Is (20, 20) occupied? {caldera_env.is_cell_occupied_by_vehicle((20, 20))}")
    print(
        "Is the agent cell occupied by another vehicle? "
        f"{caldera_env.is_cell_occupied_by_vehicle(tuple(caldera_env.position), include_agent=False)}"
    )
    print(
        f"Is the agent cell occupied when including the agent? "
        f"{caldera_env.is_cell_occupied_by_vehicle(tuple(caldera_env.position))}"
    )

    
    i = 0
    path = [caldera_env.position.copy()]
    while i < 1000:
        next_action = np.random.choice(ACTION_NAMES)
        obs, reward, terminated, truncated, info = caldera_env.step(next_action)

        current_depth = caldera_env.get_depth_value(tuple(caldera_env.position))
        path.append(caldera_env.position.copy())
        print(f"Step output: {(obs, reward, terminated, truncated, info)}")
        print(f"Agent position: {tuple(obs['position'])}")
        print(f"Current depth: {current_depth}")
        i += 1        
        if terminated or truncated:
            print("Episode finished.")
            break

    fig, _ = caldera_env.visualize_caldera(agent_path=path)
    plt.show()



def main_stochastic():
    delta = 10
    dim_x = 100
    dim_y = 100
    initial_position = (60, 20)

    caldera_env = CalderaEnv(
        pit_params=DEFAULT_PIT_PARAMS,
        pit_weights=DEFAULT_PIT_WEIGHTS,
        dim_x=dim_x,
        dim_y=dim_y,
        delta=delta,
        initial_position=initial_position,
        stochastic_movement=True,
        move_success_probabilities=(0.8, 0.8, 0.8, 0.8),
    )



if __name__ == "__main__":
    main_lawnmower()
