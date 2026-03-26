from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple

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


def default_reward_function(
    env,
    action: int,
    position: Tuple[int, int],
    is_new_sample: bool,
) -> float:
    if action != 4:
        return -0.1

    if not is_new_sample:
        return -0.5

    row, col = env._position_to_indices(position)
    return float(env.value_map[row, col])


class CalderaEnv(gym.Env):
    def __init__(
        self,
        pit_params: Sequence[BivariateNormalStruct] = DEFAULT_PIT_PARAMS,
        pit_weights: Sequence[float] = DEFAULT_PIT_WEIGHTS,
        dim_x: int = 100,
        dim_y: int = 100,
        delta: int = 10,
        initial_position: Tuple[int, int] = (0, 0),
        max_energy: int = 20,
        reward_function: Callable[..., float] = default_reward_function,
        vehicle_marker_size: int = 320,
        stochastic_movement: bool = False,
        move_success_probabilities: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    ):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.delta = delta
        self.max_energy = max_energy
        self.reward_function = reward_function
        self.vehicle_marker_size = vehicle_marker_size
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
       
        # the vehicle can move in 4 directions or sample the current cell
        # action 0: move up, 1: move down, 2: move left, 3: move right, 4: sample
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
        if self.vehicle_marker_size <= 0:
            raise ValueError("vehicle_marker_size must be positive")

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

    def _validate_vehicle_marker_size(self, marker_size: int) -> int:
        marker_size = int(marker_size)
        if marker_size <= 0:
            raise ValueError("vehicle marker size must be positive")
        return marker_size

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
        marker_size = self.vehicle_marker_size if vehicle_size is None else vehicle_size
        self.surface_vehicles[validated_position] = self._validate_vehicle_marker_size(marker_size)

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
        include_agent: bool = True,
    ) -> bool:
        validated_cell = tuple(map(int, self._validate_position(cell)))

        if include_agent and self._cell_is_inside_vehicle(
            validated_cell,
            tuple(map(int, self.position)),
            self.vehicle_marker_size,
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
            return self.position.copy()
        return next_position

    def perform_move(self, action: int) -> np.ndarray:
        x_pos, y_pos = self.position

        if action == 0:
            y_pos = max(0, y_pos - self.delta)
        elif action == 1:
            y_pos = min(int(self.max_position[1]), y_pos + self.delta)
        elif action == 2:
            x_pos = max(0, x_pos - self.delta)
        elif action == 3:
            x_pos = min(int(self.max_position[0]), x_pos + self.delta)

        next_position = np.array([x_pos, y_pos], dtype=np.int64)
        return self.avoid_obstacle(next_position)

    def perform_stochastic_move(
        self,
        action: int,
        success_probabilities: Sequence[float],
    ) -> np.ndarray:
        if action not in {0, 1, 2, 3}:
            raise ValueError("perform_stochastic_move only supports movement actions 0-3")

        validated_probabilities = self._validate_success_probabilities(success_probabilities)
        right_turn_action = {
            0: 3,
            3: 1,
            1: 2,
            2: 0,
        }

        if np.random.random() < validated_probabilities[action]:
            return self.perform_move(action)

        return self.perform_move(right_turn_action[action])

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        if action != 4:
            if self.stochastic_movement:
                self.position = self.perform_stochastic_move(
                    action,
                    self.move_success_probabilities,
                )
            else:
                self.position = self.perform_move(action)

        is_new_sample = False
        if action == 4:
            cell = tuple(self.position)
            is_new_sample = cell not in self.sampled_cells
            if is_new_sample:
                self.sampled_cells.add(cell)

        reward = float(
            self.reward_function(
                self,
                action,
                tuple(self.position),
                is_new_sample,
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
                    s=self.vehicle_marker_size,
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
    



def main():
    delta = 10
    # the x and y dimensions of our caldera
    dim_x = 100
    dim_y = 100
    initial_position = (60, 20)  

    caldera_env = CalderaEnv(
        pit_params=DEFAULT_PIT_PARAMS,
        pit_weights=DEFAULT_PIT_WEIGHTS,
        dim_x=dim_x,
        dim_y=dim_y,
        delta=delta,
        initial_position=initial_position
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
    while i < 100:
        next_action = np.random.randint(0, 5)
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
    main()
