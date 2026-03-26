from dataclasses import dataclass, replace
from typing import Any, Sequence, Set, Tuple

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


class CalderaEnv(gym.Env):
    def __init__(
        self,
        pit_params: Sequence[BivariateNormalStruct] = DEFAULT_PIT_PARAMS,
        pit_weights: Sequence[float] = DEFAULT_PIT_WEIGHTS,
        dim_x: int = 100,
        dim_y: int = 100,
        delta: int = 10,
        max_energy: int = 20,
    ):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.delta = delta
        self.max_energy = max_energy

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

        _, _, self.depth_map = self.generate_caldera_map()
        self.value_map = -self.depth_map

        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(
                    low=np.array([0, 0], dtype=np.int64),
                    high=np.array([self.num_rows - 1, self.num_cols - 1], dtype=np.int64),
                    shape=(2,),
                    dtype=np.int64,
                ),
                "energy": spaces.Discrete(self.max_energy + 1),
                "sampled_here": spaces.Discrete(2),
            }
        )
        self.action_space = spaces.Discrete(5)

        self.position = np.array([0, 0], dtype=np.int64)
        self.energy = self.max_energy
        self.sampled_cells: Set[Tuple[int, int]] = set()

    def _get_obs(self):
        sampled_here = int(tuple(self.position) in self.sampled_cells)
        return {
            "position": self.position.copy(),
            "energy": self.energy,
            "sampled_here": sampled_here,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = np.array([0, 0], dtype=np.int64)
        self.energy = self.max_energy
        self.sampled_cells = set()
        return self._get_obs(), info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        reward = -0.1
        row, col = self.position

        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(self.num_rows - 1, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        elif action == 3:
            col = min(self.num_cols - 1, col + 1)
        elif action == 4:
            cell = tuple(self.position)
            if cell not in self.sampled_cells:
                self.sampled_cells.add(cell)
                reward = float(self.value_map[cell])
            else:
                reward = -0.5

        self.position = np.array([row, col], dtype=np.int64)
        self.energy -= 1

        terminated = self.energy <= 0
        truncated = False
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


    def visualize_caldera(self):
        x, y, z = self.generate_caldera_map()
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(x, y, z, levels=20, cmap="viridis")
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

    caldera_env = CalderaEnv(
        pit_params=DEFAULT_PIT_PARAMS,
        pit_weights=DEFAULT_PIT_WEIGHTS,
        dim_x=dim_x,
        dim_y=dim_y,
        delta=delta,
    )
    fig, _ = caldera_env.visualize_caldera()
    plt.show()


if __name__ == "__main__":
    main()
