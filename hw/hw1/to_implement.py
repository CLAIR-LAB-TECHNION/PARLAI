from typing import Any

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt



class CalderaEnv(gym.Env):
    def __init__(self, a_string: str):
        self.a_string = a_string

        #TODO initialize the observation space
        self.observation_space = ...

        #TODO initialize the action space
        self.action_space = ...

    def reset(self, *, seed, options):
        super().reset(seed, options)

        #TODO implement the reset method
        ...

    def step(self, action: int):
        #TODO implement the step method
        ...


def _bivariate_normal(x, y, sigmax, sigmay, mux, muy):
    x_term = ((x - mux) ** 2) / (2 * sigmax ** 2)
    y_term = ((y - muy) ** 2) / (2 * sigmay ** 2)
    norm = 1.0 / (2 * np.pi * sigmax * sigmay)
    return norm * np.exp(-(x_term + y_term))


def caldera_sim_function(x, y, map_size=(100,100)):
    x, y = x / (map_size[0] / 10.), y / (map_size[1] / 10.)
    pit0 = _bivariate_normal(x, y, 1.6, 1.5, 2.0, 2.0)
    pit1 = _bivariate_normal(x, y, 1.8, 2.0, 5.0, 7.0)
    pit2 = _bivariate_normal(x, y, 1.7, 1.5, 8.0, 3.0)
    return -(16000.0 * pit0 + 22000.0 * pit1 + 18000.0 * pit2)


def generate_caldera_map(dim_x, dim_y):
    x = np.arange(0, dim_x, 1)
    y = np.arange(0, dim_y, 1)
    x, y = np.meshgrid(x, y)
    z = caldera_sim_function(x, y, map_size=(dim_x, dim_y))
    return z


def main():
    delta = 10

    # the x and y dimensions of our cladera 
    dim_x = 100
    dim_y = 100

    x = np.arange(0, dim_x + 1, delta)
    y = np.arange(0, dim_y + 1, delta)
    X, Y = np.meshgrid(x, y)
    Z = caldera_sim_function(X, Y, map_size=(dim_x, dim_y))
    return X, Y, Z


def visualize_caldera():
    X, Y, Z = main()
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="Relative depth")
    ax.set_title("Caldera Depth Map")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return fig, ax

if __name__ == "__main__":
    visualize_caldera()
    plt.show()