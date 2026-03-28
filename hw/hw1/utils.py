from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np


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
def bivariate_normal(params: BivariateNormalStruct):
    # x_term measures how far x is from the center mux, scaled by sigmax
    x_term = ((params.x - params.mux) ** 2) / (2 * params.sigmax ** 2)
    # y_term does the same for y
    y_term = ((params.y - params.muy) ** 2) / (2 * params.sigmay ** 2)
    #normalization constant so the whole distribution integrates to 1
    norm = 1.0 / (2 * np.pi * params.sigmax * params.sigmay)
    #makes the value highest near (mux, muy) and decay smoothly away from it
    return norm * np.exp(-(x_term + y_term))


def validate_bounds(
    position: Tuple[int, int],
    max_position: np.ndarray,
    delta: int,
) -> np.ndarray:
    x_pos, y_pos = map(int, position)

    if x_pos < 0 or x_pos > max_position[0] or x_pos % delta != 0:
        raise ValueError(
            f"x position must be on the grid between 0 and {max_position[0]}"
        )
    if y_pos < 0 or y_pos > max_position[1] or y_pos % delta != 0:
        raise ValueError(
            f"y position must be on the grid between 0 and {max_position[1]}"
        )

    return np.array([x_pos, y_pos], dtype=np.int64)


def position_to_indices(
    position: Tuple[int, int],
    delta: int,
) -> Tuple[int, int]:
    x_pos, y_pos = map(int, position)
    return y_pos // delta, x_pos // delta


def indices_to_position(
    indices: Tuple[int, int],
    delta: int,
) -> Tuple[int, int]:
    row, col = map(int, indices)
    return col * delta, row * delta


def is_cell_within_bounding_box(
    cell: Tuple[int, int],
    center_position: Tuple[int, int],
    box_size: float,
) -> bool:
    cell_x, cell_y = cell
    center_x, center_y = center_position
    half_size = box_size / 2.0

    x_min = center_x - half_size
    x_max = center_x + half_size
    y_min = center_y - half_size
    y_max = center_y + half_size

    return x_min <= cell_x < x_max and y_min <= cell_y < y_max
