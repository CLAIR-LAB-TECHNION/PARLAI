from dataclasses import dataclass
from typing import Any, List, Tuple

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

# returns True if the position is within the bounds of the map, False otherwise
def validate_bounds(
    position: Tuple[int, int],
    max_position: np.ndarray,
) -> bool:
    x_pos, y_pos = map(int, position)
    return 0 <= x_pos <= max_position[0] and 0 <= y_pos <= max_position[1]


def position_to_indices(
    position: Tuple[int, int],
    sampling_res: int,
) -> Tuple[int, int]:
    x_pos, y_pos = map(int, position)
    return y_pos // sampling_res, x_pos // sampling_res


def is_position_within_bounding_box(
    position: Tuple[int, int],
    bottom_left_position: Tuple[int, int],
    box_size: float,
) -> bool:
    position_x, position_y = position
    bottom_left_x, bottom_left_y = bottom_left_position

    x_min = bottom_left_x
    x_max = bottom_left_x + box_size
    y_min = bottom_left_y
    y_max = bottom_left_y + box_size

    return x_min <= position_x <= x_max and y_min <= position_y <= y_max


def generate_path(
    current_position: Tuple[int, int],
    proposed_position: Tuple[int, int],
) -> List[Tuple[int, int]]:
    current_x, current_y = map(int, current_position)
    proposed_x, proposed_y = map(int, proposed_position)

    if current_x != proposed_x:
        step_direction = 1 if proposed_x > current_x else -1
        return [
            (x_pos, current_y)
            for x_pos in range(
                current_x + step_direction,
                proposed_x + step_direction,
                step_direction,
            )
        ]

    step_direction = 1 if proposed_y > current_y else -1
    return [
        (current_x, y_pos)
        for y_pos in range(
            current_y + step_direction,
            proposed_y + step_direction,
            step_direction,
        )
    ]
