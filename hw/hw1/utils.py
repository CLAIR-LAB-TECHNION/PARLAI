"""Utility types and helper functions for Caldera environments."""

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np


@dataclass(frozen=True)
class BivariateNormalStruct:
    """Parameters for a 2D Gaussian (bivariate normal) evaluation.

    Attributes:
        x (Any): X-coordinate(s) at which to evaluate the density.
        y (Any): Y-coordinate(s) at which to evaluate the density.
        sigmax (float): Standard deviation along x-axis.
        sigmay (float): Standard deviation along y-axis.
        mux (float): Mean along x-axis.
        muy (float): Mean along y-axis.
    """

    x: Any
    y: Any
    sigmax: float
    sigmay: float
    mux: float
    muy: float


def bivariate_normal(params: BivariateNormalStruct):
    """Evaluate a 2D Gaussian probability density.

    Args:
        params (BivariateNormalStruct): Gaussian parameters and evaluation
            coordinates.

    Returns:
        Any: Density value(s) at ``(params.x, params.y)``. Return type follows
        input shapes (scalar or NumPy array).
    """
    # Squared Mahalanobis-like terms along each axis.
    x_term = ((params.x - params.mux) ** 2) / (2 * params.sigmax**2)
    y_term = ((params.y - params.muy) ** 2) / (2 * params.sigmay**2)

    # Normalization makes this a proper probability density.
    norm = 1.0 / (2 * np.pi * params.sigmax * params.sigmay)

    # makes the value highest near (mux, muy) and decay smoothly away from it
    return norm * np.exp(-(x_term + y_term))


def validate_bounds(
    position: Tuple[int, int],
    max_position: np.ndarray,
) -> bool:
    """Check whether a position is inside map bounds (inclusive).

    Args:
        position (Tuple[int, int]): Position ``(x, y)`` to validate.
        max_position (np.ndarray): Maximum valid coordinates as
            ``[max_x, max_y]``.

    Returns:
        bool: ``True`` if ``0 <= x <= max_x`` and ``0 <= y <= max_y``.
    """
    x_pos, y_pos = map(int, position)
    return 0 <= x_pos <= max_position[0] and 0 <= y_pos <= max_position[1]


def position_to_indices(
    position: Tuple[int, int],
    sampling_res: int,
) -> Tuple[int, int]:
    """Convert map coordinates to sampling-grid indices.

    Args:
        position (Tuple[int, int]): Map position ``(x, y)``.
        sampling_res (int): Sampling cell size in map units.

    Returns:
        Tuple[int, int]: Grid indices in ``(row, col)`` order, i.e.
        ``(y // sampling_res, x // sampling_res)``.
    """
    x_pos, y_pos = map(int, position)
    return y_pos // sampling_res, x_pos // sampling_res


def is_position_within_bounding_box(
    position: Tuple[int, int],
    bottom_left_position: Tuple[int, int],
    box_size: float,
) -> bool:
    """Check if a position lies inside an axis-aligned square box.

    Bounds are inclusive on all sides.

    Args:
        position (Tuple[int, int]): Position ``(x, y)`` to test.
        bottom_left_position (Tuple[int, int]): Bottom-left corner of the box.
        box_size (float): Side length of the box.

    Returns:
        bool: ``True`` if ``position`` is inside or on the box boundary.
    """
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
    """Generate the axis-aligned path from current to proposed position.

    The path excludes ``current_position`` and includes ``proposed_position``.
    Movement is along a single axis:
    - If x differs, path advances in x while keeping y constant.
    - Otherwise, path advances in y while keeping x constant.

    Args:
        current_position (Tuple[int, int]): Start position ``(x, y)``.
        proposed_position (Tuple[int, int]): Destination position ``(x, y)``.

    Returns:
        List[Tuple[int, int]]: Ordered list of intermediate cells to traverse.
    """
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
