"""Provides functions to normalize the input data and to calculate the mean and std for normalization."""

import numpy as np

from rlgammon.rlgammon_types import Input


def group_stats(array_list: list[Input], start: int, end: int) -> tuple[np.floating, np.floating]:
    """
    Computes the mean and standard deviation for a specific group of cells across all arrays.

    Used to find the mean and std of each cell to normalize the input.
    """
    group_values = np.concatenate([arr[start:end + 1] for arr in array_list])
    mean = np.mean(group_values)
    std = np.std(group_values)
    print(f"Group cells {start} to {end}: Mean = {mean:.3f}, Std = {std:.3f}")
    return mean, std


cell_stats = [
    (2.348, 2.787),
    (1.684, 2.157),
    (0.702, 1.152),
    (0.521, 0.947),
    (0.330, 0.777),
    (1.510, 2.145),
    (0.264, 0.639),
    (0.518, 1.058),
    (0.196, 0.512),
    (0.201, 0.569),
    (0.143, 0.397),
    (0.183, 0.578),
    (0.835, 1.500),
    (0.133, 0.373),
    (0.201, 0.510),
    (0.196, 0.497),
    (0.138, 0.419),
    (0.273, 0.627),
    (0.192, 0.694),
    (0.418, 0.872),
    (0.224, 0.649),
    (0.375, 0.885),
    (0.106, 0.363),
    (0.279, 0.672),
    (0.284, 0.662),
    (0.117, 0.374),
    (0.384, 0.886),
    (0.237, 0.654),
    (0.433, 0.874),
    (0.200, 0.698),
    (0.285, 0.631),
    (0.145, 0.425),
    (0.205, 0.503),
    (0.207, 0.513),
    (0.145, 0.384),
    (0.808, 1.455),
    (0.190, 0.582),
    (0.156, 0.408),
    (0.215, 0.576),
    (0.208, 0.520),
    (0.501, 1.036),
    (0.276, 0.643),
    (1.489, 2.120),
    (0.358, 0.785),
    (0.549, 0.952),
    (0.734, 1.153),
    (1.735, 2.168),
    (2.419, 2.800),
    (0.356, 0.592),
    (0.058, 0.273),
    (4.355, 8.740),
    (4.340, 8.749),
]


def normalize_input(arr: Input, stats: list[tuple[float, float]]) -> Input:
    """Normalize the observations."""
    standardized = np.empty_like(arr, dtype=float)
    for i, (mean, std) in enumerate(stats):
        standardized[i] = (arr[i] - mean) / std
    return standardized
