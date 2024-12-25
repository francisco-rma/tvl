import numpy as np
from scipy import ndimage

NORM_FACTOR = float(1 / 4)


def list_neighbours(values: np.ndarray, row: int, col: int):
    result = []
    directions = [(1, 0), (0, 1), (-1, 0), (0, 1)]

    for i, j in directions:
        next_i, next_j = row + i, row + j
        if 0 <= next_i < values.shape[0] and 0 <= next_j < values.shape[1]:
            result.append((next_i, next_j))


def neighbour_avg(values: np.ndarray) -> np.ndarray:
    assert values.shape[0] == values.shape[1]
    kernel = np.array(
        [[0, NORM_FACTOR, 0], [NORM_FACTOR, 0, NORM_FACTOR], [0, NORM_FACTOR, 0]]
    )
    return ndimage.convolve(values, kernel, mode="constant")


rng = np.random.default_rng()
source = rng.integers(low=-10, high=10, size=(3, 3))


target = np.zeros(source.shape)


print("Source:")
print(source)

print("Target:")
print(target)
