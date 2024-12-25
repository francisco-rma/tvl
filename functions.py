from scipy import ndimage
import scipy.stats as stats
import numpy as np
from numpy.random import default_rng
import pandas as pd

NORM_FACTOR = float(1 / 4)


def populate(size: int, lower_bound, upper_bound, mean: float, std: float):
    """
    Create a population with the desired parameters for talent distribution:
        -size: number of individuals in the population
        -lower_bound: minimum value of talent distribution
        -upper_bound: maximum value of talent distribution
        -mean: average of talent distribution
        -std: standard deviation of talent distribution

    Returns:

        -talent: array containing sorted values of talent

        -t_i: array containing the indices to the unsorted talent array"""

    talent = np.zeros(size)

    talent = stats.truncnorm.rvs(
        (lower_bound - mean) / std,
        (upper_bound - mean) / std,
        loc=mean,
        scale=std,
        size=size,
    )

    mean = np.mean(talent)
    std = np.std(talent)

    talent_sort = np.sort(talent, 0, "quicksort")
    talent_index = np.argsort(talent, 0, "quicksort")

    return talent_sort, talent_index


def interactive_populate(size: int, lower_bound, upper_bound, mean: float, std: float):
    """
    Create a population with the desired parameters for talent distribution:
        -size: number of individuals in the population
        -lower_bound: minimum value of talent distribution
        -upper_bound: maximum value of talent distribution
        -mean: average of talent distribution
        -std: standard deviation of talent distribution

    Returns:

        -talent: array containing sorted values of talent"""

    talent = np.zeros((size, size))

    # Deliberate talent bias for testing
    for i in range(size):
        talent[i, :] = stats.truncnorm.rvs(
            (lower_bound - mean) / std,
            (upper_bound - mean) / std,
            loc=mean,
            scale=std,
            size=(size),
        )

    # An unbiased solution should look like:
    # unbiased_talent = stats.truncnorm.rvs(
    #     (lower_bound - mean) / std,
    #     (upper_bound - mean) / std,
    #     loc=mean,
    #     scale=std,
    #     size=(size**2),
    # )

    # populated: set[tuple] = set()
    # for value in unbiased_talent:
    #     i, j = rng.integers(low=0, high=len(talent)), rng.integers(
    #         low=0, high=len(talent)
    #     )

    #     while (i, j) in populated:
    #         i, j = rng.integers(low=0, high=len(talent)), rng.integers(
    #             low=0, high=len(talent)
    #         )

    #     talent[i, j] = value
    #     populated.add((i, j))

    return talent


def map_to_capital(array: np.ndarray):
    """Mapping from the position of the random walk to capital"""

    new_arr = 10 * (2**array)

    return new_arr


def tvl(talent: np.ndarray, time, unlucky_event, lucky_event, history=False):
    """
    Perform the simulation proper:
        -talent: array containing the sorted talent distribution of the population
        -time: number of iterations
        -unlucky_event: chance for an individual to go through an unlucky event
        -lucky_event: chance for an individual to go through a lucky event

    Returns:

        If history==False:
            -arr_source: 1d array containing the final positions of each individual

        If history==True:
            -pos: 2d array containing the positions at each time for each individual

    """

    rng = default_rng()

    if history:
        # Returns a 2d array where:
        # The i-th row represent the time evolution of the i-th individual's position
        # The j-th column represents the population's position at the j-th iteration
        # The element (i, j) represents the position of the i-th individual at the j-th iteration

        pos = np.zeros((len(talent), time))

        for i in range(time - 1):
            a = rng.uniform(0.0, 1.0, size=len(talent))
            b = rng.uniform(0.0, 1.0, size=len(talent))

            # Creating logical masks for each scenario:

            # Scenario 1: unlucky event
            negative_mask = a < unlucky_event

            # Scenario 2: lucky event AND capitalized
            positive_mask = (
                (a > unlucky_event) & (a <= unlucky_event + lucky_event) & (b <= talent)
            )

            # Scenario 3: (no events) OR
            # (lucky event AND failed to capitalize)
            neutral_mask = (a > unlucky_event + lucky_event) | (
                (a > unlucky_event) & (a <= unlucky_event + lucky_event) & (b > talent)
            )

            arr_source = pos[:, i]

            # Upadting position of those in scenario 1:
            pos[negative_mask, i + 1] = arr_source[negative_mask] - 1

            # Upadting position of those in scenario 2:
            pos[positive_mask, i + 1] = arr_source[positive_mask] + 1

            # Upadting position of those in scenario 3:
            pos[neutral_mask, i + 1] = arr_source[neutral_mask]

        return pos

    else:
        # Default behavior
        # Returns a 1d array representing the population's final position

        iter = 0
        arr_source = np.zeros(len(talent))

        while iter < time:
            a = rng.uniform(0.0, 1.0, size=len(talent))
            b = rng.uniform(0.0, 1.0, size=len(talent))

            # Creating logical masks for each scenario:

            # Scenario 1: unlucky event
            negative_mask = a <= unlucky_event

            # Scenario 2: lucky event AND capitalized
            positive_mask = (
                (a > unlucky_event) & (a <= unlucky_event + lucky_event) & (b <= talent)
            )

            # Scenario 3: no events OR
            # lucky event AND failed to capitalize
            # neutral_mask = (a > unlucky_event + lucky_event) | (
            #     (a > unlucky_event) & (a <= unlucky_event + lucky_event) & (b > talent)
            # )

            # Upadting position of those in scenario 1:
            arr_source[negative_mask] = arr_source[negative_mask] - 1

            # Upadting position of those in scenario 2:
            arr_source[positive_mask] = arr_source[positive_mask] + 1

            iter += 1

        return arr_source


def interactive_tvl(
    talent: np.ndarray, time, unlucky_event, lucky_event, history=False
):
    """
    Perform the simulation proper:
        -talent: array containing the sorted talent distribution of the population
        -time: number of iterations
        -unlucky_event: chance for an individual to go through an unlucky event
        -lucky_event: chance for an individual to go through a lucky event

    Returns:

        If history==False:
            -arr_source: 2d array containing the final positions of each individual

        If history==True:
            -pos: 3d array containing the positions at each time for each individual

    """
    assert talent.shape[0] == talent.shape[1]

    rng = default_rng()

    iter = 0
    arr_source = np.zeros(talent.shape)
    colors = np.zeros(talent.shape)
    color_map = {0: "\033[91m", 1: "\033[92m", 2: "\033[94m"}  # Red  # Green  # Blue

    a = rng.uniform(0.0, 1.0, size=(time, talent.shape[0], talent.shape[1]))
    b = rng.uniform(0.0, 1.0, size=(time, talent.shape[0], talent.shape[1]))

    while iter < time:
        neighbours_avg = neighbour_avg(arr_source)
        # Creating logical masks for each scenario:

        # Scenario 1: unlucky event
        negative_event_mask = (a[iter] <= unlucky_event) & (
            arr_source >= neighbours_avg
        )

        # Scenario 2: lucky event AND capitalized
        positive_event_mask = (
            (a[iter] > unlucky_event)
            & (a[iter] <= unlucky_event + lucky_event)
            & (b[iter] <= talent)
        )

        # Scenario 3*: no events OR lucky event AND failed to capitalize
        # * no mask needed because no operations are made

        # Upadting position of those in scenario 1:
        arr_source[negative_event_mask] = arr_source[negative_event_mask] - 1

        # Upadting position of those in scenario 2:
        arr_source[positive_event_mask] = arr_source[positive_event_mask] + 1

        iter += 1

    return arr_source


def symmetric_evolution(
    talent: np.ndarray, time, unlucky_event, lucky_event, history=False
):
    """
    Perform the simulation with the possibility of escaping unlucky events:
        -talent: array containing the sorted talent distribution of the population
        -time: number of iterations
        -unlucky_event: chance for an individual to go through an unlucky event
        -lucky_event: chance for an individual to go through a lucky event

    Returns:

        If history=False:
            -arr_source: 1d array containing the final positions of each individual

        If history=True:
            -pos: 2d array containing the positions at each time for each individual

    Note
        The only real change is in creation of the logical masks for the arrays.

    """

    rng = default_rng()

    if history:
        # Returns a 2d array where:
        # The i-th row represent the time evolution of the i-th individual's position
        # The j-th column represents the population's position at the j-th iteration
        # The element (i, j) represents the position of the i-th individual at the j-th iteration

        pos = np.zeros((len(talent), time))

        for i in range(time - 1):
            a = rng.uniform(0.0, 1.0, size=len(talent))
            b = rng.uniform(0.0, 1.0, size=len(talent))

            # Creating logical masks for each scenario:

            # Scenario 1: unlucky event AND didn't escape
            negative_mask = (a <= unlucky_event) & (b > talent)

            # Scenario 2: lucky event AND capitalized
            positive_mask = (
                (a > unlucky_event) & (a <= unlucky_event + lucky_event) & (b <= talent)
            )

            # Scenario 3: (no events) OR
            # (lucky event AND failed to capitalize) OR
            # (unlucky event AND escaped)
            neutral_mask = (
                (a > unlucky_event + lucky_event)
                | (
                    (a > unlucky_event)
                    & (a <= unlucky_event + lucky_event)
                    & (b > talent)
                )
                | ((a <= unlucky_event) & (b <= talent))
            )

            arr_source = pos[:, i]

            # Upadting position of those in scenario 1:
            pos[negative_mask, i + 1] = arr_source[negative_mask] - 1

            # Upadting position of those in scenario 2:
            pos[positive_mask, i + 1] = arr_source[positive_mask] + 1

            # Upadting position of those in scenario 3:
            pos[neutral_mask, i + 1] = arr_source[neutral_mask]

        return pos

    else:
        # Default behavior
        # Returns a 1d array representing the population's final position

        iter = 0
        arr_source = np.zeros(len(talent))

        while iter < time:
            a = rng.uniform(0.0, 1.0, size=len(talent))
            b = rng.uniform(0.0, 1.0, size=len(talent))

            # Creating logical masks for each scenario:

            # Scenario 1: unlucky event
            negative_mask = a < unlucky_event

            # Scenario 2: lucky event AND capitalized
            positive_mask = (
                (a >= unlucky_event) & (a < unlucky_event + lucky_event) & (b <= talent)
            )

            # Scenario 3: no events OR lucky event and failed to capitalize.

            # No mask is needed because no updates are done.

            # Upadting position of those in scenario 1:
            arr_source[negative_mask] = arr_source[negative_mask] - 1

            # Upadting position of those in scenario 2:
            arr_source[positive_mask] = arr_source[positive_mask] + 1

            iter += 1

        return arr_source


def many_runs(
    talent: np.ndarray,
    time: int,
    unlucky_event: float,
    lucky_event: float,
    runs: int,
    symmetricEvolution: bool = False,
):
    # Initialize arrays to hold the position and the talent for the most succesful individual of each run:

    # mst: Most Successful Talent (talent of the most succesful individual)
    mst = np.empty(runs)

    # msp: Most Successful Position (final position of the most succesful individual)
    msp = np.empty(runs)

    # Create an array to store values of talent and position for those who were overall successful:
    positive = np.zeros((1, 2))

    # Perform the simulations:
    if symmetricEvolution:
        for i in range(runs):
            final_pos = symmetric_evolution(talent, time, unlucky_event, lucky_event)

            positive_per_run = np.column_stack(
                (talent[final_pos > 0], final_pos[final_pos > 0])
            )

            positive = np.concatenate((positive, positive_per_run))

            mst[i] = talent[np.argmax(final_pos)]

            msp[i] = np.max(final_pos)

        return mst, msp, positive[1:, :]

    else:
        for i in range(runs):
            final_pos = tvl(talent, time, unlucky_event, lucky_event)

            positive_per_run = np.column_stack(
                (talent[final_pos > 0], final_pos[final_pos > 0])
            )

            positive = np.concatenate((positive, positive_per_run))

            mst[i] = talent[np.argmax(final_pos)]

            msp[i] = np.max(final_pos)

    return mst, msp, positive[1:, :]


def list_neighbours(
    lattice: np.ndarray, list_neighbour: np.ndarray, periodic_conditions: bool = True
):
    size = len(lattice)
    if not periodic_conditions:
        for i, row in enumerate(lattice):
            for j, point in enumerate(row):
                # lattice[i, j] = (i * size) + (j + 1)
                neighbours = list()
                if j != 0:
                    neighbour_l = lattice[i, j - 1]
                    neighbours.append(neighbour_l)
                if j != size - 1:
                    neighbour_r = lattice[i, j + 1]
                    neighbours.append(neighbour_r)
                if i != 0:
                    neighbour_t = lattice[i - 1, j]
                    neighbours.append(neighbour_t)
                if i != size - 1:
                    neighbour_b = lattice[i + 1, j]
                    neighbours.append(neighbour_b)

                index = (i * size) + j

                list_neighbour[index] = neighbours
    else:
        for i, row in enumerate(lattice):
            for j, point in enumerate(row):
                # lattice[i, j] = (i * size) + (j + 1)
                neighbours = []
                if j != 0:
                    neighbour_l = lattice[i, j - 1]
                    neighbours.append(neighbour_l)
                else:
                    neighbour_l = lattice[i, size - 1]
                    neighbours.append(neighbour_l)

                if j != size - 1:
                    neighbour_r = lattice[i, j + 1]
                    neighbours.append(neighbour_r)
                else:
                    neighbour_r = lattice[i, 0]
                    neighbours.append(neighbour_r)

                if i != 0:
                    neighbour_t = lattice[i - 1, j]
                    neighbours.append(neighbour_t)
                else:
                    neighbour_t = lattice[size - 1, j]
                    neighbours.append(neighbour_t)

                if i != size - 1:
                    neighbour_b = lattice[i + 1, j]
                    neighbours.append(neighbour_b)
                else:
                    neighbour_b = lattice[0, j]
                    neighbours.append(neighbour_b)

            index = (i * size) + j

            list_neighbour[index] = neighbours

    return list_neighbour


def neighbour_avg(values: np.ndarray) -> np.ndarray:
    assert values.shape[0] == values.shape[1]
    kernel = np.array(
        [[0, NORM_FACTOR, 0], [NORM_FACTOR, 0, NORM_FACTOR], [0, NORM_FACTOR, 0]]
    )
    return ndimage.convolve(values, kernel, mode="constant")


# def interactive_tvl(
#     talent: np.ndarray, time: int, unlucky_event, lucky_event, history=True
# ):
#     """
#     - talent: 1-dimensional array containing the value of talent for each indiviudal of the simulation
#     - time: integer that represents the number of iterations of the simulation
#     - unlucky_event: real number that represents the probability of an idividual going through an unlucky event
#     - lucky_event: real number that represents the probability of an idividual going through an lucky event
#     - history:  boolean that determines whether or not the return should contain the entire history of the simulation or just the final state
#                 specifically, if history is true, the return will be a 3-dimensional array containing, along its x-axis
#     """
#     rng = default_rng()
#     side = len(talent)

#     if side - int(side) != 0.0:
#         print("Vetor de talentos inválido, deve possuir raiz quadrada inteira")
#         return

#     side = int(side)
#     talent = np.reshape(talent, (side, side))

#     if history:
#         # Returns a 2d array where:
#         # The i-th row represent the time evolution of the i-th individual's position
#         # The j-th column represents the population's position at the j-th iteration
#         # The element (i, j) represents the position of the i-th individual at the j-th iteration

#         pos = np.zeros((side, side, time))
#         # print(pos)

#         for i in range(time - 1):
#             a = rng.uniform(0.0, 1.0, size=n)
#             a = np.reshape(a, (side, side))

#             b = rng.uniform(0.0, 1.0, size=n)
#             b = np.reshape(b, (side, side))

#             # Creating logical masks for each scenario:

#             # Scenario 1: unlucky event
#             negative_mask = a < unlucky_event

#             # Scenario 2: lucky event AND capitalized
#             positive_mask = (
#                 (a > unlucky_event) & (a <= unlucky_event + lucky_event) & (b <= talent)
#             )
#             # Scenario 3: no events OR
#             # lucky event AND failed to capitalize
#             neutral_mask = (a > unlucky_event + lucky_event) | (
#                 (a > unlucky_event) & (a <= unlucky_event + lucky_event) & (b > talent)
#             )

#             arr_source = pos[:, :, i]

#             initial_pos_df = pd.DataFrame(pos[:, :, i])

#             # Upadting position of those in scenario 1:
#             pos[negative_mask, i + 1] = arr_source[negative_mask] - 1

#             # Upadting position of those in scenario 2:
#             pos[positive_mask, i + 1] = arr_source[positive_mask] + 1

#             # Upadting position of those in scenario 3:
#             pos[neutral_mask, i + 1] = arr_source[neutral_mask]

#             final_pos_df = pd.DataFrame(pos[:, :, i + 1])
#             negative_mask_df = pd.DataFrame(negative_mask)
#             positive_mask_df = pd.DataFrame(positive_mask)
#             neutral_mask_df = pd.DataFrame(neutral_mask)

#             print(f"Iteracão de número: {i}")
#             print("Máscara negativa")
#             print(negative_mask_df)
#             print("Máscara positiva")
#             print(positive_mask_df)
#             print("Máscara neutra")
#             print(neutral_mask_df)
#             print(initial_pos_df)
#             print(final_pos_df)

#         return pos

#     else:
#         # Default behavior
#         # Returns a 1d array representing the population's final position

#         iter = 0
#         arr_source = np.zeros(len(talent))

#         while iter < time:
#             a = rng.uniform(0.0, 1.0, size=len(talent))
#             b = rng.uniform(0.0, 1.0, size=len(talent))

#             # Creating logical masks for each scenario:

#             # Scenario 1: unlucky event
#             negative_mask = a < unlucky_event

#             # Scenario 2: lucky event AND capitalized
#             positive_mask = (
#                 (a >= unlucky_event) & (a < unlucky_event + lucky_event) & (b <= talent)
#             )

#             # Scenario 3: no events OR lucky event and failed to capitalize.
#             # No mask is needed because no updates are done.

#             # Upadting position of those in scenario 1:
#             arr_source[negative_mask] = arr_source[negative_mask] - 1

#             # Upadting position of those in scenario 2:
#             arr_source[positive_mask] = arr_source[positive_mask] + 1

#             iter += 1

#         return arr_source
#     return
