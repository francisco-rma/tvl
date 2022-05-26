import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import scale


def populate(size: int, lower_bound, upper_bound, mean: float, std: float):

    # size: number of individuals in the population

    # lb: array containing the lower bound of talent value
    # lb = np.full(size, lower_bound)

    # ub: array containing the upper bound of talent value
    # ub = np.full(size, upper_bound)

    # mu: array containing the mean of talent distribution
    # mu = np.full(size, mean)

    # stdev: array containing the standard deviation of talent distribution
    # stdev = np.full(size, std)

    talent = np.zeros(size)

    talent = sp.stats.truncnorm.rvs((lower_bound - mean) / std,
                                    (upper_bound - mean) / std,
                                    loc=mean,
                                    scale=std,
                                    size=size)

    # talent = np.random.default_rng().normal(mean, std, size=size)

    mean = np.mean(talent)
    std = np.std(talent)

    talent_sort = np.sort(talent, 0, 'quicksort')
    talent_index = np.argsort(talent, 0, 'quicksort')

    return talent_sort, talent_index

def cpt_map(array: np.ndarray):
    '''Mapping from the position of an individual's random walk to their capital'''

    new_arr = 10 * (2**array)

    return new_arr

def evolution(talent: np.ndarray, time, unlucky_event, lucky_event, history=False):
    '''If history=False (default behavior), returns a 1d array representing the population's final position.

       If history=True, returns a 2d array where:
            - The i-th row represent the time evolution of the i-th individual's position
            - The j-th column represents the population's position at the j-th iteration
            - The element (i, j) represents the position of the i-th individual at the j-th iteration
    '''

    rng = default_rng()

    pos = np.zeros((len(talent), time))

    # Initializing all individuals to a starting position of 0:

    np.place(pos[:, 0], mask=np.zeros(len(talent)) == 0, vals=0)

    if history:
        # Returns a 2d array where:
            # The i-th row represent the time evolution of the i-th individual's position
            # The j-th column represents the population's position at the j-th iteration
            # The element (i, j) represents the position of the i-th individual at the j-th iteration

        for i in range(time - 1):
            a = rng.uniform(0.0, 1.0, size=len(talent))
            b = rng.uniform(0.0, 1.0, size=len(talent))

            arr_source = pos[:, i]

            # Creating logical masks for each scenario:

                # Scenario 1: individual went through an unlucky event

            unlucky_mask = a < unlucky_event

                # Scenario 2: individual went through a lucky event and capitalized

            lucky_mask = ((a >= unlucky_event) &
                          (a < unlucky_event + lucky_event) &
                          (b < talent)
                          )

                # Scenario 3: individual didn't go through any events OR went through a lucky event and failed to capitalize:

            neutral_mask = ((a >= unlucky_event + lucky_event) |
                            ((a >= unlucky_event) &
                            (a < unlucky_event + lucky_event) &
                            (b > talent))
                            )

            # Upadting position of those in scenario 1:

            unlucky_vals = np.extract(unlucky_mask, arr_source) - 1

            np.place(pos[:, i + 1], mask=unlucky_mask, vals=unlucky_vals)

            # Upadting position of those in scenario 2:

            lucky_vals = np.extract(lucky_mask, arr_source) + 1

            np.place(pos[:, i + 1], mask=lucky_mask, vals=lucky_vals)

            # Upadting position of those in scenario 3:

            neutral_vals = np.extract(neutral_mask, arr_source)

            np.place(pos[:, i + 1], mask=neutral_mask, vals=neutral_vals)

        return pos

    else:
        # Default behavior
        # Returns a 1d array representing the population's final position

        iter = 0
        arr_source = pos[:, 0]

        while iter < time:

            a = rng.uniform(0.0, 1.0, size=len(talent))
            b = rng.uniform(0.0, 1.0, size=len(talent))

            # Creating logical masks for each scenario:

                # Scenario 1: individual went through an unlucky event
            unlucky_mask = a < unlucky_event

                # Scenario 2: individual went through a lucky event AND capitalized
            lucky_mask = ((a >= unlucky_event) &
                          (a < unlucky_event + lucky_event) &
                          (b <= talent))

                # Scenario 3: individual didn't go through any events OR went through a lucky event and failed to capitalize.
                # No mask is neede because no updates are done.

            # Upadting position of those in scenario 1:
            arr_source[unlucky_mask] = arr_source[unlucky_mask] - 1

            # Upadting position of those in scenario 2:
            arr_source[lucky_mask] = arr_source[lucky_mask] + 1

            iter += 1

    return arr_source
