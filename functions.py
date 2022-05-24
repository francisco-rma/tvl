import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import scale


def population(size, lower_bound, upper_bound, mean, std):

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


def evolution(talent, time, unlucky_event, lucky_event, history=False):

    rng = default_rng()

    capital = np.zeros((len(talent), time))

    # Initializing all individuals to a starting capital of 10.0:

    np.place(capital[:, 0], mask=np.zeros(len(talent)) == 0, vals=10.0)

    if history:

        # Returns a 2d array where:
            # The i-th row represent the time evolution of the i-th individual's capital
            # The j-th column represents the population's capital at the j-th iteration
            # The element (i, j) represents the capital of the i-th individual at the j-th iteration

        for i in range(time - 1):
            a = rng.uniform(0.0, 1.0, size=len(talent))
            b = rng.uniform(0.0, 1.0, size=len(talent))

            arr_source = capital[:, i]

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

            # Upadting capital of those in scenario 1:

            unlucky_vals = np.extract(unlucky_mask, arr_source) / 2

            np.place(capital[:, i + 1], mask=unlucky_mask, vals=unlucky_vals)

            # Upadting capital of those in scenario 2:

            lucky_vals = np.extract(lucky_mask, arr_source) * 2

            np.place(capital[:, i + 1], mask=lucky_mask, vals=lucky_vals)

            # Upadting capital of those in scenario 3:

            neutral_vals = np.extract(neutral_mask, arr_source)

            np.place(capital[:, i + 1], mask=neutral_mask, vals=neutral_vals)

            # Checking if the masks cover the entirety of the source array:

            full_mask = np.array((unlucky_mask, lucky_mask, neutral_mask))
            check = np.logical_or.reduce(full_mask)

            if np.all(check):
                pass
            else:
                raise ValueError('Failure to update capital')

        return capital

    else:

        # Default behavior
        # Returns a 1d array representing the population's final capital

        iter = 0
        arr_source = capital[:, 0]

        while iter < time:

            a = rng.uniform(0.0, 1.0, size=len(talent))
            b = rng.uniform(0.0, 1.0, size=len(talent))

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

            # Upadting capital of those in scenario 1:

            unlucky_vals = np.extract(unlucky_mask, arr_source) / 2

            np.place(arr_source, mask=unlucky_mask, vals=unlucky_vals)

            # Upadting capital of those in scenario 2:

            lucky_vals = np.extract(lucky_mask, arr_source) * 2

            np.place(arr_source, mask=lucky_mask, vals=lucky_vals)

            # Upadting capital of those in scenario 3:

            neutral_vals = np.extract(neutral_mask, arr_source)

            np.place(arr_source, mask=neutral_mask, vals=neutral_vals)

            # Checking if the masks cover the entirety of the source array:

            full_mask = np.array((unlucky_mask, lucky_mask, neutral_mask))
            check = np.logical_or.reduce(full_mask)

            if np.all(check):
                pass
            else:
                raise ValueError('Failure to update capital')

            iter += 1

    return arr_source
