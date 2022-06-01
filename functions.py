import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import scale

def populate(size: int, lower_bound, upper_bound, mean: float, std: float):
    '''
    Create a population with the desired parameters for talent distribution:
        -size: number of individuals in the population
        -lower_bound: minimum value of talent distribution
        -upper_bound: maximum value of talent distribution
        -mean: average of talent distribution
        -std: standard deviation of talent distribution

    Returns:

        -talent: array containing sorted values of talent

        -t_i: array containing the indices to the unsorted talent array'''

    talent = np.zeros(size)

    talent = sp.stats.truncnorm.rvs((lower_bound - mean) / std,
                                    (upper_bound - mean) / std,
                                    loc=mean,
                                    scale=std,
                                    size=size)

    mean = np.mean(talent)
    std = np.std(talent)

    talent_sort = np.sort(talent, 0, 'quicksort')
    talent_index = np.argsort(talent, 0, 'quicksort')

    return talent_sort, talent_index

def cpt_map(array: np.ndarray):
    '''Mapping from the position of the random walk to capital'''

    new_arr = 10 * (2**array)

    return new_arr

def evolution(talent: np.ndarray, time, unlucky_event, lucky_event, history=False):
    '''
    Perform the simulation proper:
        -talent: array containing the sorted talent distribution of the population
        -time: number of iterations
        -unlucky_event: chance for an individual to go through an unlucky event
        -lucky_event: chance for an individual to go through a lucky event

    Returns:

        If history=False:
            -arr_source: 1d array containing the final positions of each individual

        If history=True:
            -pos: 2d array containing the positions at each time for each individual

            '''

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
            negative_mask = (a <= unlucky_event)

                # Scenario 2: lucky event AND capitalized
            positive_mask = ((a > unlucky_event) &
                             (a <= unlucky_event + lucky_event) &
                             (b <= talent))

                # Scenario 3: (no events) OR
                # (lucky event AND failed to capitalize)
            neutral_mask = ((a > unlucky_event + lucky_event) |
                            ((a > unlucky_event) &
                            (a <= unlucky_event + lucky_event) &
                            (b > talent)))

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
            positive_mask = ((a > unlucky_event) &
                             (a <= unlucky_event + lucky_event) &
                             (b <= talent))

                # Scenario 3: no events OR
                # lucky event AND failed to capitalize.
                # No mask is needed because no updates are done.

            # Upadting position of those in scenario 1:
            arr_source[negative_mask] = arr_source[negative_mask] - 1

            # Upadting position of those in scenario 2:
            arr_source[positive_mask] = arr_source[positive_mask] + 1

            iter += 1

        return arr_source

def symmetric_evolution(talent: np.ndarray, time, unlucky_event, lucky_event, history=False):
    '''
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

            '''

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
            positive_mask = ((a > unlucky_event) &
                             (a <= unlucky_event + lucky_event) &
                             (b <= talent))

                # Scenario 3: (no events) OR
                            # (lucky event AND failed to capitalize) OR
                            # (unlucky event AND escaped)
            neutral_mask = ((a > unlucky_event + lucky_event) |
                            ((a > unlucky_event) &
                             (a <= unlucky_event + lucky_event) &
                             (b > talent)) |
                            ((a <= unlucky_event) &
                             (b <= talent)))

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

                # Scenario 1: unlucky event AND didn't escape
            negative_mask = (a <= unlucky_event) & (b > talent)

                # Scenario 2: lucky event AND capitalized
            positive_mask = ((a > unlucky_event) &
                             (a <= unlucky_event + lucky_event) &
                             (b <= talent))

                # Scenario 3: (no events) OR
                            # (lucky event AND failed to capitalize) OR
                            # (unlucky event AND escaped)
                # No mask is needed because no updates are done.

            # Upadting position of those in scenario 1:
            arr_source[negative_mask] = arr_source[negative_mask] - 1

            # Upadting position of those in scenario 2:
            arr_source[positive_mask] = arr_source[positive_mask] + 1

            iter += 1

        return arr_source

def many_runs(talent: np.ndarray, time, unlucky_event, lucky_event, runs, sym=False):

    # Initialize arrays to hold the position and the talent for the most succesful individual of each run:

    # mst: Most Successful Talent (talent of the most succesful individual)
    mst = np.empty(runs)

    # msp: Most Successful Position (final position of the most succesful individual)
    msp = np.empty(runs)

    # Create an array to store values of talent and position for those who were overall successful:
    positive = np.zeros((1, 2))

    # Perform the simulations:
    if sym:

        for i in range(runs):

            final_pos = symmetric_evolution(talent, time, unlucky_event, lucky_event)

            positive_per_run = np.column_stack((talent[final_pos > 0], final_pos[final_pos > 0]))

            positive = np.concatenate((positive, positive_per_run))

            mst[i] = talent[np.argmax(final_pos)]

            msp[i] = np.max(final_pos)

        return mst, msp, positive[1:, :]

    else:

        for i in range(runs):

            final_pos = evolution(talent, time, unlucky_event, lucky_event)

            positive_per_run = np.column_stack((talent[final_pos > 0], final_pos[final_pos > 0]))

            positive = np.concatenate((positive, positive_per_run))

            mst[i] = talent[np.argmax(final_pos)]

            msp[i] = np.max(final_pos)

        return mst, msp, positive[1:, :]
