from signal import default_int_handler

import matplotlib
import functions.core as f
from matplotlib import pyplot as plt
from matplotlib import animation as an
import matplotlib.patches as mpatches
import numpy as np

from structs.tvl_struct import tvl_struct


class tvl():

    # iter_n:
    iter_n = None

    # pop_n: number of individuals in the popoulation
    pop_n = None

    # lb: lower bound of talent
    lb = None

    # ub: upper bound of talent
    ub = None

    # mu: average value of the talent distribution
    mu = None

    # std: standard deviation of the talent distribution
    std = None

    # le: chance for an individual to go through a lucky event
    le = 0.25

    # ue: chance for an individual to go through an unlucky event
    ue = 0.25

    # runs: number of runs to aggregate over
    runs = 100

    talent = None
    talent_index = None

    def __init__(self, struct: tvl_struct) -> None:

        np.set_printoptions(precision=3)

        # iter_n:
        self.iter_n = struct.iteration_number

        # pop_n: number of individuals in the popoulation
        self.pop_n = struct.population_number

        # lb: lower bound of talent
        self.lb = struct.talent_lower_bound

        # ub: upper bound of talent
        self.ub = struct.talent_upper_bound

        # mu: average value of the talent distribution
        self.mu = struct.talent_avg

        # std: standard deviation of the talent distribution
        self.std = struct.talent_std

        # le: chance for an individual to go through a lucky event
        self.le = 0.25

        # ue: chance for an individual to go through an unlucky event
        self.ue = 0.25

        # runs: number of runs to aggregate over
        self.runs = 100

        self.talent, self.talent_index = f.populate(
            self.pop_n, self.lb, self.ub, self.mu, self.std)

    def run(self):

        # Running the simulations:
        result = f.evolution(
            talent=self.talent,
            time=self.iter_n,
            unlucky_event=self.ue,
            lucky_event=self.le,
            history=True)

        final_pos = result[:, self.iter_n - 1]

        successful = final_pos[final_pos >= 0]

        succesful_index = np.argwhere(final_pos >= 0)

        succesful_talent = [self.talent[index[0]] for index in succesful_index]

        fig = plt.figure(layout='constrained', figsize=(10, 10))
        fig.set_figheight(6)
        fig.set_figwidth(14)

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # scatter plot
        ax1.set_xlim(right=self.ub, left=self.lb)
        ax1.set_ylim(top=final_pos.max(), bottom=final_pos.min())
        ax1.set_title('Final distribution')
        ax1.set_xlabel('Talent')
        ax1.set_ylabel('Position')

        patch_mean = mpatches.Patch(
            color='white', label=f'Mean position: {np.round(final_pos.mean(), decimals=2)}')

        patch_std = mpatches.Patch(
            color='white', label=f'Standard deviation: {np.round(final_pos.std(), decimals=2)}')
        ax1.legend(handles=[patch_mean, patch_std])

        ax1.scatter(self.talent, final_pos, s=4, c=final_pos)

        # ax1.scatter(self.talent, final_pos, s=4, cmap="cmap_name_r")

        # ax1.bar(x=succesful_talent, height=successful)

        # histogram
        ax2.set_xlim(right=final_pos.max(), left=final_pos.min())
        ax2.set_title('Histogram of final distribution')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Number of individuals')

        blue_patch = mpatches.Patch(color='white', label='The blue data')
        ax2.legend(handles=[blue_patch])

        ax2.hist(final_pos, bins=50)

        plt.show()

        return result, successful
