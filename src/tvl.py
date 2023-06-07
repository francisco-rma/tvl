import functions.core as f
from matplotlib import pyplot as plt
from matplotlib import animation as an
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

        # succesful_index = np.argwhere(final_pos >= 0)

        # succesful_talent = [self.talent[index[0]] for index in succesful_index]
        # plt.bar()
        # plt.bar(x=succesful_talent, height=successful)
        # plt.title('Successful individuals')
        # plt.xlim(right=self.ub, left=self.lb)
        # plt.xlabel('Talent')
        # plt.ylim(top=(successful.max() * 1.1), bottom=successful.min())
        # plt.ylabel('Position')
        # plt.legend(['Iterations: ' + str(self.iter_n)], loc='upper left')
        # plt.savefig('successful_individuals')
        # plt.show()

        # plt.plot(self.talent, final_pos)
        # plt.bar(x=self.talent, height=final_pos, width=0.01)
        plt.scatter(x=self.talent, y=final_pos, s=4)
        plt.title('Final distribution')
        plt.xlim(right=self.ub, left=self.lb)
        plt.xlabel('Talent')
        plt.ylim(top=final_pos.max(), bottom=final_pos.min())
        plt.ylabel('Position')
        plt.legend(['Iterations: ' + str(self.iter_n)], loc='upper left')
        plt.savefig('final_distribution')
        plt.cla()

        plt.hist(final_pos, bins=50)
        plt.title('Histogram of final distribution')
        plt.xlim(right=final_pos.max(), left=final_pos.min())
        plt.xlabel('Position')
        # plt.ylim(top=final_pos.max(), bottom=final_pos.min())
        plt.ylabel('Number of individuals')
        plt.legend(['Iterations: ' + str(self.iter_n)], loc='upper left')
        plt.savefig('final_distribution_histogram')
        plt.show()

        for i in final_pos:
            print(i)

        return result, successful
