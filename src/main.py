from math import ceil
from sklearn.random_projection import sample_without_replacement
from tvl import tvl
from tvl import tvl_struct

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as an
import matplotlib.patches as mpatches


def main():
    struct = tvl_struct()

    sim = tvl(struct)

    # sim.run()

    end = 100

    fig = plt.figure(layout='constrained', figsize=(10, 10))
    fig.set_figheight(6)
    fig.set_figwidth(14)
    fig.suptitle('Talent histogram of succesful individuals')

    plots = [200, 500, 800, 1000]

    for plot_index, iterations in enumerate(plots):
        i = 0
        while i < end:
            sample = np.zeros((end, len(sim.talent)))
            successful = []

            sim.set_iter_n(iterations)
            sample[i] = sim.run(many=True)
            test = np.ravel(sim.talent[np.argwhere(sample[i] > 0)])
            successful.extend(test)
            # print(i)
            i += 1

        ax = fig.add_subplot(int(f'{ceil(len(plots)/2)}{2}{plot_index + 1}'))
        ax.set_xlim(right=1, left=0)
        # ax.set_title('Talent histogram of succesful individuals')
        ax.set_xlabel('Talent')
        ax.set_ylabel('Number of successes')

        patch_iter_n = mpatches.Patch(
            color='white', label=f'Iterations: {sim.iter_n}')

        patch_avg = mpatches.Patch(
            color='white', label=f'Average talent: {np.round(np.mean(successful), 2)}')

        ax.legend(handles=[patch_iter_n, patch_avg])

        bins = np.round(np.linspace(0.0, 1.0, 21), 2)
        ticks = np.round(np.linspace(0.0, 1.0, 21), 1)

        ax.hist(successful, bins=bins, rwidth=1, edgecolor='black')

        ax.set_xticks(ticks)

    plt.show()


main()
