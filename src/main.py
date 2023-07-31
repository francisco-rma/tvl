import time

from math import ceil
from tvl import tvl
from tvl import tvl_struct

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation as an
import matplotlib.patches as mpatches

from PIL import Image
from PIL.PngImagePlugin import PngInfo


def main():
    struct = tvl_struct()

    sim = tvl(struct)

    # sim.run()

    end = 10

    fig = plt.figure(layout='constrained', figsize=(10, 10))
    fig.set_figheight(6)
    fig.set_figwidth(14)
    fig.suptitle('Talent histogram of succesful individuals')

    plots = [200, 500, 800, 1000]

    start = time.time()

    for plot_index, iterations in enumerate(plots):
        i = 0
        while i < end:
            sample = np.zeros((end, len(sim.talent)))
            successful = []

            sim.set_iter_n(iterations)
            sample[i] = sim.run(many=True)
            test = np.ravel(sim.talent[np.argwhere(sample[i] > 0)])
            successful.extend(test)
            print(f'{i}, iterations: {sim.iter_n}')
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

    end = time.time()

    duration = end - start

    print(f'Elapsed time: {duration} seconds')

    # metadata = sim.generate_metadata()
    figname = 'tvl'
    fig.savefig(fname=figname)

    create_metadata(figname, simuation_instance=sim)

    plt.show()


def create_metadata(name, simuation_instance):
    targetImage = Image.open(f'{name}.png')

    metadata = PngInfo()
    metadata.add_text('runs', str(simuation_instance.iter_n))
    metadata.add_text('population_size', str(simuation_instance.pop_n))
    metadata.add_text('talent_lower_bound', str(simuation_instance.lb))
    metadata.add_text('talent_upper_bound', str(simuation_instance.ub))
    metadata.add_text('talent_mean', str(simuation_instance.mu))
    metadata.add_text('talent_standard_deviation', str(simuation_instance.std))
    metadata.add_text('lucky_event', str(simuation_instance.le))
    metadata.add_text('unlucky_event', str(simuation_instance.ue))

    targetImage.save(f'{name}.png', pnginfo=metadata)
    targetImage = Image.open(f'{name}.png')
    # print(targetImage.text)
    print(targetImage.info)


main()
