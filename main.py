import functions as f
from matplotlib import pyplot as plt
import numpy as np

np.set_printoptions(precision=3)

# iter_n: number of iterations to go through

iter_n = 80

# pop_n: number of individuals in the popoulation

pop_n = 1000

# lb: lower bound of talent
# ub: upper bound of talent

lb, ub = 0, 1

# mu: average value of the talent distribution
# std: standard deviation of the talent distribution

mu, std = 0.6, 0.1

# Creating a population with the desired parameters for talent distribution:

    # talent: array containing sorted values of talent

    # t_i: array containing the indices to the unsorted talent array, i.e:

        # talent[t_i[0]] is the talent of the first individual of the population

        # talent[t_i[j-1]] is the talent of the j-th individual of the population

        # talent[t_i[-1]] is the talent of the last individual of the population

talent, t_i = f.populate(pop_n, lb, ub, mu, std)

# le: chance for an individual to go through a lucky event
le = 0.03

# ue: chance for an individual to go through an unlucky event
ue = 0.03

# runs: number of runs to aggregate over
runs = 10000

# Initialize arrays to hold the position and the talent for the most succesful individual of each run:

# mst: Most Successful Talent (talent of the most succesful individual)
mst = np.empty(runs)

# msp: Most Successful Position (final position of the most succesful individual)
msp = np.empty(runs)

# Create an array to store values of talent and position for those who were overall successful:
successful = np.zeros((1, 2))

# Perform the simulations:
for i in range(runs):

    final_pos = f.evolution(talent, iter_n, ue, le)

    successful_per_run = np.column_stack((talent[final_pos > 0], final_pos[final_pos > 0]))

    successful = np.concatenate((successful, successful_per_run))

    mst[i] = talent[np.argmax(final_pos)]

    msp[i] = np.max(final_pos)

# msc: Most Successful Capital (final capital of the most succesful individual)
msc = f.cpt_map(msp)

plt.hist(successful[1:, 0], bins=100, range=(0, 1))
plt.title('Histogram of the talent of successful individuals')
plt.xlabel('Talent')
plt.ylabel('Number of occurences')
plt.text(0.0, 80000, 'Iterations: ' + str(iter_n))
plt.show()

print('\nMean position of successful individuals: ', np.mean(successful[1:, 1]))
print('Mean capital of successful individuals: ', np.mean(f.cpt_map(successful[:, 1])))
print('Mean talent of successful individuals: ', np.mean(successful[1:, 0]))


plt.hist(mst, bins=100, range=(0, 1))
plt.title('Histogram of the talent of the most successful individual')
plt.xlabel('Talent')
plt.ylabel('Number of occurences')
plt.text(0.0, 400, 'Iterations: ' + str(iter_n))
plt.show()

print('\nMean maximum position: ', np.mean(msp))
print('Mean maximum capital: ', np.mean(msc))
print('Mean associated talent: ', np.mean(mst))
