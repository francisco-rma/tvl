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

# le: chance for an individual to go through a lucky event
le = 0.03

# ue: chance for an individual to go through an unlucky event
ue = 0.03

# runs: number of runs to aggregate over
runs = 10000

# Creating a population with the desired parameters for talent distribution:
    # talent: array containing sorted values of talent

    # t_i: array containing the indices to the unsorted talent array, i.e:
        # talent[t_i[0]] is the talent of the first individual of the population
        # talent[t_i[j-1]] is the talent of the j-th individual of the population
        # talent[t_i[-1]] is the talent of the last individual of the population

talent, t_i = f.populate(pop_n, lb, ub, mu, std)

plt.hist(talent, bins=100, range=(0, 1))
plt.title('Talent distribution')
plt.xlabel('Talent')
plt.ylabel('Number of occurences')
plt.legend(['Pop size: ' + str(pop_n)], loc='upper left')
plt.show()

# Running the simulations:
mst, msp, successful = f.many_runs(talent, iter_n, ue, le, runs)

# msc: Most Successful Capital (final capital of the most succesful individual)
msc = f.cpt_map(msp)

plt.hist(successful[:, 0], bins=100, range=(0, 1))
plt.title('Histogram of the talent of successful individuals')
plt.xlabel('Talent')
plt.ylabel('Number of occurences')
plt.legend(['Iterations: ' + str(iter_n)], loc='upper left')
plt.show()

print('\nMean position of successful individuals: ', np.mean(successful[:, 1]))
print('Mean capital of successful individuals: ', np.mean(f.cpt_map(successful[:, 1])))
print('Mean talent of successful individuals: ', np.mean(successful[:, 0]))


plt.hist(mst, bins=100, range=(0, 1))
plt.title('Histogram of the talent of the most successful individual')
plt.xlabel('Talent')
plt.ylabel('Number of occurences')
plt.legend(['Iterations: ' + str(iter_n)], loc='upper left')
plt.show()

print('\nMean maximum position: ', np.mean(msp))
print('Mean maximum capital: ', np.mean(msc))
print('Mean associated talent: ', np.mean(mst))
