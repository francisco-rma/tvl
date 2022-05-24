from time import time
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

talent, t_i = f.population(pop_n, lb, ub, mu, std)

plt.hist(talent, bins=50)
plt.title('Histogram of talent distribution')
plt.show()

# le: chance for an individual to go through a lucky event
le = 0.03

# ue: chance for an individual to go through an unlucky event
ue = 0.03

final_cap = f.evolution(talent, iter_n, ue, le)

plt.hist(final_cap, bins=50)
plt.title('Histogram of the final capital')
plt.show()

plt.plot(talent, final_cap)
plt.title('Final capital vs talent')
plt.show()

print(np.max(final_cap), talent[np.argmax(final_cap)])
