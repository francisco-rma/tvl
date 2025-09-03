from time import perf_counter
import functions as f
import numpy as np


np.set_printoptions(precision=3)

# iter_n: number of iterations to go through
iter_n = 500

# size: size of population matrix. Total population will be size^2
size = 4

# lb: lower bound of talent
# ub: upper bound of talent
lb, ub = 0, 1

# mu: average value of the talent distribution
# std: standard deviation of the talent distribution
mu, std = 0.6, 0.1

# le: chance for an individual to go through a lucky event
le = 0.25

# ue: chance for an individual to go through an unlucky event
ue = 0.25

talent = f.interactive_populate(size, lb, ub, mu, std)

print(talent.shape)

pos = f.interactive_tvl(talent, iter_n, ue, le, False)
