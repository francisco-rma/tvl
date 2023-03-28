import src.functions.core as f
from matplotlib import pyplot as plt
from matplotlib import animation as an
import numpy as np
import pandas as pd

np.set_printoptions(precision=3)

# iter_n: number of iterations to go through
iter_n = 10

# pop_n: number of individuals in the popoulation
pop_n = 2**2

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

talent, talent_index = f.populate(pop_n, lb, ub, mu, std)

pos = f.interactive_tvl(talent, iter_n, ue, le, True)

print(pos)
