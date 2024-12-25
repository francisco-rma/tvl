from time import perf_counter, time
import functions as f
from matplotlib import pyplot as plt
from matplotlib import animation as an
import numpy as np

start = perf_counter()

np.set_printoptions(precision=3)

# iter_n: number of iterations to go through
iter_n = 500

# pop_n: number of individuals in the popoulation
pop_n = 1000

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

# runs: number of runs to aggregate over
runs = 100

talent, talent_index = f.populate(pop_n, lb, ub, mu, std)

# Running the simulations:
mst, msp, successful = f.many_runs(talent, iter_n, ue, le, runs)

# msc: Most Successful Capital (final capital of the most succesful individual)
msc = f.map_to_capital(msp)

# print(np.column_stack([msc, msp]))

print(f"Done simulating in {(perf_counter() - start)} seconds")

plt.hist(successful[:, 0], bins=100, range=(0, 1))
plt.title("Histogram of the talent of successful individuals")
plt.xlabel("Talent")
plt.ylabel("Number of occurences")
plt.legend(["Iterations: " + str(iter_n)], loc="upper left")
plt.savefig("successful_individuals")
plt.show()

print("\nMean position of successful individuals: ", np.mean(successful[:, 1]))
print(
    "Mean capital of successful individuals: ",
    np.mean(f.map_to_capital(successful[:, 1])),
)
print("Mean talent of successful individuals: ", np.mean(successful[:, 0]))

plt.clf()

plt.hist(mst, bins=100, range=(0, 1))
plt.title("Histogram of the talent of the most successful individual")
plt.xlabel("Talent")
plt.ylabel("Number of occurences")
plt.legend(["Iterations: " + str(iter_n)], loc="upper left")
plt.savefig("mst")
plt.show()

print("\nMean maximum position: ", np.mean(msp))
print("Mean maximum capital: ", np.mean(msc))
print("Mean associated talent: ", np.mean(mst))
