from time import time
import functions as f
from matplotlib import pyplot as plt
import numpy as np
iterations = 80

n = 1000

lb, ub = 0, 1

mu, sigma = 0.6, 0.1

talent, t_i = f.population(n, lb, ub, mu, sigma)

cap = f.evolution(talent, iterations, 0.03, 0.03)

final_cap = cap[:, iterations - 1]

plt.plot(talent, final_cap)
plt.show()

print(np.max(final_cap), talent[np.argmax(final_cap)])
