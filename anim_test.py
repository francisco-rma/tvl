import functions as f
from matplotlib import pyplot as plt, animation as animation, cm
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

talent, t_i = f.populate(pop_n, lb, ub, mu, std)

positions = f.evolution(talent, iter_n, ue, le, history=True)

fig1, ax1 = plt.subplots()

ax1.set_ylim(np.min(positions), np.max(positions))
ax1.set_xlim(talent.min(), talent.max())
ax1.set_title('Position distribution')
ax1.set_xlabel('Talent')
ax1.set_ylabel('Position')

bar = ax1.bar(np.linspace(0, 1, num=1000), positions[:, 0], width=0.001)

def bar_animate(i):
    for b, h in zip(bar, positions[:, i]):
        b.set_height(h)
    return


ani2 = animation.FuncAnimation(fig1, bar_animate, frames=iter_n + 1, blit=True, repeat=False)

plt.show()
