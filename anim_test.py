import functions as f
from matplotlib import pyplot as plt, animation as animation, cm
import numpy as np

np.set_printoptions(precision=3)

# iter_n: number of iterations to go through
iter_n = 8000

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

talent, t_i = f.populate(pop_n, lb, ub, mu, std)

positions = f.evolution(talent, iter_n, ue, le, history=True)

fig1, ax1 = plt.subplots()

ax1.set_facecolor('lavender')

ax1.set_ylim(np.min(positions), np.max(positions))
ax1.set_xlim(np.min(talent), np.max(talent))
ax1.set_title('Position distribution')
ax1.set_xlabel('Talent')
ax1.set_ylabel('Position')

# Plot reference black line at position = 0:
ref_line, = ax1.plot(talent, np.zeros((len(talent))),
                     color='black')

# Bar plot of position for each individual
bar = ax1.bar(talent,
              positions[:, 0],
              width=0.001,
              alpha=0.5)

# Line plot of <position(time)> x talent:
line1, = ax1.plot(talent, -0 * le * (np.ones((pop_n)) - talent),
                  color='black',
                  linestyle='-',
                  alpha=0.7)

# Scatter of position x talent
line2, = ax1.plot(talent, -0 * le * (np.ones((pop_n)) - talent),
                  color='fuchsia',
                  marker='.',
                  linestyle='none',
                  markersize=2,
                  alpha=0.7)


def animate(i):
    for b, h in zip(bar, positions[:, i]):
        b.set_height(h)

    line1.set_ydata(-i * le * (np.ones((pop_n)) - talent))
    line2.set_ydata(positions[:, i])
    return


ani1 = animation.FuncAnimation(fig1,
                               animate,
                               frames=iter_n + 1,
                               blit=True, repeat=False,
                               interval=1)

plt.show()
