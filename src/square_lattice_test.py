import functions as f
from matplotlib import pyplot as plt
from matplotlib import animation as an
import numpy as np
import pandas as pd

side = 4

grid = np.zeros((side, side))
listNeighbour = np.empty((side**2, 4))

f.list_neighbours(grid, listNeighbour, periodic_conditions=True)

grid = np.vander([1, 2, 3, 4], increasing=True)

lattice = pd.DataFrame(grid)
neighbours = pd.DataFrame(listNeighbour)
print(lattice)
print(neighbours)
