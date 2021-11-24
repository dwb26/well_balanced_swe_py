import numpy as np
import matplotlib.pyplot as plt
from solver import WB_solver


# -------------------------------------------------------------------------------------------------------------------- #
#
# Parameters
#
# -------------------------------------------------------------------------------------------------------------------- #
nx = 250
space_left = -10.0; space_right = 10.0
T_stop = 2.5
dx = (space_right - space_left) / (nx - 1)
xs = np.linspace(space_left - dx, space_right + dx, nx + 2)
h_init = 10 * np.exp(- 0.5 * ((xs - 0.0) / 0.1) ** 2) + 5
q_init = np.zeros(nx + 2)
W = np.empty((nx + 2, 2))
W[:, 0] = h_init; W[:, 1] = q_init
WB_solver(W, nx, dx, T_stop, xs)