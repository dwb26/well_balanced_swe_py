import numpy as np
import matplotlib.pyplot as plt
from solver import WB_solver


# -------------------------------------------------------------------------------------------------------------------- #
#
# Parameters
#
# -------------------------------------------------------------------------------------------------------------------- #
nx = 200														# Number of interior cells
space_left = 0.0; space_right = 50.0 							# Boundaries of space domain
T_stop = 100.0 													# Experiment stopping time
dx = (space_right - space_left) / (nx - 1) 						# Space increment
xs = np.linspace(space_left - dx, space_right + dx, nx + 2)		# Space variable
centre = 10; height = 0.2; width = 2							# Centre and height of topography bump
inc_start = centre + width + 10									# Beginning of sea bed incline incline to land
land_height = 0.2 												# Value at which we define land height


# -------------------------------------------------------------------------------------------------------------------- #
#
# Topography configurations
#
# -------------------------------------------------------------------------------------------------------------------- #
def gen_Z_drain(x):
	"""
	The topography for the drain experiment over the topography bump
	"""
	Z = np.empty(len(x))
	for i in range(len(x)):
		Z[i] = np.max((height - 0.05 * (x[i] - centre) ** 2, 0))
	return Z
def gen_Z_step(x):
	"""
	The topography for the double rarefaction wave over a topography step
	"""
	Z = np.empty(len(x))
	a = 25 / 3; b = 25 / 2
	for i in range(len(x)):
		if x[i] < a or x[i] > b:
			Z[i] = 0
		else:
			Z[i] = 1
	return Z
def gen_Z_coast(x):
	"""
	The topography bump but with an incline to land level
	"""
	Z = np.empty(len(x))
	for i in range(len(x)):
		Z[i] = np.max((height - 0.05 * (x[i] - centre) ** 2, 0))
	m = 0.5 * np.pi / (space_right - inc_start)
	for i in range(len(x)):
		if x[i] >= inc_start:
			Z[i] = land_height * np.sin(m * (x[i] - inc_start))
	return Z


# -------------------------------------------------------------------------------------------------------------------- #
#
# Initial conditions
#
# -------------------------------------------------------------------------------------------------------------------- #
def gen_q_step(x):
	q = 350 * np.ones(len(x))
	for i in range(len(x)):
		if x[i] < 50 / 3:
			q[i] = -350
	return q
Z_drain = gen_Z_drain(xs)
Z_step = gen_Z_step(xs)
Z_coast = gen_Z_coast(xs)
h_drain = 0.1 + Z_drain
h_init =  np.exp(-0.5 * ((xs - 0.5 * (space_left + space_right)) / 5) ** 2) + 0.0
h_coast = land_height - Z_coast

q_init = np.zeros(nx + 2)
q = np.zeros(nx + 2)
W_drain = np.empty((nx + 2, 2))
W_step = np.empty((nx + 2, 2))
W_coast = np.empty((nx + 2, 2))
W = np.empty((nx + 2, 2))
W_drain[:, 0] = h_drain; W_drain[:, 1] = q
W_step[:, 0] = 10; W_step[:, 1] = gen_q_step(xs)
W[:, 0] = h_init; W[:, 1] = q_init
W_coast[:, 0] = h_coast; W_coast[:, 1] = q

# WB_solver(W, nx, dx, T_stop, xs, centre, height, inc_start, land_height, space_right)
# WB_solver(W_drain, nx, dx, T_stop, xs, centre, height, inc_start, land_height, space_right)
# WB_solver(W_step, nx, dx, T_stop, xs, centre, height, inc_start), land_height, space_right)
WB_solver(W_coast, nx, dx, T_stop, xs, centre, height, inc_start, land_height, space_right)



