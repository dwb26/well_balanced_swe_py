import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from solver import WB_solver
from scipy import stats


# -------------------------------------------------------------------------------------------------------------------- #
#
# Parameters
#
# -------------------------------------------------------------------------------------------------------------------- #
nx = 200														# Number of interior cells
space_left = 1.0; space_right = 26.0 							# Boundaries of space domain
T_stop = 1.0 													# Experiment stopping time
dx = (space_right - space_left) / (nx - 1) 						# Space increment
xs = np.linspace(space_left - dx, space_right + dx, nx + 2)		# Space variable
centre = 10; height = 0.2; width = 2							# Centre and height of topography bump
inc_start = centre + width + 10									# Beginning of sea bed incline incline to land
land_height = 0.2 												# Value at which we define land height
k = 5.5; theta = 2.0
length = 3
np.random.seed(13)

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
def gen_Z_topography(x, k, theta):
	return -25 * x ** (k - 1) * np.exp(-xs / theta) / (gamma(k) * theta ** k)


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
Z_top = gen_Z_topography(xs, k, theta)
h_drain = 0.5 - Z_drain
h_init = 10 * np.exp(-0.5 * ((xs - 0.5 * (space_left + space_right)) / 2) ** 2) + 0.0
h_coast = land_height - Z_coast
h_top = 2.0 * np.ones(nx + 2)

q_init = np.zeros(nx + 2)
q = np.zeros(nx + 2)
W_drain = np.empty((nx + 2, 2))
W_step = np.empty((nx + 2, 2))
W_coast = np.empty((nx + 2, 2))
W_top = np.empty((nx + 2, 2))
W = np.empty((nx + 2, 2))
W_drain[:, 0] = h_drain; W_drain[:, 1] = q
W_step[:, 0] = 10; W_step[:, 1] = gen_q_step(xs)
W[:, 0] = h_init; W[:, 1] = q_init
W_coast[:, 0] = h_coast; W_coast[:, 1] = q
W_top[:, 0] = h_init; W_top[:, 1] = 0.0

curve_data_f = open("curve_data.txt", "w")
Z_data_f = open("Z_data.txt", "w")
times_f = open("times.txt", "w")
hmm_data_f = open("hmm_data.txt", "w")
observations_f = open("observations.txt", "w")
# WB_solver(W, nx, dx, T_stop, xs, centre, height, inc_start, land_height, space_right)
# WB_solver(W_drain, nx, dx, T_stop, xs, k, theta, space_right)
# WB_solver(W_step, nx, dx, T_stop, xs, centre, height, inc_start), land_height, space_right)
# WB_solver(W_coast, nx, dx, T_stop, xs, centre, height, inc_start, land_height, space_right)

thetas = np.empty(length); ys = np.empty(length)

# HMM data generaration #
# --------------------- #
n = 0
sig_sd = 1.0; obs_sd = 0.1
for t in range(length):
	m, W_temp = WB_solver(W_top, nx, dx, T_stop, xs, k, theta, space_right, curve_data_f, Z_data_f, times_f, hmm_data_f)
	y = W_temp[nx, 0] + stats.norm.rvs(loc=0, scale=obs_sd)
	observations.write("{}".format(y))
	thetas[n] = theta; ys[n] = y
	W_top = np.copy(W_temp)
	theta += stats.norm.rvs(loc=0, scale=sig_sd)
	n += m


# Particle filtering #
# ------------------ #
N = 100

# Prior sample
xis = stats.norm.rvs(loc=thetas[0], scale=0.1, size=N)
phis = np.empty(N)
weights = np.empty(N)
x_hats = np.empty(length)

for n in range(length):

	y = ys[n]

	# Weight assignment #
	# ----------------- #
	for i in range(N):
		m, W_temp = WB_solver(W_top, nx, dx, T_stop, xs, k, xis[i], space_right, curve_data_f, Z_data_f, times_f, hmm_data_f)
		phis[i] = W_temp[nx, 0]
		weights[i] = stats.norm.pdf(x=y, loc=phis[i], scale=obs_sd)
		W_top = np.copy(W_temp)

	# Normalisation #
	# ------------- #
	weights /= np.sum(weights)
	x_hats[n] = np.sum(xis * weights)

	# Resample #
	# -------- #
	custm = stats.rv_discrete(values=(xis, weights))
	xis = xis[custm.rvs(size=N)]

	# Mutate #
	# ------ #
	noises = stats.norm.rvs(loc=0, scale=sig_sd, size=N)
	xis += noises


curve_data_f.write("{}".format(n))
curve_data_f.close()
Z_data_f.close()
times_f.close()
hmm_data_f.close()
observations_f.close()





























