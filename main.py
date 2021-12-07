import numpy as np
import matplotlib.pyplot as plt
from solver import WB_solver, Z_topography


# -------------------------------------------------------------------------------------------------------------------- #
#
# Parameters
#
# -------------------------------------------------------------------------------------------------------------------- #
nx = 200
space_left = 0.0; space_right = 25.0
T_stop = 100.0
# T_stop = 0.25
# T_stop = 0.05
dx = (space_right - space_left) / (nx - 1)
xs = np.linspace(space_left - dx, space_right + dx, nx + 2)

def gen_Z_c(x):
	foo = np.empty(len(x))
	for i in range(len(x)):
		foo[i] = np.max([0, 0.5 - 2 * np.abs(x[i] - 0.5)])
	return -foo
def gen_Z_d(x):
	return x >= 0.5
def gen_Z_drain(x):
	Z = np.empty(len(x))
	for i in range(len(x)):
		Z[i] = np.max((0.2 - 0.05 * (x[i] - 10) ** 2, 0))
	return Z

Z_c = gen_Z_c(xs)
Z_d = gen_Z_d(xs)
Z_drain = gen_Z_drain(xs)
h_c = 1 - Z_c
h_d = 1 - Z_d
h_drain = 0.5 - Z_drain
h_init = 20 * np.exp(- 0.5 * ((xs - 0.5 * (space_left + space_right)) / 0.01) ** 2) + 20
q_init = np.zeros(nx + 2)
q = np.zeros(nx + 2)

W_c = np.empty((nx + 2, 2))
W_d = np.empty((nx + 2, 2))
W_drain = np.empty((nx + 2, 2))
W = np.empty((nx + 2, 2))
W_c[:, 0] = h_c; W_c[:, 1] = q * h_c
W_d[:, 0] = h_d; W_d[:, 1] = q * h_d
W_drain[:, 0] = h_drain; W_drain[:, 1] = q
W[:, 0] = h_init; W[:, 1] = q_init

WB_solver(W_drain, nx, dx, T_stop, xs)