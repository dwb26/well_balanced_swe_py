import numpy as np

g = 9.81
eps = 0.01

def flux(W):
	h, q = W
	soln = np.empty(2)
	soln[0] = q
	soln[1] = q ** 2 / h + 0.5 * g * h ** 2
	return soln


def compute_wave_speeds(W_L, W_R):

	# assert(W_L[0] >= 0 and W_R[0] >= 0), "Fluid heights are negative!! hL = {}, hR = {}".format(W_L[0], W_R[0])

	# Extract the velocities and the speed of sounds
	u_L = W_L[1] / W_L[0]; c_L = np.sqrt(g * W_L[0])
	u_R = W_R[1] / W_R[0]; c_R = np.sqrt(g * W_R[0])

	# Compute the left and right wave speeds
	lmbda_L = np.min([-np.abs(u_L) - c_L, -np.abs(u_R) - c_R, -eps])
	lmbda_R = np.max([np.abs(u_L) + c_L, np.abs(u_R) + c_R, eps])
	# print(lmbda_L, lmbda_R)
	return lmbda_L, lmbda_R


def Z_topography(x):
	"""
	Ultimately we will make this a random walk source term with (alpha, beta) (for example).
	"""
	return np.max([0, 0.5 - 2 * np.abs(x - 0.5)])


def h_cutoff(h_L, h_R, dx):

	C = 0.1
	if np.abs(h_R - h_L) <= C * dx:
		return h_R - h_L
	return np.sign(h_R - h_L) * C * dx


def estimate_source_mean(x_L, x_R, W_L, W_R, dx):

	h_L = W_L[0]; h_R = W_R[0]
	Z_diff = Z_topography(x_R) - Z_topography(x_L)
	h_C = h_cutoff(h_L, h_R, dx)
	a = -2 * g * Z_diff * h_L * h_R / (h_L + h_R)
	b = 0.5 * g * (h_C ** 3) / (h_L + h_R)
	return (a + b) / dx


def compute_intermediate_states(S_hat, dx, lmbda_L, lmbda_R, W_L, W_R):

	# Define the Harten-Lax-van Leer terms
	W_HLL = (lmbda_R * W_R - lmbda_L * W_L + flux(W_L) - flux(W_R)) / (lmbda_R - lmbda_L)
	h_HLL = W_HLL[0]
	q_HLL = W_HLL[1]
	assert(h_HLL >= 0), "h_HLL is not positive!, lmbda_L = {}, lmbda_R = {}, a = {}".format(lmbda_L, lmbda_R, a)

	# Compute the intermediate states
	h_L = W_L[0]; h_R = W_R[0]
	assert(W_L[0] >= 0 and W_R[0] >= 0), "Fluid heights are negative in the intermediate states!! hL = {}, hR = {}".format(W_L[0], W_R[0])
	q_tilde = q_HLL + S_hat * dx / (lmbda_R - lmbda_L)
	alpha = -q_tilde ** 2 / (h_L * h_R) + 0.5 * g * (h_L + h_R)
	h_L_star = h_HLL - lmbda_R * S_hat * dx / (alpha * (lmbda_R - lmbda_L))
	h_R_star = h_HLL - lmbda_L * S_hat * dx / (alpha * (lmbda_R - lmbda_L))
	q_star = q_tilde
	W_L_star = np.array([h_L_star, q_star])
	W_R_star = np.array([h_R_star, q_star])
	return W_L_star, W_R_star


def WB_solver(W, nx, dx, T_stop, xs):

	"""
	This is the well-balanced solver within which we apply the conservative formula. To do this we need to compute the time increment and the intermediate states.

	nx :: number of cells
	The space discretisation consists in cells (x_{i - 1/2}, x_{i + 1/2}) of volume dx and centered at x_i = x_{i - 1/2} + dx / 2
	"""

	W_int = np.zeros((nx + 2, 2))
	W_star_L = np.zeros((nx + 1, 2))
	W_star_R = np.zeros((nx + 1, 2))
	lmbda_neg = np.zeros(nx + 1)
	lmbda_pos = np.zeros(nx + 1)

	curve_data_f = open("curve_data.txt", "w")

	t = 0; n = 0
	while t < T_stop:

		# Assign the values of the ghost cells and write out the interior solution
		W[0] = W[1]
		W[nx + 1] = W[nx]
		for j in range(1, nx + 1):
			curve_data_f.write("{} ".format(W[j, 0]))
		curve_data_f.write("\n")
		for j in range(1, nx + 1):
			curve_data_f.write("{} ".format(W[j, 1]))
		curve_data_f.write("\n")

		# Compute the wave speeds from the left and right cell values
		for j in range(nx + 1):
			assert(W[j, 0] >= 0 and W[j + 1, 0] >= 0), "Fluid heights are negative!! hL = {}, hR = {}, n = {}".format(W[j, 0], W[j + 1, 0], n)
			lmbda_neg[j], lmbda_pos[j] = compute_wave_speeds(W[j], W[j + 1])

		# Compute the time step from the wave speeds and the CFL-like condition
		lmbda_max = np.max((-np.max(lmbda_neg), np.max(lmbda_pos)))
		assert (lmbda_max > 0)
		dt = dx / (2.0 * lmbda_max)
		assert(dt / dx <= 1), "CFL condition is violated! {}".format(dt / dx)

		for j in range(nx + 1):

			# Compute the mean estimators
			# S_hat = estimate_source_mean(xs[j], xs[j + 1], W[j], W[j + 1], dx)
			S_hat = 0.0

			# Compute the intermediate states
			W_star_L[j], W_star_R[j] = compute_intermediate_states(S_hat, dx, lmbda_neg[j], lmbda_pos[j], W[j], W[j + 1])

		for j in range(1, nx + 1):

			# Apply the conservative formula
			W_int[j] = W[j] - dt / dx * (lmbda_neg[j] * (W_star_L[j] - W[j]) - lmbda_pos[j - 1] * (W_star_R[j - 1] - W[j]))

		W = np.copy(W_int)
		t += dt
		n += 1

	# curve_data_f.write("\n")
	curve_data_f.write("{}".format(n))

	curve_data_f.close()














































