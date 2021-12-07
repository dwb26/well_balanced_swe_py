import numpy as np

g = 9.81
eps = 1e-10
thresh = 1e-22

def flux(W):
	"""
	Flux function for the system. Checked and correct.
	"""
	h, q = W
	soln = np.empty(2)
	soln[0] = q
	if h > 0:
		soln[1] = q ** 2 / h + 0.5 * g * h ** 2
	else:
		soln[1] = 0.0
	return soln


def compute_forward_solution(dx, dt, nx, W, lmbda_neg, lmbda_pos, W_star_L, W_star_R):
	"""
	Find the next time increment solution using the conservative formula.
	"""
	incr_ratio = dt / dx
	for j in range(1, nx + 1):
		W[j] = W[j] - incr_ratio * (lmbda_neg[j] * (W_star_L[j] - W[j]) - lmbda_pos[j - 1] * (W_star_R[j - 1] - W[j]))
	return W


def output_data(curve_data_f, nx, W):

	for j in range(1, nx + 1):
		curve_data_f.write("{} ".format(W[j, 0]))
	curve_data_f.write("\n")
	for j in range(1, nx + 1):
		curve_data_f.write("{} ".format(W[j, 1]))
	curve_data_f.write("\n")


def Z_topography(x):
	"""
	For the topgraphy source term S(W, Z) = -g * h * del_x(Z), Z(x) describes the characteristic form of the topography.
	"""
	# f = 10 * np.max([0, 0.5 - 2 * np.abs(x - 0.5)])
	# f = 10 * np.max([0, 0.25 - 2 * np.abs(x - 0.25)]) + 10 * np.max([0, 0.75 - 2 * np.abs(x - 0.75)])
	# if x >= 0.5:
	# 	f = 1
	# else:
	# 	f = 0
	# f = 0
	f = np.max((0.2 - 0.05 * (x - 10) ** 2, 0))
	return f 


def h_cutoff(h_L, h_R, dx):
	"""
	Cutoff function to ensure that the scheme is well-balanced according to smooth steady states/ However, it means that the source term approximation does not vanish when the topography is flat. Checked and correct.
	"""

	# C = np.infty
	C = 1.35
	if np.abs(h_R - h_L) <= C * dx:
		return h_R - h_L
	return np.sign(h_R - h_L) * C * dx


def estimate_source_mean(x_L, x_R, W_L, W_R, dx):
	"""
	Estimate the space-time average of the topography source term.
	"""
	h_L = W_L[0]; h_R = W_R[0]
	if h_L == 0 and h_R == 0:
		return 0
	elif h_L == 0 or h_R == 0:
		Z_diff = Z_topography(x_R) - Z_topography(x_L)
		return - 0.5 * g * Z_diff * (h_L + h_R)
	else:
		Z_diff = Z_topography(x_R) - Z_topography(x_L)
		h_C = h_cutoff(h_L, h_R, dx)
		a = -2 * g * Z_diff * h_L * h_R / (h_L + h_R)
		b = 0.5 * g * (h_C ** 3) / (h_L + h_R)
		return a + b


def compute_timestep(lmbda_neg, lmbda_pos, dx):
	"""
	Compute the time increment using the maximum magnitude wavespeed and a CFL type condition.
	"""
	lmbda_max = np.max((np.max(-lmbda_neg), np.max(lmbda_pos)))
	assert (lmbda_max > 0), "lmbda_max is not positive! {}".format(lmbda_max)
	dt = dx / (2.0 * lmbda_max)
	assert(dt / dx <= 1), "CFL condition is violated! {}".format(dt / dx)
	return dt


def compute_wave_speeds(W):
	"""
	Compute the negative travelling (< 0) and positive travelling (> 0) wave speeds. Checked and correct.
	"""

	nx = len(W) - 2
	lmbda_neg = np.zeros(nx + 1)
	lmbda_pos = np.zeros(nx + 1)
	for j in range(nx + 1):

		W_L = W[j]; W_R = W[j + 1]
		h_L = W_L[0]; h_R = W_R[0]
		assert(h_L >= 0 and h_R >= 0), "Fluid heights are negative!! hL = {}, hR = {}".format(h_L, h_R)

		# Extract the velocities and the speed of sounds
		c_L = np.sqrt(g * h_L); c_R = np.sqrt(g * h_R) 
		if h_L > 0:
			u_L = W_L[1] / h_L
		else:
			u_L = 0
		
		if h_R > 0:
			u_R = W_R[1] / h_R
		else:
			u_R = 0

		# Compute the left and right wave speeds
		lmbda_L = np.min([-np.abs(u_L) - c_L, -np.abs(u_R) - c_R, -eps])
		lmbda_R = np.max([np.abs(u_L) + c_L, np.abs(u_R) + c_R, eps])
		lmbda_neg[j], lmbda_pos[j] = lmbda_L, lmbda_R

	return lmbda_neg, lmbda_pos


def perform_intermediate_height_corrections(h_L, h_L_star, h_R_star, h_R, h_HLL, lmbda_L, lmbda_R):
	"""
	Performs the height corrections (if required) to ensure positive intermediate water heights.
	"""
	# delta = np.min([h_L, h_R, h_HLL])
	delta = 0
	if h_L_star < delta:
		h_L_star = delta
		h_R_star = ((lmbda_R - lmbda_L) * h_HLL + lmbda_L * h_L_star) / lmbda_R
	if h_R_star < delta:
		h_R_star = delta
		h_L_star = (lmbda_R * h_R_star - (lmbda_R - lmbda_L) * h_HLL) / lmbda_L
	assert(h_L_star >= delta and h_R_star >= delta), "Something's wrong with the intermediate states!! {} {}".format(h_L_star, h_R_star)
	return h_L_star, h_R_star


def estimate_source_by_alpha(x_L, x_R, h_L, h_R, q_tilde, S_hat_dx):

	if h_L > thresh and h_R > thresh:
		S_hat_dx_by_alph = S_hat_dx / (-q_tilde ** 2 / (h_L * h_R) + 0.5 * g * (h_L + h_R))
	elif h_L <= thresh or h_R <= thresh:
		Z_diff = Z_topography(x_R) - Z_topography(x_L)
		S_hat_dx_by_alph = -Z_diff
	else:
		S_hat_dx_by_alph = 0
	return S_hat_dx_by_alph


def compute_intermediate_states(nx, dx, lmbda_neg, lmbda_pos, W, xs):

	W_star_L = np.zeros((nx + 1, 2))	# The left intermediate state associated to the Riemann problem boundary
	W_star_R = np.zeros((nx + 1, 2))	# The right intermediate state associated to the Riemann problem boundary

	for j in range(nx + 1):

		# Assign the left and right terms
		x_L = xs[j]; x_R = xs[j + 1]
		W_L = W[j]; W_R = W[j + 1]
		h_L = W_L[0]; h_R = W_R[0]
		lmbda_L = lmbda_neg[j]; lmbda_R = lmbda_pos[j]
		assert(h_L >= 0 and h_R >= 0), "Fluid heights are negative in the intermediate states!! hL = {}, hR = {}".format(h_L, h_R)

		# Define the Harten-Lax-van Leer terms
		W_HLL = (lmbda_R * W_R - lmbda_L * W_L + flux(W_L) - flux(W_R)) / (lmbda_R - lmbda_L)
		h_HLL = W_HLL[0]
		q_HLL = W_HLL[1]
		assert(h_HLL >= 0), "h_HLL is not positive!, lmbda_L = {}, lmbda_R = {}".format(lmbda_L, lmbda_R)

		# Compute the intermediate states
		S_hat_dx = estimate_source_mean(x_L, x_R, W_L, W_R, dx)
		q_tilde = q_HLL + S_hat_dx / (lmbda_R - lmbda_L)
		S_hat_dx_by_alph = estimate_source_by_alpha(x_L, x_R, h_L, h_R, q_tilde, S_hat_dx)

		h_L_star = h_HLL - lmbda_R * S_hat_dx_by_alph / (lmbda_R - lmbda_L)
		h_R_star = h_HLL - lmbda_L * S_hat_dx_by_alph / (lmbda_R - lmbda_L)

		h_L_star, h_R_star = perform_intermediate_height_corrections(h_L, h_L_star, h_R_star, h_R, h_HLL, lmbda_L, lmbda_R)
		q_star = q_tilde
		W_star_L[j] = np.array([h_L_star, q_star])
		W_star_R[j] = np.array([h_R_star, q_star])

	return W_star_L, W_star_R


def left_neumann_bc(t):
	# return np.array([3 * np.sin(25 * t), 0.0]) + 5
	return np.array([0.0, 0.0])


def right_neumann_bc(t):
	# return np.array([3 * np.sin(25 * t), 0.0])
	return np.array([0.0, 0.0])


def WB_solver(W, nx, dx, T_stop, xs):
	"""
	This is the well-balanced solver within which we apply the conservative formula. To do this we need to compute the time increment and the intermediate states.

	nx :: number of cells
	The space discretisation consists in cells (x_{i - 1/2}, x_{i + 1/2}) of volume dx and centered at x_i = x_{i - 1/2} + dx / 2
	"""

	W_star_L = np.zeros((nx + 1, 2))	# The left intermediate state associated to the Riemann problem boundary
	W_star_R = np.zeros((nx + 1, 2))	# The right intermediate state associated to the Riemann problem boundary
	lmbda_neg = np.zeros(nx + 1)		# The left moving wave speed associated to the Riemann problem boundary
	lmbda_pos = np.zeros(nx + 1)		# The right moving wave speed associated to the Riemann problem boundary

	curve_data_f = open("curve_data.txt", "w")
	steady_state_test_f = open("steady_state_test.txt", "w")

	t = 0; n = 0
	while t < T_stop:

		# Assign the values of the ghost cells and write out the interior solution
		W[0, 0] = W[1, 0] - 2 * dx * left_neumann_bc(t)[0]
		W[0, 1] = 0.0
		# W[nx + 1, 0] = W[nx, 0] + 2 * dx * right_neumann_bc(t)[0]
		# W[0] = W[1] - 2 * dx * left_neumann_bc(t)
		# W[nx + 1] = W[nx] + 2 * dx * right_neumann_bc(t)
		hN, qN = W[nx]
		if hN > 0:
			uN = qN / hN
		else:
			uN = 0
		hR = np.min((1 / (9 * g) * (uN + 2 * np.sqrt(g * hN)) ** 2, hN))
		qR = hR / 3.0 * (uN + 2 * np.sqrt(g * hN))
		W[nx + 1] = [hR, qR]
		output_data(curve_data_f, nx, W)
		for j in range(1, nx + 1):
			steady_state_test_f.write("{} ".format(W[j, 0] + Z_topography(xs[j])))
		steady_state_test_f.write("\n")

		# Find the next time increment solution
		lmbda_neg, lmbda_pos = compute_wave_speeds(W)
		dt = compute_timestep(lmbda_neg, lmbda_pos, dx)
		W_star_L, W_star_R = compute_intermediate_states(nx, dx, lmbda_neg, lmbda_pos, W, xs)
		W = compute_forward_solution(dx, dt, nx, W, lmbda_neg, lmbda_pos, W_star_L, W_star_R)

		t += dt; n += 1

	curve_data_f.write("{}".format(n))
	curve_data_f.close()
	steady_state_test_f.write("{}".format(n))
	steady_state_test_f.close()























