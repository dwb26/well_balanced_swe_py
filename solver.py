import numpy as np
import matplotlib.pyplot as plt

g = 9.81
glob_eps = 1e-10
ZERO_THRESH = 1e-15
C_CUTOFF = 1.35
m_glob = 0.5
M_glob = 1e-10
# C_CUTOFF = 1
# m_glob = 1e-10
# M_glob = 1e+04


def Z_topography(x, centre, height, inc_start, land_height, space_right):
	"""
	For the topgraphy source term S(W, Z) = -g * h * del_x(Z), Z(x) describes the characteristic form of the topography.
	"""
	Z = np.max((height - 0.05 * (x - centre) ** 2, 0))
	m = 0.5 * np.pi / (space_right - inc_start)
	if x > inc_start:
		Z = land_height * np.sin(m * (x - inc_start))
	# a = 25 / 3; b = 25 / 2
	# if x < a or x > b:
		# f = 0
	# else:
		# f = 1
	return Z


def prescribe_left_bcs(W, dx, t):
	# W[0, 0] = W[1, 0]; W[0, 1] = 0.0
	# scaler = 2 * np.pi
	scaler = 1
	# W[0, 0] = W[1, 0] - 2 * dx * np.sin(scaler * t); W[0, 1] = 0.0
	# W[0, 0] = W[1, 0] - 2 * dx * np.exp(-scaler * t ** 2); W[0, 1] = 0.0
	W[0, 0] = W[1, 0] + np.exp(-scaler * (t % 10) ** 2); W[0, 1] = 0.0
	return W


def prescribe_right_bcs(W, nx):
	# W[nx + 1, 0] = W[nx, 0]; W[nx + 1, 1] = 0.0
	W[nx + 1, 0] = 0.0; W[nx + 1, 1] = 0.0
	# hN = W[nx, 0]
	# if hN <= ZERO_THRESH:
		# uN = 0
	# else:
		# uN = W[nx, 1] / hN
	# hR = np.min((1 / (9 * g) * (uN + 2 * np.sqrt(g * hN)) ** 2, hN))
	# qR = hR / 3.0 * (uN + 2 * np.sqrt(g * hN))
	# W[nx + 1] = hR, qR
	# W[nx + 1] = W[nx]
	return W


def h_cutoff(h_L, h_R, dx):
	"""
	Cutoff function to ensure that the scheme is well-balanced according to smooth steady states. However, it means that the source term approximation does not vanish when the topography is flat.
	"""
	if np.abs(h_R - h_L) <= C_CUTOFF * dx:
		return h_R - h_L
	return np.sign(h_R - h_L) * C_CUTOFF * dx


def output_data(curve_data_f, times_f, t, nx, W, data, xs, h_init, Z_arr):

	for j in range(1, nx + 1):
		curve_data_f.write("{} ".format(W[j, 0] + Z_arr[j]))
		# curve_data_f.write("{} ".format(W[j, 0]))
	curve_data_f.write("\n")
	for j in range(1, nx + 1):
		curve_data_f.write("{} ".format(W[j, 1]))
	curve_data_f.write("\n")
	times_f.write("{} ".format(t))


def zero_cutoff(W_L, W_R):

	h_L = W_L[0]; h_R = W_R[0]
	if (h_L < ZERO_THRESH and h_R < ZERO_THRESH):
		W_L = np.array([0, 0]); W_R = np.array([0, 0])
	elif h_L < ZERO_THRESH:
		W_L = np.array([0, 0])
	elif h_R < ZERO_THRESH:
		W_R = np.array([0, 0])
	return W_L, W_R


def compute_forward_solution(W, dx, dt, nx, lmbda_neg, lmbda_pos, W_L_stars, W_R_stars):
	"""
	Find the next time increment solution using the conservative formula.
	"""
	incr_ratio = dt / dx
	W_forward = np.copy(W)
	for j in range(1, nx + 1):
		W_forward[j] = W[j] - incr_ratio * (lmbda_neg[j] * (W_L_stars[j] - W[j]) - lmbda_pos[j - 1] * (W_R_stars[j - 1] - W[j]))
	return W_forward


def flux(W):
	"""
	Flux function for the system. This is used in the computation of the HLL terms
	"""
	h, q = W
	soln = np.zeros(2)
	if h >= ZERO_THRESH:
		soln[0] = q
		soln[1] = q ** 2 / h + 0.5 * g * h ** 2
	return soln


def minmod(a, b):
	if (np.abs(a) < np.abs(b)) and (a * b > 0):
		return a
	if (np.abs(b) < np.abs(a)) and (a * b > 0):
		return b
	return 0


def compute_timestep(lmbda_neg, lmbda_pos, dx):
	"""
	Compute the time increment using the maximum magnitude wavespeed and a CFL type condition.
	"""
	lmbda_max = np.max((np.max(-lmbda_neg), np.max(lmbda_pos)))
	assert (lmbda_max > 0), "lmbda_max is not positive! {}".format(lmbda_max)
	dt = dx / (2.0 * lmbda_max)
	assert(dt / dx <= 1), "CFL condition is violated! {}".format(dt / dx)
	return dt


def compute_wave_speeds(W_L, W_R):
	"""
	Compute the negative travelling (< 0) and positive travelling (> 0) wave speeds.
	"""
	h_L = W_L[0]; h_R = W_R[0]
	assert(h_L >= 0 and h_R >= 0), "Fluid heights are negative!! hL = {}, hR = {}".format(h_L, h_R)

	# Extract the velocities and the speed of sounds
	if h_L == 0:
		c_L = 0
		u_L = 0
	else:
		c_L = np.sqrt(g * h_L)
		u_L = W_L[1] / h_L
	if h_R == 0:
		c_R = 0
		u_R = 0
	else:
		c_R = np.sqrt(g * h_R)
		u_R = W_R[1] / h_R

	# Compute the left and right wave speeds
	lmbda_L = np.min([-np.abs(u_L) - c_L, -np.abs(u_R) - c_R, -glob_eps])
	lmbda_R = np.max([np.abs(u_L) + c_L, np.abs(u_R) + c_R, glob_eps])
	assert lmbda_L < 0 and lmbda_R > 0, "lmbda issue"
	return lmbda_L, lmbda_R


def compute_W_HLL(lmbda_L, lmbda_R, W_L, W_R):
	W_HLL = (lmbda_R * W_R - lmbda_L * W_L + flux(W_L) - flux(W_R)) / (lmbda_R - lmbda_L)
	assert(W_HLL[0] >= 0), "h_HLL is negative! h_HLL = {}".format(W_HLL[0])
	return W_HLL


def compute_S_dx(h_L, h_R, Z_L, Z_R, dx):

	Z_diff = Z_R - Z_L
	if h_L == 0 and h_R == 0:
		return 0
	elif h_L == 0 or h_R == 0:
		return -0.5 * g * Z_diff * (h_L + h_R)
	else:
		h_C = h_cutoff(h_L, h_R, dx)
		a = -g * Z_diff * 2 * h_L * h_R / (h_L + h_R)
		b = 0.5 * g * h_C ** 3 / (h_L + h_R)
		return a + b


def compute_q_star(q_HLL, S_dx, lmbda_L, lmbda_R):
	"""
	This is the discharge component of the intermediate states. Note that there is only one since we set q_L* = q_R* = q_*.
	"""
	return q_HLL + S_dx / (lmbda_R - lmbda_L)


def compute_S_dx_by_alph(h_L, h_R, Z_L, Z_R, S_dx, q_star):

	if h_L == 0 and h_R == 0:
		return 0
	elif h_L == 0 or h_R == 0:
		return -(Z_R - Z_L)
	else:
		alph = -(q_star ** 2) / (h_L * h_R) + 0.5 * g * (h_L + h_R)
		return S_dx / alph


def compute_h_stars(h_HLL, lmbda_L, lmbda_R, S_dx_by_alph, h_L, h_R):
	"""
	Computes the left and right intermediate height states.
	"""
	eps = np.min([h_L, h_R, h_HLL])
	lmbda_jump = lmbda_R - lmbda_L
	lmbda_R_rat = lmbda_R / lmbda_L
	lmbda_L_rat = lmbda_L / lmbda_R
	assert (eps >= 0), "Heights are negative via the eps parameter! eps = {}".format(eps)

	# Compute the left intermediate height state
	a = np.max([h_HLL - lmbda_R * S_dx_by_alph / lmbda_jump, eps])
	b = (1 - lmbda_R_rat) * h_HLL + lmbda_R_rat * eps
	h_L_star = np.min([a, b])

	# Compute the right intermediate state
	a = np.max([h_HLL - lmbda_L * S_dx_by_alph / lmbda_jump, eps])
	b = (1 - lmbda_L_rat) * h_HLL + lmbda_L_rat * eps
	h_R_star = np.min([a, b])

	diff = lmbda_R * h_R_star - lmbda_L * h_L_star - lmbda_jump * h_HLL 
	assert (diff <= 1e-10), "Eqn 3.17a not satisfied {}".format(diff)
	assert (h_L_star >= eps and h_R_star >= eps), "H_L_star and H_R_star should be < eps {} {} {}".format(h_L_star, h_R_star, eps)

	return h_L_star, h_R_star


def compute_theta(W, Z_arr, dx, j):
	"""
	This is a parameter that describes the convexity of the reconstruction. Depends only on h and q.
	"""
	W_L = W[j - 1]; W_M = W[j]; W_R = W[j + 1]
	Z_L = Z_arr[j - 1]; Z_M = Z_arr[j]; Z_R = Z_arr[j + 1]
	h_L, q_L = W_L
	h_M, q_M = W_M
	h_R, q_R = W_R
	S_dx_LM = compute_S_dx(h_L, h_M, Z_L, Z_M, dx)
	S_dx_MR = compute_S_dx(h_M, h_R, Z_M, Z_R, dx)

	# Compute the boundary psi terms
	if h_L < ZERO_THRESH:
		if h_M < ZERO_THRESH:
			psi_LM = -S_dx_LM
		else:
			psi_LM = q_M ** 2 / h_M + 0.5 * g * h_M ** 2 - S_dx_LM
	elif h_M < ZERO_THRESH:
		if h_L < ZERO_THRESH:
			psi_LM = -S_dx_LM
		else:
			psi_LM = -q_L ** 2 / h_L - 0.5 * g * h_L ** 2 - S_dx_LM
	else:
		psi_LM = q_M ** 2 / h_M - q_L ** 2 / h_L + 0.5 * g * (h_M ** 2 - h_L ** 2) - S_dx_LM

	if h_M < ZERO_THRESH:
		if h_R < ZERO_THRESH:
			psi_MR = -S_dx_MR
		else:
			psi_MR = q_R ** 2 / h_R + 0.5 * g * h_R ** 2 - S_dx_MR
	elif h_R < ZERO_THRESH:
		if h_M < ZERO_THRESH:
			psi_MR = -S_dx_MR
		else:
			psi_MR = -q_M ** 2 / h_M - 0.5 * g * h_M ** 2 - S_dx_MR
	else:	
		psi_MR = q_R ** 2 / h_R - q_M ** 2 / h_M + 0.5 * g * (h_R ** 2 - h_M ** 2) - S_dx_MR

	# Compute the centred-cell varphi term
	a = np.linalg.norm(np.array([[q_M - q_L], [psi_LM]]), ord=2)
	b = np.linalg.norm(np.array([[q_R - q_M], [psi_MR]]), ord=2)
	varphi = a + b

	# Compute the centred-cell theta term
	if varphi < m_glob * dx:
		theta = 0
	elif varphi <= M_glob * dx:
		theta = (varphi - m_glob * dx) / (M_glob * dx - m_glob * dx)
	else:
		theta = 1

	assert (theta >= 0 and theta <= 1), "Theta should be between 0 and 1!! {}".format(theta)
	return theta


def compute_slope(W, dx, j):
	"""
	Computes the slope of the linear reconstruction corresponding to the same cell w_M is considered on.
	"""
	W_L = W[j - 1]; W_M = W[j]; W_R = W[j + 1]
	return minmod((W_R - W_M) / dx, (W_M - W_L) / dx)


def numerical_flux(W_L, W_R, lmbda_L, lmbda_R, W_L_star, W_R_star):

	a = 0.5 * (flux(W_L) + flux(W_R))
	b = 0.5 * lmbda_L * (W_L_star - W_L)
	c = 0.5 * lmbda_R * (W_R_star - W_R)
	return a + b + c


def g_eval(W_LM, W_ML, W_MR, W_RM, Z_LM, Z_ML, Z_MR, Z_RM, lmbda_neg, lmbda_pos, W_L_stars, W_R_stars, dx, j):
	
	h_LM = W_LM[0]; h_ML = W_ML[0]; h_MR = W_MR[0]; h_RM = W_RM[0]

	# Produce the flux approximations
	flux_MR = numerical_flux(W_MR, W_RM, lmbda_neg[j], lmbda_pos[j], W_L_stars[j], W_R_stars[j])
	flux_LM = numerical_flux(W_LM, W_ML, lmbda_neg[j - 1], lmbda_pos[j - 1], W_L_stars[j - 1], W_R_stars[j - 1])

	# Produce the source term approximations
	S_MR = compute_S_dx(h_MR, h_RM, Z_MR, Z_RM, dx) / dx
	S_LM = compute_S_dx(h_LM, h_ML, Z_LM, Z_ML, dx) / dx
	s_MR = np.array([0, S_MR])
	s_LM = np.array([0, S_LM])

	return -1 / dx * (flux_MR - flux_LM) + 0.5 * (s_MR + s_LM)


def MUSCL_reconstruction(w, slopes, thetas, dx, j):

	w_LM = w[j - 1] + 0.5 * dx * slopes[j - 1] * thetas[j - 1]
	w_ML = w[j] - 0.5 * dx * slopes[j] * thetas[j]
	w_MR = w[j] + 0.5 * dx * slopes[j] * thetas[j]
	w_RM = w[j + 1] - 0.5 * dx * slopes[j + 1] * thetas[j + 1]

	return w_LM, w_ML, w_MR, w_RM


def MUSCL_forward_solution(W, dx, dt, nx, Z_arr, lmbda_neg, lmbda_pos, W_L_stars, W_R_stars):

	h = np.copy(W[:, 0]); q = np.copy(W[:, 1]); h_plus_Z = np.copy(h + Z_arr)
	thetas = np.zeros(nx + 2)
	slopes = np.zeros((3, nx + 2))	# (h, q, h + Z)
	W_tilde = np.copy(W)
	gs = np.zeros((nx + 2, 2))

	for j in range(1, nx + 1):

		# Compute the theta parameters for the convex MUSCL reconstruction
		thetas[j] = compute_theta(W, Z_arr, dx, j)

		# Compute the respective slopes
		slopes[0, j] = compute_slope(h, dx, j)
		slopes[1, j] = compute_slope(q, dx, j)
		slopes[2, j] = compute_slope(h_plus_Z, dx, j)

	# Apply the forward formula
	for j in range(1, nx + 1):

		# Height MUSCL reconstruction
		h_LM, h_ML, h_MR, h_RM = MUSCL_reconstruction(h, slopes[0], thetas, dx, j)

		# Discharge MUSCL reconstruction
		q_LM, q_ML, q_MR, q_RM = MUSCL_reconstruction(q, slopes[1], thetas, dx, j)

		# Topography MUSCL reconstruction
		Z_LM, Z_ML, Z_MR, Z_RM = MUSCL_reconstruction(h_plus_Z, slopes[2], thetas, dx, j)
		Z_LM -= h_LM; Z_ML -= h_ML; Z_MR -= h_MR; Z_RM -= h_RM

		# Vector entries
		W_LM = np.array([h_LM, q_LM])
		W_ML = np.array([h_ML, q_ML])
		W_MR = np.array([h_MR, q_MR])
		W_RM = np.array([h_RM, q_RM])

		# Intermediate state
		gs[j] = g_eval(W_LM, W_ML, W_MR, W_RM, Z_LM, Z_ML, Z_MR, Z_RM, lmbda_neg, lmbda_pos, W_L_stars, W_R_stars, dx, j)
		W_tilde[j] = W[j] + dt * gs[j]


	# Do the same for the tilde quantities for Heun's stable time method #
	# ----------------------------------------------------------------- #
	h_tilde = np.copy(W_tilde[:, 0]); q_tilde = np.copy(W_tilde[:, 1]); h_plus_Z_tilde = np.copy(h_tilde + Z_arr)
	for j in range(1, nx + 1):

		# Compute the theta parameters for the convex MUSCL reconstruction
		thetas[j] = compute_theta(W_tilde, Z_arr, dx, j)

		# Compute the respective slopes
		slopes[0, j] = compute_slope(h_tilde, dx, j)
		slopes[1, j] = compute_slope(q_tilde, dx, j)
		slopes[2, j] = compute_slope(h_plus_Z_tilde, dx, j)

	# Apply the forward formula
	for j in range(1, nx + 1):

		# Height MUSCL reconstruction
		h_LM, h_ML, h_MR, h_RM = MUSCL_reconstruction(h_tilde, slopes[0], thetas, dx, j)

		# Discharge MUSCL reconstruction
		q_LM, q_ML, q_MR, q_RM = MUSCL_reconstruction(q_tilde, slopes[1], thetas, dx, j)

		# Topography MUSCL reconstruction
		Z_LM, Z_ML, Z_MR, Z_RM = MUSCL_reconstruction(h_plus_Z_tilde, slopes[2], thetas, dx, j)
		Z_LM -= h_LM; Z_ML -= h_ML; Z_MR -= h_MR; Z_RM -= h_RM

		# Vector entries
		W_LM = np.array([h_LM, q_LM])
		W_ML = np.array([h_ML, q_ML])
		W_MR = np.array([h_MR, q_MR])
		W_RM = np.array([h_RM, q_RM])

		# Intermediate state
		g_tilde = g_eval(W_LM, W_ML, W_MR, W_RM, Z_LM, Z_ML, Z_MR, Z_RM, lmbda_neg, lmbda_pos, W_L_stars, W_R_stars, dx, j)
		W[j] = W[j] + 0.5 * dt * (gs[j] + g_tilde)

	return W_tilde


def WB_solver(W, nx, dx, T_stop, xs, centre, height, inc_start, land_height, space_right):
	"""
	This is the well-balanced solver within which we apply the conservative formula. To do this we need to compute the time increment, the wave speeds and the intermediate states at each time iterate. We apply the conservative formula until T_stop is exceeded.

	nx :: number of cells
	The space discretisation consists in cells (x_{i - 1/2}, x_{i + 1/2}) of volume dx and centered at x_i = x_{i - 1/2} + dx / 2
	"""

	W_L_stars = np.zeros((nx + 1, 2))	# The left intermediate states associated to each boundary
	W_R_stars = np.zeros((nx + 1, 2))	# The right intermediate states associated to each boundary
	lmbda_neg = np.zeros(nx + 1)		# The left moving wave speed associated to each boundary
	lmbda_pos = np.zeros(nx + 1)		# The right moving wave speed associated to each boundary
	Z_arr = np.empty(nx + 2)			# Array of the characteristic form of the topography
	for j in range(nx + 2):
		Z_arr[j] = Z_topography(xs[j], centre, height, inc_start, land_height, space_right)

	curve_data_f = open("curve_data.txt", "w")
	times_f = open("times.txt", "w")
	data = open("steady_state_test.txt", "w")
	h_init = np.copy(W[1:nx + 1, 0])
	W_forward = np.copy(W)

	t = 0; n = 0
	while (t <= T_stop):
		
		W = np.copy(W_forward)

		# Assign the values of the ghost cells wrt the Neumann boundary conditions
		print(t)
		W = prescribe_left_bcs(W, dx, t)
		W = prescribe_right_bcs(W, nx)
		output_data(curve_data_f, times_f, t, nx, W, data, xs, h_init, Z_arr)
		W_forward = np.copy(W)

		# Iterate along each boundary (Riemann problem)
		for j in range(nx + 1):

			# Define the Riemann problem left and right states
			W_L, W_R = zero_cutoff(W[j], W[j + 1])
			Z_L = Z_arr[j]; Z_R = Z_arr[j + 1]; h_L = W_L[0]; h_R = W_R[0]; q_L = W_L[1]; q_R = W_R[1]

			# Compute the wave speeds and the HLL terms
			lmbda_neg[j], lmbda_pos[j] = compute_wave_speeds(W_L, W_R)
			lmbda_L = lmbda_neg[j]; lmbda_R = lmbda_pos[j]; lmbda_jump = lmbda_R - lmbda_L
			h_HLL, q_HLL = compute_W_HLL(lmbda_neg[j], lmbda_pos[j], W_L, W_R)

			# Compute the intermediate states
			S_dx = compute_S_dx(h_L, h_R, Z_L, Z_R, dx)
			q_star = compute_q_star(q_HLL, S_dx, lmbda_neg[j], lmbda_pos[j])
			S_dx_by_alph = compute_S_dx_by_alph(h_L, h_R, Z_L, Z_R, S_dx, q_star)
			h_L_star, h_R_star = compute_h_stars(h_HLL, lmbda_neg[j], lmbda_pos[j], S_dx_by_alph, h_L, h_R)
			W_L_stars[j] = [h_L_star, q_star]; W_R_stars[j] = [h_R_star, q_star]			


		dt = compute_timestep(lmbda_neg, lmbda_pos, dx)
		W_forward = MUSCL_forward_solution(W, dx, dt, nx, Z_arr, lmbda_neg, lmbda_pos, W_L_stars, W_R_stars)
		# W_forward = compute_forward_solution(W, dx, dt, nx, lmbda_neg, lmbda_pos, W_L_stars, W_R_stars)
		t += dt; n += 1


	curve_data_f.write("{}".format(n))
	curve_data_f.close()
	times_f.close()
	data.write("{}".format(n))
	data.close()





































			# if (h_L > 0 and h_R > 0):

			# 	# This is an expression of (3.18), the discrete version of a steady state
			# 	q_diff = np.abs(q_R - q_L)
			# 	S_dx_diff = np.abs(q_star ** 2 * (1 / h_R - 1 / h_L) + 0.5 * g * (h_R ** 2 - h_L ** 2) - S_dx)
			# 	if q_diff < ZERO_THRESH:
			# 		q_L = q_R
			# 		q_diff = 0
			# 	if S_dx_diff < ZERO_THRESH:
			# 		S_dx = q_star ** 2 * (1 / h_R - 1 / h_L) + 0.5 * g * (h_R ** 2 - h_L ** 2)
			# 		S_dx_diff = 0

			# 	# This signals entering a steady state, in which (3.18) holds.
			# 	if (S_dx_diff == 0) and (np.abs(q_diff) == 0) and (j > 0 and j < nx):

			# 		q0 = q_L 	# In a steady state, q_L = q_R =: q0

			# 		# Our defined value of q0 = q_L = q_R should equal the discharge intermediate state
			# 		q_diff = np.abs(q_star - q0)
			# 		if q_diff < ZERO_THRESH:
			# 			q_star = q0
			# 			q_diff = 0
			# 		assert q_diff == 0, "q's are off {}".format(q_diff)

			# 		# In a steady state the h_HLL term should reduce to the following
			# 		assert np.abs(h_HLL - (lmbda_R * h_R - lmbda_L * h_L) / lmbda_jump) <= ZERO_THRESH, "KHJK"

			# 		# We should also have h_HLL - lmbda_R * S_dx_by_alpha / lmbda_jump = h_L
			# 		err_L = np.abs(h_L_star - h_L)
			# 		err_R = np.abs(h_R_star - h_R)
			# 		if err_L < ZERO_THRESH:
			# 			h_L_star = h_L
			# 			err_L = 0
			# 		if err_R < ZERO_THRESH:
			# 			h_R_star = h_R
			# 			err_R = 0
			# 		assert err_L == 0 and err_R == 0, "Not well balanced!! {} {} {} {}".format(err_L, err_R, lmbda_jump, j)

			# elif (h_L == 0 and h_R == 0):

			# 	assert np.abs(S_dx) == 0 and np.abs(S_dx_by_alph) == 0				
			# 	q_diff = np.abs(q_R - q_L)
			# 	q0 = q_L
			# 	h_p_Z = h_R + Z_R - (h_L + Z_L)
			# 	if q_diff < ZERO_THRESH:
			# 		q_L = q_R
			# 		q_diff = 0
			# 	if q0 < ZERO_THRESH:
			# 		q0 = 0
			# 	if h_p_Z < ZERO_THRESH:
			# 		h_p_Z = 0

			# 	# This signals entering a steady state when the heights vanish
			# 	if q0 == 0 and h_p_Z == 0 and q_diff == 0:
			# 		assert np.abs(q_star) == 0, "q_star when heights vanish = {}".format(q_star)
			# 		assert np.abs(S_dx) == 0
			# 		assert np.abs(S_dx_by_alph) == 0



		# for j in range(nx + 1):

		# 	W_L = W[j]; W_R = W[j + 1]; h_L = W_L[0]; h_R = W_R[0]; q_L = W_L[1]; q_R = W_R[1]
		# 	W_LF = W_forward[j]; W_RF = W_forward[j + 1]; h_LF = W_LF[0]; h_RF = W_RF[0]

		# 	if (h_L > ZERO_THRESH and h_R > ZERO_THRESH and h_LF > ZERO_THRESH and h_RF > ZERO_THRESH):

		# 		# This is an expression of (3.18), the discrete version of a steady state
		# 		q_diff = np.abs(q_R - q_L)
		# 		S_dx_diff = np.abs(q_star ** 2 * (1 / h_R - 1 / h_L) + 0.5 * g * (h_R ** 2 - h_L ** 2) - S_dx)

		# 		# This signals entering a steady state, in which (3.18) holds.
		# 		if (S_dx_diff <= ZERO_THRESH) and (np.abs(q_diff) <= ZERO_THRESH) and (j > 0 and j < nx):

		# 			h_diff, q_diff = np.abs(W_forward[j] - W[j])
		# 			assert h_diff < ZERO_THRESH and q_diff < ZERO_THRESH, "(h, q) diffs = {} {}".format(h_diff, q_diff)

					# a_L = (lmbda_R * h_R - lmbda_L * h_L) / lmbda_jump
					# c_L = lmbda_R * S_dx_by_alph / lmbda_jump
					# a_R = (lmbda_R * h_R - lmbda_L * h_L) / lmbda_jump
					# c_R = lmbda_L * S_dx_by_alph / lmbda_jump
					# diff = np.abs(h_HLL - a_L)
					# assert diff < ZERO_THRESH, "GEN {}".format(diff)
					# h_L_tilde = h_HLL - lmbda_R * S_dx_by_alph / lmbda_jump
					# h_R_tilde = h_HLL - lmbda_L * S_dx_by_alph / lmbda_jump
					# diff_L = np.abs(h_L_tilde - (a_L - c_L))
					# diff_R = np.abs(h_R_tilde - (a_R - c_R))
					# assert diff_L < ZERO_THRESH and diff_R < ZERO_THRESH, "{} {}".format(diff_L, diff_R)


			# if h_L <= ZERO_THRESH:
			# 	S_dx_diff = np.abs(0.5 * g * (h_R ** 2 - h_L ** 2) - S_dx)
			# 	if S_dx_diff <= ZERO_THRESH and (j > 0 and j < nx):
			# 		err = h_R + Z_R - h_L - Z_L
			# 		assert err <= ZERO_THRESH, "Note well balanced!! {}".format(err)
			# 		assert h_L_star <= ZERO_THRESH and h_R_star - h_R <= ZERO_THRESH, "YA"

			# if h_R <= ZERO_THRESH:
			# 	S_dx_diff = np.abs(0.5 * g * (h_R ** 2 - h_L ** 2) - S_dx)
			# 	if S_dx_diff <= ZERO_THRESH and (j > 0 and j < nx):
			# 		err = h_R + Z_R - h_L - Z_L
			# 		assert err <= ZERO_THRESH, "Note well balanced!! {}".format(err)
			# 		assert h_R_star <= ZERO_THRESH and h_L_star - h_L <= ZERO_THRESH, "YAR"
