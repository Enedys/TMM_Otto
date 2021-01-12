import matplotlib.pyplot as plt
import numpy as np
import itertools
# from numba import njit
#import pandas as pd

#@njit
def transfer_matrix(N1, N2, polar='TE', n1=1, n2=1):
	tm = np.empty((2,2), dtype = np.complex)
	if polar == 'TE':
		tm[0, 0] = (N2 + N1) / (2. * N2)
		tm[0, 1] = (N2 - N1) / (2. * N2)
		tm[1, 0] = (N2 - N1) / (2. * N2)
		tm[1, 1] = (N2 + N1) / (2. * N2)
	else:
		tm[0, 0] = (n2 ** 2 * N1 + n1 ** 2 * N2) / (2. * n1 * n2 * N2)
		tm[0, 1] = (n2 ** 2 * N1 - n1 ** 2 * N2) / (2. * n1 * n2 * N2)
		tm[1, 0] = (n2 ** 2 * N1 - n1 ** 2 * N2) / (2. * n1 * n2 * N2)
		tm[1, 1] = (n2 ** 2 * N1 + n1 ** 2 * N2) / (2. * n1 * n2 * N2)
	return tm

#@njit
def N_calculation(n_first_media, incident_angle, n_current_media):
	return np.sqrt(n_current_media ** 2 - n_first_media ** 2 * (np.sin(incident_angle)) ** 2, dtype=np.complex)

#@njit
def phase_shift(d, N, k0):
	p_shift = np.empty((2, 2), dtype = np.complex)
	p_shift[0, 0] = np.exp(1.j * d * N * k0)
	p_shift[0, 1] = 0.+0.j
	p_shift[1, 0] = 0.+0.j
	p_shift[1, 1] = np.exp(-1.j * d * N * k0)
	return p_shift

#@njit
def R_func(dictionary_structure={}, wl = 780, teta = 45, polar='TE'):
	list_N = []
	list_n = []
	T = np.eye(2, dtype=np.complex)
	tet = teta * np.pi / 180
	k0 = 2 * np.pi / wl
	n0 = complex(0., 0.)  # current n
	for i in range(0, len(dictionary_structure)):
		if i == 0:
			n0 = dictionary_structure[i]['n']
			list_N.append(N_calculation(n0, tet, n0))
			list_n.append(n0) # добавил n для TM
			continue
		if dictionary_structure[i]['name'] == 'ФК':
			n1 = dictionary_structure[i]['n1']
			n2 = dictionary_structure[i]['n2']
			d1 = dictionary_structure[i]['d1']
			d2 = dictionary_structure[i]['d2']
			num_layers = int(dictionary_structure[i]['N'])
			N1 = N_calculation(n0, tet, n1)
			N2 = N_calculation(n0, tet, n2)
			from_upper_to_1 = transfer_matrix(list_N[-1], N1, polar, list_n[-1], n1)
			from_1_to_2 = transfer_matrix(N1, N2, polar, n1, n2)
			from_2_to_1 = transfer_matrix(N2, N1, polar, n2, n1)
			F_n1 = phase_shift(d1, N1, k0) # x1 - d, x2 - N, x3 - k0
			F_n2 = phase_shift(d2, N2, k0)

			T = F_n2 @ from_1_to_2 @ F_n1 @ from_upper_to_1 @ T
			T_bilayer = F_n2 @from_1_to_2 @ F_n1 @from_2_to_1
			T = np.linalg.matrix_power(T_bilayer, num_layers - 1) @ T
			list_N.append(N1)
			list_N.append(N2)
			list_n.append(n1)
			list_n.append(n2)
		elif dictionary_structure[i]['name'] == 'Слой':
			n = dictionary_structure[i]['n']
			d = dictionary_structure[i]['d']
			N = N_calculation(n0, tet, n)
			from_this_to_next = transfer_matrix(list_N[-1], N, polar, list_n[-1], n)
			F = phase_shift(d, N, k0)
			T = np.dot(from_this_to_next, T)
			T = np.dot(F, T)
			list_N.append(N)
			list_n.append(n) # добавил n для TM
		elif dictionary_structure[i]['name'] == 'Среда':
			n = dictionary_structure[i]['n']
			N = N_calculation(n0, tet, n)
			from_last_to_sub = transfer_matrix(list_N[-1], N, polar, list_n[-1], n)
			T = np.dot(from_last_to_sub, T)
			list_N.append(N)
			list_n.append(n)
	return -(T[1, 0] / T[1, 1])

'''
get_rotation function take angle
in grad and return rotation matrix
'''
#@njit
def get_rotation(angle):
	rot_matrix = np.empty((2, 2), dtype = np.complex)
	rot_matrix[0, 0] = np.cos(np.pi * angle / 180)
	rot_matrix[0, 1] = np.sin(np.pi * angle / 180)
	rot_matrix[1, 0] = np.sin(np.pi * angle / 180)
	rot_matrix[1, 1] = -np.cos(np.pi * angle / 180)
	return rot_matrix

#@njit
def decor1(angles, wl, struct, polar):
	return R_func(struct, wl, angles, polar)

vectorize_R = np.vectorize(decor1)
vectorize_N = np.vectorize(N_calculation)

#@njit
def transfer_matrix_vec(N1, N2, polar='TE', n1=1+0.j, n2=1+0.j):
	if polar == 'TE':
		t00 = (N2 + N1) / (2. * N2)
		t01 = (N2 - N1) / (2. * N2)
		t10 = (N2 - N1) / (2. * N2)
		t11 = (N2 + N1) / (2. * N2)
	else:
		t00 = (n2 ** 2 * N1 + n1 ** 2 * N2) / (2. * n1 * n2 * N2)
		t01 = (n2 ** 2 * N1 - n1 ** 2 * N2) / (2. * n1 * n2 * N2)
		t10 = (n2 ** 2 * N1 - n1 ** 2 * N2) / (2. * n1 * n2 * N2)
		t11 = (n2 ** 2 * N1 + n1 ** 2 * N2) / (2. * n1 * n2 * N2)
	return np.vstack((t00, t01, t10, t11)).reshape((2, 2, -1)).T

#@njit
def phase_shift_vec(d, N, k0):
	p00 = np.exp(1.j * d * N * k0)
	p01 = np.zeros(np.shape(p00), dtype=np.complex)
	p10 = np.zeros(np.shape(p00), dtype=np.complex)
	p11 = np.exp(-1.j * d * N * k0)
	return np.vstack((p00, p01, p10, p11)).reshape((2, 2, -1)).T

#@njit
def generate_TMM(n0, n1, n2, angles, polar):
	N1 = N_calculation(n0, np.pi / 180 * angles, n1)
	N2 = N_calculation(n0, np.pi / 180 * angles, n2)
	TMMs = transfer_matrix_vec(N1, N2, polar, n1, n2)
	return TMMs

#@njit
def generate_TMM_to_metal(n0, n2, n3, angles, polar, zk, TMM1):
	N2 = N_calculation(n0, np.pi / 180 * angles, n2)
	N3 = N_calculation(n0, np.pi / 180 * angles, n3)
	TMM2 = transfer_matrix_vec(N2, N3, polar, n2, n3)
	Phases = phase_shift_vec(zk, N2, 1.)
	TMMs = TMM2 @ Phases @ TMM1
	return TMMs

#@njit
def get_K_xz_air(k_mkm, k_mkm_gap, angles):
	k_x_b = k_mkm * np.sin(np.pi / 180 * angles, dtype=np.complex)
	k_z_b = np.sqrt(k_mkm_gap ** 2 - k_x_b ** 2, dtype=np.complex)
	k_matrix = np.vstack((k_x_b, k_z_b))
	return k_matrix

#@njit
def E_air_spectrum(s, R, TMMs_):
	E_air = np.empty((2, np.size(s)), dtype=np.complex)
	E_inc = np.array(np.hstack((s, s * R)).T, dtype=np.complex)

	# tmp = E_inc.T.reshape((np.size(s), 2, 1))
	# res = TMMs_ @ tmp
	# E_plus = res[:, 0, 0].reshape((-1, 1))
	# E_minus = res[:, 1, 0].reshape((-1, 1))

	for i in range(np.size(s)):
		E_air[:, i] = TMMs_[i] @ E_inc[:, i]
	E_plus = E_air[0].reshape((-1, 1))
	E_minus = E_air[1].reshape((-1, 1))
	return E_plus, E_minus

#@njit
def get_defocus_phase(k_mkm, angles, L):
	if L == 0:
		return 1.
	median_angle = np.pi / 180 * angles[np.size(angles) // 2]
	h_shift = L * np.cos(median_angle)
	x_shift = L * np.sin(median_angle)
	k_x = k_mkm * np.sin(np.pi / 180 * angles)
	k_z = k_mkm * np.cos(np.pi / 180 * angles)
	phase_shift = k_x * x_shift + k_z * h_shift
	return np.exp(1.0j * phase_shift).reshape((-1, 1))

#@njit
def spectral(a, k):
	if a == np.inf:
		return np.array([1.], dtype=np.float64)
	return np.sqrt(a)/(2 * np.sqrt(np.pi)) * np.exp(-a**2 * k**2 / 4)

#@njit
def get_k_x(a=5, N_x=401, decr_times=20):
	if a == np.inf:
		return np.array([0.], dtype=np.float64)
	k_min = 2 * np.sqrt(np.log(decr_times)) / a
	dk_x = 2 * k_min / N_x
	return np.arange(-N_x // 2, N_x // 2) * dk_x

def cut_BSW_intesity(angles_range, X, struct, a=5, Z=0.3, L=0, polar='TE', wl=780., plasmon=False):
	'''
	set default parameters range
	'''
	# _, ax = plt.subplots(1, 4, figsize=(20, 5))

	k = 2 * np.pi * 1.0e9 / wl * struct[0]['n']
	k_mkm = k / 1.0e6
	k_mkm_gap = k_mkm / struct[0]['n'] * struct[1]['n']
	k_x = get_k_x(a=a, N_x=401, decr_times=5)
	dk = k_x[1] - k_x[0]
	s = spectral(a, k_x).reshape((-1, 1))
	r_pos = np.empty((1, 2))
	r_pos[0,1] = Z
	E_destr = np.empty((np.size(angles_range), np.size(X)), dtype=np.complex)
	for j, alpha in enumerate(angles_range):
		angles = 180 / np.pi * np.arcsin(k_x / k_mkm) + alpha
		R = vectorize_R(angles, wl, struct, polar).reshape((-1, 1))
		TMM1 = generate_TMM(struct[0]['n'], struct[0]['n'], struct[1]['n'], angles, polar)
		if plasmon == True:
			TMM2 = generate_TMM_to_metal(struct[0]['n'], struct[1]['n'], struct[2]['n'],
										angles, polar, Z * k_mkm / struct[0]['n'], TMM1)
			TMMs = generate_TMM_to_metal(struct[0]['n'], struct[2]['n'], struct[3]['n1'],
										angles, polar, struct[2]['d'] / 1000. * k_mkm / struct[0]['n'], TMM2)
			r_pos[0,1] = 0
		else:
			TMMs = TMM1
		k_xz = get_K_xz_air(k_mkm, k_mkm_gap, angles)
		defocus_phase = get_defocus_phase(k_mkm, angles, L) # эта фаза считается в призме, поэтому вычтсленные выше k_xz для воздуха не используются
		E_plus, E_minus = E_air_spectrum(s, R, TMMs)
		E_forward = np.zeros(np.size(X), dtype=np.complex)
		E_backward = np.zeros(np.size(X), dtype=np.complex)

		# if j % 8 == 0:
		# 	ax[0].plot(angles.real, abs(E_plus))
		# 	ax[1].plot(angles.real, abs(E_minus))
		# 	ax[2].plot(angles.real, abs(R) + 0.05 * (j // 8))
		# 	ax[3].plot(angles.real, np.angle(R) + 0.05 * (j // 8))
		
		# if angles[0].real < 42.2:
		# 	E_destr[j] = np.zeros(np.size(X), dtype=np.complex)
		# 	continue

		# coord_arr = np.vstack((X, Z * np.ones(np.size(X)))).T
		# E_forward = np.exp(1.0j * (coord_arr @ k_xz), dtype=np.complex) @ (E_plus * defocus_phase) * dk
		# coord_arr = np.vstack((X, -Z * np.ones(np.size(X)))).T
		# E_backward = np.exp(1.0j * (coord_arr @ k_xz), dtype=np.complex) @ (E_minus * defocus_phase) * dk
		# E_destr[j] = (E_forward + E_backward).T

		for i, x in enumerate(X):
			r_pos[0, 0] = x
			E_forward[i] = np.exp(1.0j * r_pos @ k_xz, dtype=np.complex) @ (E_plus * defocus_phase) * dk
			r_pos[0, 1] *= -1
			E_backward[i] = np.exp(1.0j * r_pos @ k_xz, dtype=np.complex) @(E_minus * defocus_phase) * dk
			r_pos[0, 1] *= -1
		E_destr[j] = E_forward + E_backward
	return E_destr

#@njit
def plot_2Dmap(data, extent=None, isE=False):
	plt.figure(figsize=(9, 5))
	if isE == True:
		proc_data = data
	else:
		proc_data = abs(data) ** 2
	plt.imshow(proc_data, cmap=plt.cm.hot, origin='lower', aspect='auto', extent=extent)
	cax = plt.axes([0.95, 0.13, 0.05, 0.75])
	plt.colorbar(cax=cax)
	plt.show()

#@njit
def E_beam_calc(X, Z, s, k_matrix, mode='i', alpha=0, l_0=0, plot_I=True):
	'''
	This function compute field distribution by matrix multiplications.
	grid - array of coordinates in form:
		[(x_0, z_0), (x_0, z_1), ..., (x_0, z_n),
						...,
		(x_m, z_0), (x_m, z_1), ..., (x_m, z_n)]
	
	'''
	rot_matrix = get_rotation(alpha)
	dk = k_matrix[0][1] - k_matrix[0][0]
	h_shift = 0. if l_0 == 0 else l_0 * np.cos(np.pi / 180 * alpha)
	sign = 1. if mode == 'i' else -1
	grid = np.array(list(itertools.product(X, sign * Z - h_shift)), dtype=np.complex)
	E = np.exp(1.0j * (grid @ rot_matrix.T) @ k_matrix, dtype=np.complex) @ s * dk
	E_res = E.reshape((np.size(X), -1)).T
	if plot_I == True:
		plot_2Dmap(E_res)
	return E_res

#@njit
def vizualize_beam(X, Z, struct, alpha, l_0=0, a=5, mode='r', wl=780., polar='TE', plot=False):

	'''
	This function return complex value E (electromagnetic field)
	distribution in the modulation field and plot Intensity 
	distribution, that can be set throught input parameters.
	
	Parameters:
		X - x-range coorinates
		Z - z-range coorinates
		struct - structure (Otto configuration for example) description
				in dictionaty form
		alpha - incidence angle (degree)
		l_0 - defocusing length (mkm)
		mode - simalation regime (r - reflected, i - incidence, f - full)
		wl - wavelength (nm)
	'''

	k = 2 * np.pi * 1.0e9 / wl * struct[0]['n']
	k_mkm = k / 1.0e6
	k_x = get_k_x(a=a, N_x=401)
	k_z = np.sqrt(k_mkm ** 2 - k_x ** 2, dtype=np.complex)
	s = spectral(a=a, k=k_x).reshape((-1, 1))
	k_matrix = np.vstack((k_x, k_z))
	s_angle_range = 180 / np.pi * np.arcsin(k_x / k_mkm)

	angles = s_angle_range + alpha
	R = vectorize_R(angles, wl, struct, polar).reshape((-1, 1))
	if mode == 'r' or mode == 'f':
		E_refl = E_beam_calc(X, Z, s * R, k_matrix, 'r', alpha, l_0, plot)
	if mode == 'i' or mode == 'f':
		E_inc = E_beam_calc(X, Z, s, k_matrix, 'i', alpha, l_0, plot)

	if mode == 'f':
		return [E_inc, E_refl, E_inc + E_refl]
	elif mode == 'r':
		return [E_refl]
	elif mode == 'i':
		return [E_inc]


def get_max_angle(E_bsw, return_shift=None, plot_I_a=None, plot_I_x=None):
    I = np.abs(E_bsw) ** 2
    I_max = []
    for line in I:
        I_max.append(np.max(line))
    if plot_I_a != None:
        plt.plot(angles_range, I_max)
        plt.grid()
        plt.xlabel("Inc angle")
        plt.ylabel("Intensity")
    angle_max = angles_range[np.argmax(I_max)]
    if return_shift != None:
        if plot_I_x != None:
            plt.plot(X_range, I[np.argmax(I_max)])
        shift = X_range[np.argmax(I[np.argmax(I_max)])]
        return [angle_max, shift, np.max(I)]
    print("I_max: ", np.max(I_max), "angle_max: ", angles_range[np.argmax(I_max)])
    return [angle_max]