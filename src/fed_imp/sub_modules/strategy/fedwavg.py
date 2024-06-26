import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve
from config import settings

def fedwavg(weights, losses, missing_infos, alpha, beta, loss, ms_field):

	# get field need to use
	if ms_field is None:
		raise ValueError('compute_weight_field_name is not set')

	weights = np.array(list(weights.values()))

	##################################################################################
	# Losses
	##################################################################################
	losses = np.array([v[loss] for k, v in losses.items()])
	if loss == 'rmse':
		losses = normalization(losses, inverse=True, ratio=0.2)
	elif loss == 'r2':
		losses = normalization(losses, ratio=0.2)

	losses = losses / losses.sum()  # 1/exp(-x) 0.92 0.93 0.94

	##################################################################################
	# weights missing
	##################################################################################
	ms_weights = np.array([1 - v[ms_field] + 0.0001 for v in missing_infos.values()])
	#ms_weights = normalization(ms_weights, ratio=0.02)
	ms_weights = ms_weights / ms_weights.sum()

	##################################################################################
	# sample size
	##################################################################################
	sample_size = np.array([v['sample_size'] for v in missing_infos.values()])
	sample_size = sample_size / sample_size.sum()

	# compute weights
	c = alpha * sample_size + beta * losses + (1 - alpha - beta) * ms_weights

	# take average
	ret = np.average(weights, axis=0, weights=c)

	return ret


def fedwavg2(weights, losses, missing_infos):

	weights = np.array(list(weights.values()))
	alpha = settings['algo_params']['fedwavg']['alpha']

	##################################################################################
	# Losses
	##################################################################################
	loss = 'rmse'
	losses = np.array([v[loss] for k, v in losses.items()])
	if loss == 'rmse':
		losses = normalization(losses, inverse=True, ratio=0.2)
	elif loss == 'r2':
		losses = normalization(losses, ratio=0.2)

	losses = losses / losses.sum()  # 1/exp(-x) 0.92 0.93 0.94

	##################################################################################
	# weights missing
	##################################################################################
	ms_field = 'missing_cell_pct'
	ms_weights = np.array([1 - v[ms_field] + 0.0001 for v in missing_infos.values()])
	#ms_weights = normalization(ms_weights, ratio=0.02)
	ms_weights = ms_weights / ms_weights.sum()

	##################################################################################
	# sample size
	##################################################################################
	sample_size = np.array([v['sample_size'] for v in missing_infos.values()])
	sample_size = sample_size / sample_size.sum()

	# compute weights
	scale_factor = settings['algo_params']['scale_factor']
	c = (alpha*sample_size + (1 - alpha)*ms_weights)**scale_factor
	c = c/c.sum()

	# take average
	ret = np.average(weights, axis=0, weights=c)

	return ret


def fedwavg3(weights, projection_matrix, missing_infos):

	sample_size = np.array([v['sample_size'] for v in missing_infos.values()])
	scale_factor = 2
	c = sample_size**scale_factor
	c = c/c.sum()

	proj_matrices = np.array(list(projection_matrix.values()))
	weights = np.array(list(weights.values()))

	# Calculate A and b
	A = sum([c[i]*(proj_matrices[i].T @ proj_matrices[i]) for i in range(len(proj_matrices))])
	b = sum([c[i]*(proj_matrices[i].T @ proj_matrices[i] @ weights[i]) for i in range(len(proj_matrices))])

	try:
		# Try solving the linear system exactly
		w_optimal = solve(A, b)
	except LinAlgError:
		# If the exact solution fails, use the least squares method
		w_optimal, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
	#print(w_optimal.shape)
	return w_optimal


def normalization(x, inverse=False, ratio=0.02):
	if inverse:
		x = 1/np.exp(x)  # 0.0110 0.0111 0.0112
	#return x
	x_range = x.max(axis=0) - x.min(axis=0)
	return (x - (x.mean(axis=0) - ratio*x_range)) / ((1 + ratio)*x_range)
