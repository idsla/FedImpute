import numpy as np
from loguru import logger
from scipy.linalg import solve, LinAlgError
from collections import OrderedDict
from copy import deepcopy



def testavg(weights, missing_infos, frac = 0.5):
	clients_weights = np.array(list(weights.values()))
	n_clients = clients_weights.shape[0]
	cluster1_idx = list(range(0, int(n_clients*frac)))
	cluster2_idx = list(range(int(n_clients*frac), n_clients))

	sample_size = np.array([v['sample_size'] + 1e-4 for v in missing_infos.values()])
	cluster1_avg_params = np.average(clients_weights[cluster1_idx, :], axis=0, weights = sample_size[cluster1_idx])
	cluster2_avg_params = np.average(clients_weights[cluster2_idx, :], axis=0, weights = sample_size[cluster2_idx])

	final_parameters, w = [], []
	for client_idx in range(len(weights)):
		#print(client_idx, cluster1_idx, cluster2_idx)
		if client_idx in cluster1_idx:
			#print("cluster1 - {}".format(client_idx))
			final_parameters.append(cluster2_avg_params)
			#w.append(sample_size[cluster2_idx]/np.sum(sample_size[cluster2_idx]))
		else:
			#print("cluster2 - {}".format(client_idx))
			final_parameters.append(cluster1_avg_params)
			#w.append(sample_size[cluster1_idx]/np.sum(sample_size[cluster1_idx]))

	return final_parameters, w

def fedavgs(weights, missing_infos):
	clients_weights = np.array(list(weights.values()))
	sample_size = np.array([v['sample_size'] + 1e-4 for v in missing_infos.values()])
	ret = np.average(clients_weights, axis=0, weights=sample_size)

	final_parameters = []
	for client in range(len(weights)):
		final_parameters.append(ret)

	return final_parameters, sample_size/np.sum(sample_size)


def fedavg(weights):
	clients_weights = np.array(list(weights.values()))
	ret = np.mean(clients_weights, axis=0)
	return ret

def fedavg_vae(weights, missing_infos):

	# federated averaging implementation
	averaged_model_state_dict = OrderedDict()  # global parameters
	sample_sizes = np.array([v['sample_size'] + 1e-4 for v in missing_infos.values()])
	normalized_coefficient = [size / sum(sample_sizes) for size in sample_sizes]

	for it, local_model_state_dict in enumerate(weights.values()):
		for key in local_model_state_dict.keys():
			if it == 0:
				averaged_model_state_dict[key] = normalized_coefficient[it] * local_model_state_dict[key]
			else:
				averaged_model_state_dict[key] += normalized_coefficient[it] * local_model_state_dict[key]

	# copy parameters for each client
	agg_model_parameters = [deepcopy(averaged_model_state_dict) for _ in range(len(weights.values()))]
	return agg_model_parameters, sample_sizes/np.sum(sample_sizes)



def fedavgh(weights):
	clients_weights = np.array(list(weights.values()))[:5, :]
	ret = np.mean(clients_weights, axis=0)
	return ret


def fedavgcross(weights):
	rets = []
	clients_weights = np.array(list(weights.values()))[:5, :]
	ret = np.mean(clients_weights, axis=0)
	rets.append(ret)
	clients_weights = np.array(list(weights.values()))[5:, :]
	ret = np.mean(clients_weights, axis=0)
	rets.append(ret)
	return rets


def fedavg2(weights, projection_matrix):

	if projection_matrix is None:
		return fedavg(weights)
	else:
		proj_matrices = np.array(list(projection_matrix.values()))
		weights = np.array(list(weights.values()))

		# Calculate A and b
		A = sum([proj_matrices[i].T @ proj_matrices[i] for i in range(len(proj_matrices))])
		b = sum([proj_matrices[i].T @ proj_matrices[i] @ weights[i] for i in range(len(proj_matrices))])

		try:
			# Try solving the linear system exactly
			w_optimal = solve(A, b)
		except LinAlgError:
			# If the exact solution fails, use the least squares method
			w_optimal, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
		#print(w_optimal.shape)
		return w_optimal

