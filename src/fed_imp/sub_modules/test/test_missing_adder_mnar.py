from ..missing_simulate.ampute_mnar import (
	generate_missing_dist, add_missing_column, add_missing_mnar_noniid_client, simulate_nan_mnar,
)
import numpy as np


def test_generate_missing():

	column_indices = [0, 1, 2]
	missing_ratio = 0.5
	client_sigma = 0.3
	n_clients = 3
	n_bins = 5
	ret = generate_missing_dist(column_indices, missing_ratio, client_sigma, n_clients, n_bins)
	print('\n')
	for item in ret:
		print(item)

	for client in range(n_clients):
		data = np.random.rand(500, 5)
		mask = add_missing_mnar_noniid_client(data, column_indices, ret[client])
		print(mask.sum(axis=0) / mask.shape[0])

	assert True


def test_simulate_nan_mnar():
	datas = [np.random.rand(500, 5), np.random.rand(500, 5), np.random.rand(500, 5)]
	missing_ratio = 0.5
	p_na_cols = 1.0
	sigma = 0.3
	n_bins = 5
	seed = 129301
	for data in datas:
		seed = (seed + 1093039) % 2 ** 32
		ret = simulate_nan_mnar(data, missing_ratio, p_na_cols, sigma, n_bins, seed=seed)
		mask = np.isnan(ret['train_data_ms'])
		print(mask.sum(axis=0) / mask.shape[0])
	assert True
