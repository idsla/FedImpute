import random
import numpy as np
import scipy.stats as stats


def test_missing_distribution():
	missing_ratio = 0.5
	client_sigma = 0.2
	n_clients = 10
	n_bins = 10
	# Set the desired mean and standard deviation
	lower, upper = 0, 1
	mu, sigma = missing_ratio, client_sigma
	X = stats.truncnorm(
		(lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
	)
	print('\n')
	seed = 199
	for i in range(n_clients):
		np.random.seed(seed)
		seed += 1093039
		while True:
			sample = X.rvs(n_bins)
			if abs(sample.mean() - missing_ratio) < 0.05 and abs(sample.std() - sigma) < 0.05:
				break
		np.sort(sample)
		np.random.shuffle(sample)
		print(sample)


def test_missing_distribution_mnar_noniid():

	missing_ratio = 0.5
	client_sigma = 0.2
	n_clients = 10
	n_bins = 10
	column_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	missing_distribution_dict = {}
	for col_idx in column_indices:
		ret = np.random.random((n_clients, n_bins))
		missing_distribution_dict[col_idx] = ret

	return missing_distribution_dict


def test_binning_array():
	num_bins = 5
	input_array = np.random.rand(50, 2)
	p = np.array([[0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5]])
	# compute quantiles
	q = np.linspace(0, 1, num_bins + 1)
	quantiles = np.quantile(input_array, q)
	# find the bin each sample falls in
	bin_indices = np.digitize(input_array, quantiles)
	print(bin_indices)
	# # generate missing values
	mask = np.zeros(input_array.shape, dtype=bool)
	for i in range(p.shape[0]):
		inds = np.where(bin_indices == (i+1))[0]
	# 	mask[inds] = np.random.rand(len(inds)) < p[i]
	# print(mask.sum()/len(mask))




