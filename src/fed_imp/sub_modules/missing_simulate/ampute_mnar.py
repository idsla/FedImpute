import numpy as np
import scipy.stats as stats


def MNAR_mask_random(data, important_feature_list, missing_ratio, p_na_cols, sigma=0.2, n_bins=5, seed=199):
	np.random.seed(seed)
	d = data.shape[1]
	d_na = np.clip(int(p_na_cols * d), 1, d - 1)
	if important_feature_list is not None:
		column_indices = np.random.choice(important_feature_list, d_na, replace=False)
	else:
		column_indices = np.random.choice(d - 1, d_na, replace=False)
	missing_dist_matrix = generate_missing_dist_one_client(missing_ratio, sigma, d_na, n_bins, seed)
	mask = add_missing_mnar_noniid_client(data, column_indices, missing_dist_matrix, seed)

	return mask


########################################################################################################################
# Missing distribution
########################################################################################################################
def generate_missing_dist_one_client(missing_ratio, client_sigma, n_cols, n_bins, seed=199):
	# Set the desired mean and standard deviation
	lower, upper = 0, 1
	mu, sigma = missing_ratio, client_sigma
	X = stats.truncnorm(
		(lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
	)
	ret = []
	for i in range(n_cols):
		np.random.seed(seed)
		seed += 1093039
		while True:
			sample = X.rvs(n_bins)
			if abs(sample.mean() - missing_ratio) < 0.1 and abs(sample.std() - sigma) < 0.1:
				break
		np.sort(sample)
		np.random.shuffle(sample)
		ret.append(sample)

	return np.array(ret)


def generate_missing_dist(column_indices, missing_ratio, client_sigma, n_clients, n_bins, seed=199):
	rets = []
	for _ in range(n_clients):
		seed = (seed + 10394949) % 1000000007
		ret = generate_missing_dist_one_client(missing_ratio, client_sigma, len(column_indices), n_bins, seed)
		rets.append(ret)

	return rets


########################################################################################################################
# Add missing
########################################################################################################################
def add_missing_column(input_array, num_bins, missing_dist_array, seed=199):
	# compute quantiles
	q = np.linspace(0, 1, num_bins + 1)
	quantiles = np.quantile(input_array, q)

	# find the bin each sample falls in
	bin_indices = np.digitize(input_array, quantiles)

	# generate missing values
	mask = np.zeros(input_array.shape, dtype=bool)
	for i in range(len(missing_dist_array)):
		inds = np.where(bin_indices == (i + 1))[0]
		np.random.seed(seed)
		mask[inds] = np.random.rand(len(inds)) < missing_dist_array[i]
	return mask


def add_missing_mnar_noniid_client(data, column_indices, missing_dist_matrix, seed=199):
	mask = np.zeros(data.shape, dtype=bool)
	for idx, col_idx in enumerate(column_indices):
		seed = (seed + 1093039) % 2 ** 32
		mask_col = add_missing_column(
			data[:, col_idx], missing_dist_matrix.shape[1], missing_dist_matrix[idx, :], seed
		)
		mask[:, col_idx] = mask_col
	return mask
