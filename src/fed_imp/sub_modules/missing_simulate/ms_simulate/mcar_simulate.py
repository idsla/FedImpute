import numpy as np
import random


def simulate_nan_mcar(data, cols, missing_ratio, seed=201030):

	mask = np.zeros_like(data, dtype=bool)

	for col_idx in cols:
		indices = np.arange(data.shape[0])
		seed = (seed + 102989221) % 1000000007
		rng = np.random.default_rng(seed)
		random.seed(seed)
		if isinstance(missing_ratio, dict):
			ratio = missing_ratio[col_idx]
		else:
			ratio = missing_ratio
		na_indices = rng.choice(indices, int(ratio * data.shape[0]), replace=False)
		mask[na_indices, col_idx] = True

	data_nas = data.copy()
	data_nas[mask.astype(bool)] = np.nan

	return data_nas


