import numpy as np
import random


def simulate_nan_mcar(data, cols, missing_ratio, rng=np.random.default_rng(201030)):

	mask = np.zeros_like(data, dtype=bool)

	for idx, col_idx in enumerate(cols):
		indices = np.arange(data.shape[0])
		if isinstance(missing_ratio, dict):
			ratio = missing_ratio[idx]
		elif isinstance(missing_ratio, list):
			ratio = missing_ratio[idx]
		else:
			ratio = missing_ratio
		na_indices = rng.choice(indices, int(ratio * data.shape[0]), replace=False)
		mask[na_indices, col_idx] = True

	data_nas = data.copy()
	data_nas[mask.astype(bool)] = np.nan

	return data_nas


