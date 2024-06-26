from typing import List, Dict

from .missing_strategy import missing_strategy
from .ampute_mnar import MNAR_mask_random
from .ampute_hpyimp import MCAR_mask, MAR_mask
import numpy as np


def missing_adding_clients(train_data_list, params, important_feature_list, seed=201030) -> List[Dict[str, np.ndarray]]:

	# get missing strategy for each client
	num_clients = len(train_data_list)

	strategies = missing_strategy(params, num_clients, seed)

	rets = []
	for idx, (data, strategy) in enumerate(zip(train_data_list, strategies)):
		seed = (seed + 10097) % 1000000007
		strategy["important_feature_list"] = None
		if "use_important_feature" in strategy:
			if strategy["use_important_feature"]:
				strategy["important_feature_list"] = important_feature_list

		rets.append(simulate_nan(data=data, seed=seed, **strategy))

	return rets


def simulate_nan(
		data, important_feature_list, missing_ratio, mechanism='mcar', p_na_cols=0.5, sigma=0.2, n_bins=5, seed=201030
):
	"""
	Simulate process of adding missing data to the dataset
	:param seed: seed for random number generator
	:param sample_columns: whether to sample columns with missing data
	:param p_na_cols: proportion of columns with missing data
	:param data: original data matrix
	:param missing_ratio: proportion of missing data of the dataset
	:param mechanism: missing mechanism, 'mcar' or 'mar' or 'mnar'
	:return: original data matrix, incomplete data matrix, mask matrix
	"""
	if mechanism == 'mcar':
		mask = MCAR_mask(data, important_feature_list, missing_ratio, p_na_cols, seed=seed).astype(float)
	elif mechanism == 'mar':
		mask = MAR_mask(data, important_feature_list, missing_ratio, p_na_cols, use_y=False, seed=seed).astype(float)
	elif mechanism == 'mary':
		mask = MAR_mask(data, important_feature_list, missing_ratio, p_na_cols, use_y=True, seed=seed).astype(float)
	elif mechanism == 'mnar-random':
		mask = MNAR_mask_random(
			data, important_feature_list, missing_ratio, p_na_cols, sigma=sigma, n_bins=n_bins, seed=seed
		)
	else:
		raise NotImplementedError

	X_nas = data.copy()
	X_nas[mask.astype(bool)] = np.nan

	return {"train_data": data, "train_data_ms": X_nas}
