from pyampute.ampute import MultivariateAmputation
import numpy as np
import random

from scipy import optimize
from scipy.special import expit
from sklearn.feature_selection import mutual_info_regression


########################################################################################################################
# Missing Pattern Simulation MCAR
########################################################################################################################
def simulate_nan_patterns_mcar_pyamp(data, ms_patterns, ms_ratio, seed):
	if isinstance(ms_patterns, tuple):
		ms_patterns = [ms_patterns]

	ret = []
	for ms_pattern in ms_patterns:
		ret.append({"incomplete_vars": ms_pattern, "mechanism": "MCAR", "score_to_probability_func": "sigmoid-left"})

	ma = MultivariateAmputation(prop=ms_ratio, patterns=ret, seed=seed)
	data_ms = ma.fit_transform(data)

	return data_ms


def simulate_nan_mcar_patterns(data, ms_patterns, ms_ratio, seed=2010202):

	mask = np.zeros_like(data, dtype=bool)
	ms_ratio = ms_ratio / len(ms_patterns)

	for ms_pattern in ms_patterns:
		indices = np.arange(data.shape[0])
		seed = (seed + 102989221) % 1000000007
		rng = np.random.default_rng(seed)
		random.seed(seed)
		na_indices = rng.choice(indices, int(ms_ratio * data.shape[0]), replace=False)
		mask[na_indices, ms_pattern] = True

	data_nas = data.copy()
	data_nas[mask.astype(bool)] = np.nan

	return data_nas


########################################################################################################################
# Missing Pattern Simulation MAR
########################################################################################################################
def simulate_nan_patterns_mar_pyamp(data, ms_patterns, ms_ratio, ms_func, seed):
	if isinstance(ms_patterns, tuple):
		ms_patterns = [ms_patterns]

	if ms_func == 'left':
		ms_func = 'sigmoid-left'
	elif ms_func == 'right':
		ms_func = 'sigmoid-right'
	elif ms_func == 'mid':
		ms_func = 'sigmoid-mid'
	elif ms_func == 'tail':
		ms_func = 'sigmoid-tail'
	else:
		raise ValueError("Invalid ms_func: {}".format(ms_func))

	ret = []
	for ms_pattern in ms_patterns:
		ms_pattern = tuple(ms_pattern)
		ret.append({"incomplete_vars": ms_pattern, "mechanism": "MAR", "score_to_probability_func": ms_func})

	ma = MultivariateAmputation(prop=ms_ratio, patterns=ret, seed=seed)
	data_ms = ma.fit_transform(data)

	return data_ms


def simulate_nan_patterns_mar_quantile(
		data, ms_patterns, missing_ratio, missing_func='left', strict=True, seed=201030
):
	mask = np.zeros_like(data, dtype=bool)
	missing_ratio = missing_ratio / len(ms_patterns)

	for cols in ms_patterns:
		seed = (seed + 1203941) % (2 ^ 32 - 1)
		# find the columns that are not to be adding missing values
		obs_cols = []
		for index in range(data.shape[1]):
			if index not in cols:
				obs_cols.append(index)
		obs_cols = np.array(obs_cols)
		if len(obs_cols) == 0:
			raise ValueError(f"No columns to observe for this pattern {cols}, try to set obs to be False")

		# find most correlated col
		score_list = []
		for col in cols:
			X = data[:, obs_cols]
			score = mutual_info_regression(
				X, data[:, col], discrete_features=False, n_neighbors=5, copy=True, random_state=seed
			)
			score_list.append(score)
		score_array = np.array(score_list)
		score_array = score_array.mean(0)
		most_correlated_col = obs_cols[np.argmax(score_array)]
		data_corr = data[:, most_correlated_col]

		# get mask based on quantile
		mask = mask_mar_quantile_pattern(data, mask, cols, data_corr, missing_ratio, missing_func, strict, seed)

	# assign the missing values
	data_ms = data.copy()
	data_ms[mask] = np.nan

	return data_ms


def simulate_nan_patterns_mar_sigmoid(data, ms_patterns, missing_ratio, missing_func, k, strict=False, seed=1002031):

	mask = np.zeros(data.shape, dtype=bool)

	# add missing for each column
	for cols in ms_patterns:

		# set the seed
		seed = (seed + 1203941) % (2 ^ 32 - 1)

		##################################################################################
		# all other cols
		keep_mask = np.ones(data.shape[1], dtype=bool)
		keep_mask[list(cols)] = False
		X_rest = data[:, keep_mask]
		# na associated with all other columns

		#################################################################################
		# get k most correlated columns or all columns
		if k == 'all' or k >= X_rest.shape[1]:
			indices_obs = np.arange(data.shape[1])
			indices_obs = np.setdiff1d(indices_obs, cols)
			data_corr = data[:, indices_obs]
		else:
			score_list = []
			for col in cols:
				score = mutual_info_regression(
					X_rest, data[:, col], discrete_features=False, n_neighbors=5, copy=True, random_state=seed
				)
				score_list.append(score)
			score_array = np.array(score_list)
			score_array = score_array.mean(0)
			most_correlated_cols = np.argsort(score_array)[::-1][:k]
			data_corr = data[:, most_correlated_cols]

		#################################################################################
		# pick coefficients and mask missing values
		#################################################################################
		mask = mask_mar_sigmoid_pattern(mask, cols, data_corr, missing_ratio, missing_func, strict, seed)

	# assign the missing values
	data_ms = data.copy()
	data_ms[mask] = np.nan

	return data_ms


########################################################################################################################
# Utils
########################################################################################################################
def mask_mar_sigmoid_pattern(mask, cols, data_corr, missing_ratio, missing_func, strict, seed):
	np.random.seed(seed)
	random.seed(seed)
	#################################################################################
	# pick coefficients
	#################################################################################
	# Pick coefficients so that W^Tx has unit variance (avoids shrinking)

	# copy data and do min-max normalization
	data_copy = data_corr.copy()
	data_copy = (data_copy - data_copy.min(0, keepdims=True)) / (
				data_copy.max(0, keepdims=True) - data_copy.min(0, keepdims=True))
	data_copy = (data_copy - data_copy.mean(0, keepdims=True)) / data_copy.std(0, keepdims=True)

	coeffs = np.random.rand(data_copy.shape[1], 1)
	# print(coeffs)
	Wx = data_copy @ coeffs
	# print(Wx)
	wss = (Wx) / np.std(Wx, 0, keepdims=True)

	if missing_func == 'random':
		missing_func = random.choice(['left', 'right', 'mid', 'tail'])

	def f(x: np.ndarray) -> np.ndarray:
		if missing_func == 'left':
			return expit(-wss + x).mean().item() - missing_ratio
		elif missing_func == 'right':
			return expit(wss + x).mean().item() - missing_ratio
		elif missing_func == 'mid':
			return expit(np.absolute(wss) - 0.75 + x).mean().item() - missing_ratio
		elif missing_func == 'tail':
			return expit(-np.absolute(wss) + 0.75 + x).mean().item() - missing_ratio
		else:
			raise NotImplementedError

	intercept = optimize.bisect(f, -50, 50)

	if missing_func == 'left':
		ps = expit(-wss + intercept)
	elif missing_func == 'right':
		ps = expit(wss + intercept)
	elif missing_func == 'mid':
		ps = expit(-np.absolute(wss) + 0.75 + intercept)
	elif missing_func == 'tail':
		ps = expit(np.absolute(wss) - 0.75 + intercept)
	else:
		raise NotImplementedError

	# strict false means using random simulation
	if strict is False:
		ber = np.random.binomial(n=1, size=mask.shape[0], p=ps.flatten())
		for col in cols:
			mask[:, col] = ber
	# strict mode based on rank on calculated probability, strictly made missing
	else:
		ps = ps.flatten()
		end_value = np.sort(ps)[::-1][int(missing_ratio * data_copy.shape[0])]
		indices = np.where((ps - end_value) > 1e-3)[0]
		if len(indices) < int(missing_ratio * data_copy.shape[0]):
			end_indices = np.where(np.absolute(ps - end_value) <= 1e-3)[0]
			end_indices = np.random.choice(
				end_indices, int(missing_ratio * data_copy.shape[0]) - len(indices), replace=False
				)
			indices = np.concatenate((indices, end_indices))
		elif len(indices) > int(missing_ratio * data_copy.shape[0]):
			indices = np.random.choice(indices, int(missing_ratio * data_copy.shape[0]), replace=False)

		for col in cols:
			mask[indices, col] = True

	return mask


def mask_mar_quantile_pattern(data, mask, cols, data_corr, missing_ratio, missing_func,  strict, seed):

	# set the seed
	np.random.seed(seed)
	random.seed(seed)
	# find the quantile of the most correlated column
	if missing_func == 'random':
		missing_func = random.choice(['left', 'right', 'mid', 'tail'])

	if strict:
		total_missing = int(missing_ratio * data.shape[0])
		sorted_values = np.sort(data_corr)
		if missing_func == 'left':
			q = sorted_values[int(missing_ratio * data.shape[0]) - 1]
			indices = np.where(data_corr < q)[0]
			if len(indices) < total_missing:
				end_indices = np.where(data_corr == q)[0]
				add_up_indices = np.random.choice(
					end_indices, size=total_missing - len(indices), replace=False
				)
				na_indices = np.concatenate((indices, add_up_indices))
			elif len(indices) > total_missing:
				na_indices = np.random.choice(indices, size=total_missing, replace=False)
			else:
				na_indices = indices
		elif missing_func == 'right':
			q = sorted_values[int((1 - missing_ratio) * data.shape[0])]
			indices = np.where(data_corr > q)[0]
			if len(indices) < total_missing:
				start_indices = np.where(data_corr == q)[0]
				add_up_indices = np.random.choice(
					start_indices, size=total_missing - len(indices), replace=False
				)
				na_indices = np.concatenate((indices, add_up_indices))
			elif len(indices) > total_missing:
				na_indices = np.random.choice(indices, size=total_missing, replace=False)
			else:
				na_indices = indices
		elif missing_func == 'mid' or missing_func == 'tail':
			q0 = sorted_values[int((1 - missing_ratio) / 2 * data.shape[0])]
			q1 = sorted_values[int((1 + missing_ratio) / 2 * data.shape[0]) - 1]
			if missing_func == 'mid':
				indices = np.where((data_corr > q0) & (data_corr < q1))[0]
			else:
				indices = np.where((data_corr < q0) | (data_corr > q1))[0]
			if len(indices) < total_missing:
				end_indices_q0 = np.where(data_corr == q0)[0]
				end_indices_q1 = np.where(data_corr == q1)[0]
				end_indices = np.union1d(end_indices_q0, end_indices_q1)
				add_up_indices = np.random.choice(end_indices, size=total_missing - len(indices), replace=False)
				na_indices = np.concatenate((indices, add_up_indices))
			elif len(indices) > total_missing:
				na_indices = np.random.choice(indices, size=total_missing, replace=False)
			else:
				na_indices = indices
		else:
			raise NotImplementedError
	else:
		if missing_func == 'left':
			q0 = 0
			q1 = 0.5 if missing_ratio <= 0.5 else missing_ratio
		elif missing_func == 'right':
			q0 = 0.5 if missing_ratio <= 0.5 else 1 - missing_ratio
			q1 = 1
		elif missing_func == 'mid' or missing_func == 'tail':
			q0 = 0.25 if missing_ratio <= 0.5 else 0.5 - missing_ratio / 2
			q1 = 0.75 if missing_ratio <= 0.5 else 0.5 + missing_ratio / 2
		else:
			raise NotImplementedError

		sorted_values = np.sort(data_corr)
		q0 = sorted_values[int(q0 * data.shape[0])]
		q1 = sorted_values[int(q1 * data.shape[0]) - 1]

		if missing_func != 'tail':
			indices = np.where((data_corr >= q0) & (data_corr <= q1))[0]
		else:
			indices = np.where((data_corr <= q0) | (data_corr >= q1))[0]

		na_indices = np.random.choice(indices, size=int(missing_ratio * data.shape[0]), replace=False)

	for col in cols:
		mask[na_indices, col] = True

	return mask
