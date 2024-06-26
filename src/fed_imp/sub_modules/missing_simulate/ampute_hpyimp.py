from typing import Union

import numpy as np
from scipy.special import expit
from .utils import pick_coeffs, fit_intercepts
import random


########################################################################################################################
# Utils
########################################################################################################################
def MCAR_mask(X: np.ndarray, important_feature_list, p: float, p_na_cols: float, seed: int) -> np.ndarray:
	"""
	Missing completely at random mechanism. Missing values are generated independently for each entry of the data
	matrix.
	Args:
		X : Data for which missing values will be simulated.
		p : Proportion of missing values to generate.
		p_na_cols: Proportion of columns with missing values.
	Returns:
		mask : Mask of generated missing values (True if the value is missing).
	"""
	n, d = X.shape[0], X.shape[1] - 1
	if important_feature_list is not None:
		d = len(important_feature_list)
	d_na = int(np.clip(int(p_na_cols * d), 1, d))
	rng = np.random.default_rng(seed)
	if important_feature_list is not None:
		idxs_nas = rng.choice(np.array(important_feature_list), d_na, replace=False)
	else:
		idxs_nas = rng.choice(d, d_na, replace=False)
	mask = np.zeros_like(X, dtype=bool)
	for col_idx in idxs_nas:
		indices = np.arange(n)
		seed = (seed + 102989221) % 1000000007
		rng = np.random.default_rng(seed)
		na_indices = rng.choice(indices, int(p * n), replace=False)
		mask[na_indices, col_idx] = True
	return mask


def MAR_mask(
		X: np.ndarray,
		important_feature_list: Union[list, None],
		p: float,
		p_na_cols: float,
		use_y: bool = False,
		seed: int = 201030,
) -> np.ndarray:
	"""
	Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
	randomly selected. The remaining variables have missing values according to a logistic model with random weights,
	re-scaled to attain the desired proportion of missing values on those variables.
	Args:
	    important_feature_list: important features to be considered
	    seed: random seed
		X : Data for which missing values will be simulated.
		p : Proportion of missing values to generate for variables which will have missing values.
		p_na_cols : Proportion of variables with *no* missing values that will be used for the logistic masking model.
		sample_columns: whether to sample columns with missing data
		use_y: whether to use the last column as a predictor of missingness
	Returns:
		mask : Mask of generated missing values (True if the value is missing).
	"""

	n, d = X.shape[0], X.shape[1] - 1
	if important_feature_list is not None:
		d = len(important_feature_list)

	np.random.seed(seed)
	mask = np.zeros((X.shape[0], X.shape[1])).astype(bool)

	if not use_y:
		d_na = np.clip(int(p_na_cols * d), 1, d - 1)
	else:
		d_na = np.clip(int(p_na_cols * d), 1, d)

	if important_feature_list is not None:
		indices_nas = np.random.choice(np.array(important_feature_list), d_na, replace=False)
	else:
		indices_nas = np.random.choice(d, d_na, replace=False)

	# Sample variables that will all be observed, and those with missing values:
	if use_y:
		indices_obs = np.array([i for i in range(X.shape[1]) if i not in indices_nas])
	else:
		indices_obs = np.array([i for i in range(X.shape[1] - 1) if i not in indices_nas])

	# Other variables will have NA proportions that depend on those observed variables, through a logistic model
	# The parameters of this logistic model are random.
	# Pick coefficients so that W^Tx has unit variance (avoids shrinking)
	coeffs = pick_coeffs(X, indices_obs, indices_nas)
	# Pick the intercepts to have a desired amount of missing values
	intercepts = fit_intercepts(X[:, indices_obs], coeffs, p)

	ps = expit(X[:, indices_obs] @ coeffs + intercepts)

	ber = np.random.rand(n, d_na)
	mask[:, indices_nas] = ber < ps

	return mask


def MNAR_mask_logistic(
		X: np.ndarray, p: float, p_params: float = 0.3, exclude_inputs: bool = True
) -> np.ndarray:
	"""
	Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
	(i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that
	are
	inputs can also be missing.
	(ii) Variables are split into a set of intputs for a logistic model, and a set whose missing probabilities are
	determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
	depend on masked values.
	In either case, weights are random and the intercept is selected to attain the desired proportion of missing
	values.
	Args:
		X : Data for which missing values will be simulated.
		p : Proportion of missing values to generate for variables which will have missing values.
		p_params : Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).
		exclude_inputs : True: mechanism (ii) is used, False: (i)
	Returns:
		mask : Mask of generated missing values (True if the value is missing).
	"""

	n, d = X.shape

	mask = np.zeros((n, d)).astype(bool)

	d_params = (
		max(int(p_params * d), 1) if exclude_inputs else d
	)  # number of variables used as inputs (at least 1)
	d_na = (
		d - d_params if exclude_inputs else d
	)  # number of variables masked with the logistic model

	# Sample variables that will be parameters for the logistic regression:
	idxs_params = (
		np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
	)
	idxs_nas = (
		np.array([i for i in range(d) if i not in idxs_params])
		if exclude_inputs
		else np.arange(d)
	)

	# Other variables will have NA proportions selected by a logistic model
	# The parameters of this logistic model are random.

	# Pick coefficients so that W^Tx has unit variance (avoids shrinking)
	coeffs = pick_coeffs(X, idxs_params, idxs_nas)
	# Pick the intercepts to have a desired amount of missing values
	intercepts = fit_intercepts(X[:, idxs_params], coeffs, p)

	ps = expit(X[:, idxs_params] @ coeffs + intercepts)

	ber = np.random.rand(n, d_na)
	mask[:, idxs_nas] = ber < ps

	# If the inputs of the logistic model are excluded from MNAR missingness,
	# mask some values used in the logistic model at random.
	# This makes the missingness of other variables potentially dependent on masked values

	if exclude_inputs:
		mask[:, idxs_params] = np.random.rand(n, d_params) < p

	return mask


def MNAR_self_mask_logistic(X: np.ndarray, p: float) -> np.ndarray:
	"""
	Missing not at random mechanism with a logistic self-masking model. Variables have missing values probabilities
	given by a logistic model, taking the same variable as input (hence, missingness is independent from one variable
	to another). The intercepts are selected to attain the desired missing rate.
	Args:
		X : Data for which missing values will be simulated.
		p : Proportion of missing values to generate for variables which will have missing values.
	Returns:
		mask : Mask of generated missing values (True if the value is missing).
	"""

	n, d = X.shape

	# Variables will have NA proportions that depend on those observed variables, through a logistic model
	# The parameters of this logistic model are random.

	# Pick coefficients so that W^Tx has unit variance (avoids shrinking)
	coeffs = pick_coeffs(X, self_mask=True)
	# Pick the intercepts to have a desired amount of missing values
	intercepts = fit_intercepts(X, coeffs, p, self_mask=True)

	ps = expit(X * coeffs + intercepts)

	ber = np.random.rand(n, d)
	mask = ber < ps

	return mask


def MNAR_mask_quantiles(
		X: np.ndarray,
		p: float,
		q: float,
		p_params: float,
		cut: str = "both",
		MCAR: bool = False,
) -> np.ndarray:
	"""
	Missing not at random mechanism with quantile censorship. First, a subset of variables which will have missing
	variables is randomly selected. Then, missing values are generated on the q-quantiles at random. Since
	missingness depends on quantile information, it depends on masked values, hence this is a MNAR mechanism.
	Args:
		X : Data for which missing values will be simulated.
		p : Proportion of missing values to generate for variables which will have missing values.
		q : Quantile level at which the cuts should occur
		p_params : Proportion of variables that will have missing values
		cut : 'both', 'upper' or 'lower'. Where the cut should be applied. For instance, if q=0.25 and cut='upper',
		then missing values will be generated in the upper quartiles of selected variables.
		MCAR : If true, masks variables that were not selected for quantile censorship with a MCAR mechanism.
	Returns:
		mask : Mask of generated missing values (True if the value is missing).
	"""
	n, d = X.shape

	mask = np.zeros((n, d)).astype(bool)

	d_na = max(int(p_params * d), 1)  # number of variables that will have NMAR values

	# Sample variables that will have imps at the extremes
	idxs_na = np.random.choice(
		d, d_na, replace=False
	)  # select at least one variable with missing values

	# check if values are greater/smaller that corresponding quantiles
	if cut == "upper":
		quants = np.quantile(X[:, idxs_na], 1 - q, dim=0)
		m = X[:, idxs_na] >= quants
	elif cut == "lower":
		quants = np.quantile(X[:, idxs_na], q, dim=0)
		m = X[:, idxs_na] <= quants
	elif cut == "both":
		u_quants = np.quantile(X[:, idxs_na], 1 - q, axis=0)
		l_quants = np.quantile(X[:, idxs_na], q, axis=0)
		m = (X[:, idxs_na] <= l_quants) | (X[:, idxs_na] >= u_quants)

	# Hide some values exceeding quantiles
	ber = np.random.rand(n, d_na)
	mask[:, idxs_na] = (ber < p) & m

	if MCAR:
		# Add a mcar mecanism on top
		mask = mask | (np.random.rand(n, d) < p)

	return mask
