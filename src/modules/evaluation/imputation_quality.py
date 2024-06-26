from typing import Optional, Union, Any

import numpy as np
from scipy.stats import wasserstein_distance
import ot

def mae(X: np.ndarray, X_true: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Mean Absolute Error (MAE) between imputed variables and ground truth.
	Args:
		X : Data with imputed variables.
		X_true : Ground truth.
		mask : Missing value mask (missing if True)
	Returns:
		MAE : np.ndarray
	"""
	mask_ = mask.astype(bool)
	return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()


def rmse(X: np.ndarray, X_true: np.ndarray, mask: np.ndarray) -> Union[int, Any]:
	"""
	Root Mean Squared Error (MAE) between imputed variables and ground truth
	Args:
		X : Data with imputed variables.
		X_true : Ground truth.
		mask : Missing value mask (missing if True)
	Returns:
		RMSE : np.ndarray
	"""
	mask_ = mask.astype(bool)
	if mask_.sum() == 0:
		return 0
	else:
		# print("="*100)
		# for col in range(X.shape[1]):
		# 	print(col, np.sqrt(((X[:, col][mask_[:, col]] - X_true[:, col][mask_[:, col]]) ** 2).sum() / mask_[:, col].sum()))
		return np.sqrt(((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum())


def ws_cols(X: np.ndarray, X_true: np.ndarray) -> int:
	res = 0
	for idx in range(X_true.shape[1]):
		res += wasserstein_distance(X_true[:, idx], X[:, idx])
	return res


def sliced_ws(X: np.ndarray, X_true: np.ndarray) -> np.ndarray:
	rets = []
	for i in range(10):
		rets.append(ot.sliced_wasserstein_distance(X_true, X, seed = i*102930 + 109099))
	return np.mean(rets)


__all__ = ["mae", "rmse", "ws_cols", "sliced_ws"]
