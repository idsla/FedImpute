from scipy import optimize
from typing import List
from scipy.special import expit
import numpy as np


def pick_coeffs(
		X: np.ndarray,
		idxs_obs=None,
		idxs_nas=None,
		self_mask: bool = False,
		seed=201030
) -> np.ndarray:
	if idxs_nas is None:
		idxs_nas = []
	if idxs_obs is None:
		idxs_obs = []

	n, d = X.shape
	np.random.seed(seed)
	if self_mask:
		coeffs = -np.random.rand(d)
		Wx = X * coeffs
		coeffs /= np.std(Wx, 0)
	else:
		d_obs = len(idxs_obs)
		d_na = len(idxs_nas)
		coeffs = np.random.rand(d_obs, d_na)
		Wx = X[:, idxs_obs] @ coeffs
		coeffs /= np.std(Wx, 0, keepdims=True)
	return coeffs


def fit_intercepts(
		X: np.ndarray, coeffs: np.ndarray, p: float, self_mask: bool = False
) -> np.ndarray:

	if self_mask:
		d = len(coeffs)
		intercepts = np.zeros(d)
		for j in range(d):
			def f(x: np.ndarray) -> np.ndarray:
				return expit(X * coeffs[j] + x).mean().item() - p

			intercepts[j] = optimize.bisect(f, -50, 50)
	else:
		d_obs, d_na = coeffs.shape
		intercepts = np.zeros(d_na)
		for j in range(d_na):
			def f(x: np.ndarray) -> np.ndarray:
				return expit(np.dot(X, coeffs[:, j]) + x).mean().item() - p

			intercepts[j] = optimize.bisect(f, -50, 50)
	return intercepts





def show_missing_information(clients):
	import missingno
	import matplotlib.pyplot as plt
	for name, client in clients.items():
		df = client.X_train_ms_df
		fig, axes = plt.subplots(1, 2, figsize=(16, 12), squeeze=False)
		axes[0, 0].set_title("Missing proportion")
		missingno.bar(df, fontsize=8, color='lightblue', ax=axes[0, 0])
		axes[0, 1].set_title("Missing matrix")
		missingno.matrix(df, fontsize=8, ax=axes[0, 1], sparkline=False)
		plt.tight_layout()
		plt.title(name)
		plt.show()
  

MECH_NAME_MAPPING = {
    'mcar': 'MCAR',
    'mar_quantile': 'MAR (Quantile)',
    'mar_logit': 'MAR (Logit)',
    'mnar_quantile': 'MNAR (Quantile)',
    'mnar_logit': 'MNAR (Logit)',
    'mnar_sm_logit': 'MNAR (Self Masking Logit)'
}
