import pytest
from scipy.special import expit
from .utils import pick_coeffs, fit_intercepts
import numpy as np


def test_mar():
	data = np.array(
		[
			[0, 0, 0, 0, 0, 0, 0, 1, 2, 3],
			[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
			[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
			#[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
		]
	)
	data = data.T
	idx_col_na = np.array([0])
	idx_col_obs = np.array([1, 2])
	coeffs = pick_coeffs(data, idx_col_obs, idx_col_na, False, 24)
	print(coeffs)
	intercepts = fit_intercepts(data[:, idx_col_obs], coeffs, 0.5)
	print(intercepts)
	ps = expit(data[:, idx_col_obs] @ coeffs + intercepts)
	print(ps)

	ber = np.random.rand(10, 1)
	mask = np.zeros((10, 4), dtype=bool)
	mask[:, idx_col_na] = ber < ps
	print(mask)
	assert True
