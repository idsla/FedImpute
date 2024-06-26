import numpy as np

from ..strategy.fedwavgcl import fedwavgcl


def test_fedwavgcl():
	# create unit test for fedwavgcl
	weights = {
		'c1': [0.1, 0.1, 0.1, 0.1],
		'c2': [0.2, 0.2, 0.2, 0.2],
		'c3': [0.4, 0.4, 0.4, 0.4],
		'c4': [0.5, 0.5, 0.5, 0.5]
	}
	losses = {
		'c1': {'rmse':0.1} ,
		'c2': {'rmse': 0.14},
		'c3': {'rmse': 0.3},
		'c4': {'rmse':0.34}
	}
	missing_infos = {
		'c1': {'missing_cell_pct': 0.1, 'total_missing_cell_pct': 0.1, 'missing_row_pct': 0.1, 'total_missing_row_pct': 0.1},
		'c2': {'missing_cell_pct': 0.2, 'total_missing_cell_pct': 0.2, 'missing_row_pct': 0.2, 'total_missing_row_pct': 0.2},
		'c3': {'missing_cell_pct': 0.5, 'total_missing_cell_pct': 0.5, 'missing_row_pct': 0.5, 'total_missing_row_pct': 0.5},
		'c4': {'missing_cell_pct': 0.6, 'total_missing_cell_pct': 0.6, 'missing_row_pct': 0.6, 'total_missing_row_pct': 0.6}
	}
	loss_field = 'rmse'
	ms_field = 'missing_cell_pct'
	agg_weight = fedwavgcl(weights, losses, missing_infos, loss_field, ms_field)
	print(agg_weight)

	assert True
