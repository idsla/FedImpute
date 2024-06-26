from ..missing_simulate.missing_scenario import n_cluster_mm


# create test case for n_cluster_mm_mar
def test_n_cluster_mm():
	n_cluster = 3
	mechanisms = [2, 3, 4]
	ms_ratio = 0.5
	n_clients = 10
	cols = [1, 2, 3, 4, 5]
	seed = 0
	expected = [
		{'missing_ratio': 0.5, 'missing_mechanism': 'mar_quantile_left', 'missing_features': [1, 2, 3, 4, 5]},
		{'missing_ratio': 0.5, 'missing_mechanism': 'mar_quantile_left', 'missing_features': [1, 2, 3, 4, 5]},
		{'missing_ratio': 0.5, 'missing_mechanism': 'mar_quantile_left', 'missing_features': [1, 2, 3, 4, 5]},
		{'missing_ratio': 0.5, 'missing_mechanism': 'mar_quantile_left', 'missing_features': [1, 2, 3, 4, 5]},
		{'missing_ratio': 0.5, 'missing_mechanism': 'mar_quantile_right', 'missing_features': [1, 2, 3, 4, 5]},
		{'missing_ratio': 0.5, 'missing_mechanism': 'mar_quantile_right', 'missing_features': [1, 2, 3, 4, 5]},
		{'missing_ratio': 0.5, 'missing_mechanism': 'mar_quantile_right', 'missing_features': [1, 2, 3, 4, 5]},
		{'missing_ratio': 0.5, 'missing_mechanism': 'mar_sigmoid_left', 'missing_features': [1, 2, 3, 4, 5]},
		{'missing_ratio': 0.5, 'missing_mechanism': 'mar_sigmoid_left', 'missing_features': [1, 2, 3, 4, 5]},
		{'missing_ratio': 0.5, 'missing_mechanism': 'mar_sigmoid_left', 'missing_features': [1, 2, 3, 4, 5]}]
	actual = n_cluster_mm(n_cluster, mechanisms, ms_ratio, n_clients, cols, seed)
	assert actual == expected
