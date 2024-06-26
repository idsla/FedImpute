from ..load_config import process_experiment_marker


def test_process_experiment_marker():

	# Test 1
	# --------------------------------------------------------------------------------------------
	config_tmpl = {
		"fed_imp": {
			"n_rounds": 10,
			"n_clients": 10,
			"n_epochs": 10,
			"n_clients_per_round": 10,
			"n_clients_per_rounds": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
		},
		"dataset_name": 'breast'
	}

	experiment_marker = ['dataset_name', 'fed_imp@n_rounds', 'fed_imp@n_clients']

	marker = process_experiment_marker(experiment_marker, config_tmpl)
	print(marker)

	# Test 2
	# --------------------------------------------------------------------------------------------
	config_tmpl = {
		"fed_imp": {
			"n_rounds": 10,
		}
	}

	experiment_marker = []
	marker = process_experiment_marker(experiment_marker, config_tmpl)
	print(marker)

	assert True
