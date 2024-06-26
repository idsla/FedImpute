
def clean_result(ret_dict):
	cleaned_ret = dict()
	# add configuration info
	config = ret_dict['params']['config']
	cleaned_ret.update(
		{
			'dataset_name': config['data']['dataset_name'],
			"n_clients": config['num_clients'],
			'data_partition_strat': config['data_partition']['strategy'],
			'ms_ratio_strat': config['missing_simulate']['ms_ratio']['strategy'],
			'ms_ratio': config['missing_simulate']['ms_ratio']['params']['ratio'],
			'ms_features_strat': config['missing_simulate']['ms_features']['strategy'],
			'p_missing': config['missing_simulate']['ms_features']['params']['ratio'],
			'ms_mechanism_strat': config['missing_simulate']['ms_mechanism']['strategy'],
			"ms_mechnism": config['missing_simulate']['ms_mechanism']['params']['mechanism'],
			"agg_methods": config['agg_strategy_imp']['strategy'],
		}
	)

	cleaned_ret["other_params"] = {
		"local_epochs": config['server']['imp_local_epochs'],
		"imp_mode": config['server']['impute_mode'],
		"rounds": config['server']['imp_round'],
		"initial_strategy_num": config["imputation"]['initial_strategy_num'],
		"initial_strategy_cat": config["imputation"]['initial_strategy_cat'],
		"estimator_num": config["imputation"]['estimator_num'],
		"estimator_cat": config["imputation"]['estimator_cat'],
		"clip": config["imputation"]['clip']
	}

	# results
	results = ret_dict['results']["avg_rets_final"]
	cleaned_ret.update(
		{
			"acc": results['accuracy'],
			"rmse": results['imp@rmse'],
			"w2": results['imp@w2'],
		}
	)

	return cleaned_ret