import numpy as np
from .ms_simulate import mcar_simulate, mar_simulate, mnar_simulate
from .missing_scenario import load_scenario1, load_scenario2, load_scenario3


def add_missing(train_data_list, scenario, cols, seed=201030):
	mf_strategy = scenario['mf_strategy']
	if "mr_strategy" not in scenario:
		mm_strategy = scenario['mm_strategy_new']
		ret = load_scenario3(
			n_clients=len(train_data_list), cols=cols, mm_strategy=mm_strategy, seed=seed
		)
	else:
		mr_strategy = scenario['mr_strategy']
		mm_strategy = scenario['mm_strategy']
		ret = load_scenario2(
			n_clients=len(train_data_list), cols=cols, mr_strategy=mr_strategy, mf_strategy=mf_strategy,
			mm_strategy=mm_strategy, seed=seed
		)

		format_ret = [i['missing_mechanism'] + '@' + str(i['missing_ratio']) for i in ret]
		print(format_ret)

	train_ms_list = []
	for i in range(len(train_data_list)):
		data = train_data_list[i]
		X_train, y_train = data[:, :-1], data[:, -1]
		missing_ratios = ret[i]['missing_ratio']
		missing_mechanisms = ret[i]['missing_mechanism']
		missing_features = ret[i]['missing_features']
		seed = (seed + i * 10089) % (2 ^ 32 - 1)
		X_train_ms = simulate_nan_new(X_train, y_train, missing_features, missing_ratios, missing_mechanisms, seed)
		train_ms_list.append(np.concatenate([X_train_ms, y_train.reshape(-1, 1)], axis=1).copy())

	return train_ms_list


########################################################################################################################
# Simulate missing for one client
########################################################################################################################
def simulate_nan_new(
		X_train, y_train, cols, missing_ratio, mechanism='mcar', seed=201030
):
	if isinstance(mechanism, list):
		if mechanism[0].startswith('mnar_quantile'):
			mechanism_truncated = [item.split('_')[-1] for item in mechanism]
			data_ms = mnar_simulate.simulate_nan_mnar_quantile(
				X_train, cols, missing_ratios = missing_ratio, missing_funcs=mechanism_truncated, seed=seed)
			X_train_ms = data_ms
   
		elif mechanism[0].startswith('mnar_sigmoid'):

			beta_option = 'sphere'   # can also set to 'b1', 'b2', 'sphere2'
			strict = True
			corr_type = 'allk0.25'   # most 25% correlated features
			mechanism_truncated = [item.split('_')[-1] for item in mechanism]
			data_ms = mnar_simulate.simulate_nan_mnar_sigmoid(
				X_train, cols, missing_ratios=missing_ratio, missing_funcs=mechanism_truncated, strict = strict,
				corr_type=corr_type, seed=seed, beta_corr=beta_option
			)
			X_train_ms = data_ms
		
		else:
			raise NotImplementedError
	else:
		if mechanism == 'mcar':
			X_train_ms = mcar_simulate.simulate_nan_mcar(X_train, cols, missing_ratio, seed)
		elif mechanism == 'mar_quantile_left':
			if len(cols) == X_train.shape[1]:
				cols = np.arange(0, X_train.shape[1]-1)
			X_train_ms = mar_simulate.simulate_nan_mar_quantile(
				X_train, cols, missing_ratio, missing_func='left', obs=True, seed=seed
			)
		elif mechanism == 'mar_quantile_right':
			if len(cols) == X_train.shape[1]:
				cols = np.arange(0, X_train.shape[1]-1)
			X_train_ms = mar_simulate.simulate_nan_mar_quantile(
				X_train, cols, missing_ratio, missing_func='right', obs=True, seed=seed
			)
		elif mechanism == 'mar_quantile_mid':
			if len(cols) == X_train.shape[1]:
				cols = np.arange(0, X_train.shape[1]-1)
			X_train_ms = mar_simulate.simulate_nan_mar_quantile(
				X_train, cols, missing_ratio, missing_func='mid', obs=True, seed=seed
			)
		elif mechanism == 'mar_quantile_tail':
			if len(cols) == X_train.shape[1]:
				cols = np.arange(0, X_train.shape[1]-1)
			X_train_ms = mar_simulate.simulate_nan_mar_quantile(
				X_train, cols, missing_ratio, missing_func='tail', obs=True, seed=seed
			)
		elif mechanism == 'mar_sigmoid_left':
			if len(cols) == X_train.shape[1]:
				cols = np.arange(0, X_train.shape[1]-1)
			X_train_ms = mar_simulate.simulate_nan_mar_sigmoid(
				X_train, cols, missing_ratio, missing_func='right', obs=True, k='all', seed=seed
			)
		elif mechanism == 'mar_sigmoid_right':
			if len(cols) == X_train.shape[1]:
				cols = np.arange(0, X_train.shape[1]-1)
			X_train_ms = mar_simulate.simulate_nan_mar_sigmoid(
				X_train, cols, missing_ratio, missing_func='right', obs=True, k='all', seed=seed
			)
		elif mechanism == 'mar_sigmoid_mid':
			if len(cols) == X_train.shape[1]:
				cols = np.arange(0, X_train.shape[1]-1)
			X_train_ms = mar_simulate.simulate_nan_mar_sigmoid(
				X_train, cols, missing_ratio, missing_func='mid', obs=True, k='all', seed=seed
			)
		elif mechanism == 'mar_sigmoid_tail':
			if len(cols) == X_train.shape[1]:
				cols = np.arange(0, X_train.shape[1] - 1)
			X_train_ms = mar_simulate.simulate_nan_mar_sigmoid(
				X_train, cols, missing_ratio, missing_func='tail', obs=True, k='all', seed=seed
			)
		elif mechanism == 'mary_left':
			data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
			data_ms = mar_simulate.simulate_nan_mary_quantile(data, cols, missing_ratio, missing_func='left', seed=seed)
			X_train_ms = data_ms[:, :-1]
		elif mechanism == 'mary_right':
			data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
			data_ms = mar_simulate.simulate_nan_mary_quantile(data, cols, missing_ratio, missing_func='right', seed=seed)
			X_train_ms = data_ms[:, :-1]
		elif mechanism == 'mary_sigmoid_left':
			data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
			data_ms = mar_simulate.simulate_nan_mary_sigmoid(data, cols, missing_ratio, missing_func='left', seed=seed)
			X_train_ms = data_ms[:, :-1]
		elif mechanism == 'mary_sigmoid_right':
			data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
			data_ms = mar_simulate.simulate_nan_mary_sigmoid(data, cols, missing_ratio, missing_func='right', seed=seed)
			X_train_ms = data_ms[:, :-1]
		elif mechanism == 'mary_mid':
			data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
			data_ms = mar_simulate.simulate_nan_mary_quantile(data, cols, missing_ratio, missing_func='mid', seed=seed)
			X_train_ms = data_ms[:, :-1]
		elif mechanism == 'mary_tail':
			data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
			data_ms = mar_simulate.simulate_nan_mary_quantile(data, cols, missing_ratio, missing_func='tail', seed=seed)
			X_train_ms = data_ms[:, :-1]
		elif mechanism == 'mnar_sigmoid_left':
			data_ms = mnar_simulate.simulate_nan_mnar_sigmoid(X_train, cols, missing_ratio, missing_func='left', seed=seed)
			X_train_ms = data_ms
		elif mechanism == 'mnar_sigmoid_right':
			data_ms = mnar_simulate.simulate_nan_mnar_sigmoid(
				X_train, cols, missing_ratio, missing_func='right', seed=seed)
			X_train_ms = data_ms
		elif mechanism == 'mnar_quantile_left':
			data_ms = mnar_simulate.simulate_nan_mnar_quantile(
				X_train, cols, missing_ratios = missing_ratio, missing_funcs='left', seed=seed)
			X_train_ms = data_ms
		elif mechanism == 'mnar_quantile_right':
			data_ms = mnar_simulate.simulate_nan_mnar_quantile(
				X_train, cols, missing_ratios = missing_ratio, missing_funcs='right', seed=seed)
			X_train_ms = data_ms
		elif mechanism == 'mnar_quantile_mid':
			data_ms = mnar_simulate.simulate_nan_mnar_quantile(
				X_train, cols, missing_ratios = missing_ratio, missing_funcs='mid', seed=seed)
			X_train_ms = data_ms
		elif mechanism == 'mnar_quantile_tail':
			data_ms = mnar_simulate.simulate_nan_mnar_quantile(
				X_train, cols, missing_ratios = missing_ratio, missing_funcs='tail', seed=seed)
			X_train_ms = data_ms
		else:
			raise NotImplementedError

	return X_train_ms
