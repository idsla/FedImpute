import numpy as np
import random
from src.modules.sampling import generate_dirichlet_noniid_distribution, generate_alphas


# generate missing strategy for each client
def missing_strategy(params, n_clients, seed=201030):

	random.seed(seed)
	np.random.seed(seed)

	# missing ratio
	missing_ratio_strategy = params['ms_ratio']['strategy']
	missing_ratio_strategy_params = params['ms_ratio']['params']
	ret_mr = missing_ratio_distribution(missing_ratio_strategy, missing_ratio_strategy_params, n_clients)

	# missing features
	missing_features_strategy = params['ms_features']['strategy']
	missing_features_strategy_params = params['ms_features']['params']
	ret_mf = missing_features_distribution(missing_features_strategy, missing_features_strategy_params, n_clients)

	# missing mechanism
	missing_mechanism_strategy = params['ms_mechanism']['strategy']
	missing_mechanism_strategy_params = params['ms_mechanism']['params']
	ret_mech = missing_mechanism_distribution(missing_mechanism_strategy, missing_mechanism_strategy_params, n_clients)

	# merge results
	ret = [{**ret_mr[i], **ret_mf[i], **ret_mech[i]} for i in range(n_clients)]

	return ret


def missing_ratio_distribution(strategy, strategy_params, num_clients):
	if strategy == 'uniform':
		missing_ratio = strategy_params['ratio']
		return [{"missing_ratio": missing_ratio} for _ in range(num_clients)]
	elif strategy == 'dirichlet':
		alphas = generate_alphas(1, num_clients)
		missing_ratios = generate_dirichlet_noniid_distribution(alphas)
		return missing_ratios
	elif strategy == 'random':
		missing_ratio_range = strategy_params['ratio_range']
		return [{"missing_ratio": random.uniform(missing_ratio_range[0], missing_ratio_range[1])}
		        for _ in range(num_clients)]
	elif strategy == "two-clusters-19":
		N = num_clients  # total number of clients
		clients = np.ones(N)  # client ids
		groups = np.array_split(clients, 2)  # split clients into 3 groups
		groups[0] = groups[0] * 0.1  # assign weight 0.3 to the first group
		groups[1] = groups[1] * 0.9  # assign weight 0.6 to the second group
		weights = np.concatenate(groups)  # concatenate the groups

		# Convert the weights to a list of dictionaries
		return [{"missing_ratio": w} for w in weights]
	elif strategy == 'three-clusters-369':
		N = num_clients  # total number of clients
		clients = np.ones(N)  # client ids
		groups = np.array_split(clients, 3)  # split clients into 3 groups
		groups[0] = groups[0] * 0.3  # assign weight 0.3 to the first group
		groups[1] = groups[1] * 0.6  # assign weight 0.6 to the second group
		groups[2] = groups[2] * 0.9  # assign weight 0.9 to the third group
		weights = np.concatenate(groups)  # concatenate the groups

		# Convert the weights to a list of dictionaries
		return [{"missing_ratio": w} for w in weights]
	else:
		raise NotImplementedError('missing ratio distribution strategy not implemented')


def missing_features_distribution(strategy, strategy_params, num_clients):
	if strategy == 'identical-sample':
		p_missing = strategy_params['ratio']
		return [{"p_na_cols": p_missing} for _ in range(num_clients)]
	elif strategy == 'identical-important':
		p_missing = strategy_params['ratio']
		return [{"p_na_cols": p_missing, "use_important_feature": True} for _ in range(num_clients)]
	elif strategy == 'random':
		ratio_range = strategy_params['ratio_range']
		return [{"p_na_cols": random.uniform(ratio_range[0], ratio_range[1]), "sample_columns": True}
		        for _ in range(num_clients)]
	elif strategy == 'dirichlet':
		raise NotImplementedError
	elif strategy == 'cluster':
		N = num_clients  # total number of clients
		n = N // 3  # number of clients in each group
		remainder = N % 3  # number of clients in the last group

		# Create an array of weights for the first two groups of clients
		weights = np.repeat([0.3, 0.6, 0.9], n)[:N - remainder]

		# Add the remaining clients to the last group with a weight of 0.9
		weights = np.concatenate((weights, np.repeat(0.9, remainder)))

		# Convert the weights to a list of dictionaries
		return [{"p_na_cols": w, "sample_columns": True} for w in weights]
	else:
		raise ValueError('missing features distribution strategy not supported')

	# 0.3, 0.6, 0.9 - 3 groups


def missing_mechanism_distribution(strategy, strategy_params, num_clients):
	if strategy == 'identical':
		missing_mechanism = strategy_params['mechanism']
		return [{"mechanism": missing_mechanism} for _ in range(num_clients)]
	elif strategy == 'random':
		missing_mechanisms = ['MCAR', 'MAR', 'MARY', 'MNAR', 'MNARY']
		return [{"mechanism": random.choice(missing_mechanisms)} for _ in range(num_clients)]
	elif strategy == 'clustered':
		raise NotImplementedError
	else:
		raise ValueError('missing mechanism distribution strategy not supported')
