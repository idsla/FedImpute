import random


#####################################################################################################################
def generate_missing_add_config(params):
    # dataset_config and seed
    dataset_config = params['dataset_config']
    seed = params['seed']
    random.seed(seed)

    # firstly select incomplete vars
    params['incomplete_vars']['dataset_config'] = dataset_config
    incomplete_vars = select_incomplete_vars_strategy(**params['incomplete_vars'])

    ret = {}
    for var in incomplete_vars:
        missing_config = {
            'missing_mechanism': params['missing_mechanism'],
            'missing_ratio': params['missing_ratio']
        }

        if params['missing_mechanism'] == 'mcar':
            pass

        elif params['missing_mechanism'] == 'mar':
            # select associated vars
            params['associated_vars_strategy']['current_feature_idx'] = var
            params['associated_vars_strategy']['dataset_config'] = dataset_config
            params['associated_vars_strategy']['seed'] = seed + var
            associated_vars = select_associated_vars(**params['associated_vars_strategy'])

            # pick associated vars range
            associated_vars_range = []
            for associated_var in associated_vars:
                params['associated_vars_range_strategy']['associated_var'] = associated_var
                params['associated_vars_range_strategy']['data_config'] = dataset_config
                params['associated_vars_range_strategy']['seed'] = seed + associated_var
                associated_var_range = select_associated_vars_range(
                    **params['associated_vars_range_strategy']
                )
                associated_vars_range.append(associated_var_range)

            missing_config['associated_vars'] = associated_vars
            missing_config['associated_vars_range'] = associated_vars_range

        elif params['missing_mechanism'] == 'mnar':
            associated_vars = [var]
            params['associated_vars_range_strategy']['associated_var'] = var
            params['associated_vars_range_strategy']['data_config'] = dataset_config
            params['associated_vars_range_strategy']['seed'] = seed + var
            associated_vars_range = [
                select_associated_vars_range(**params['associated_vars_range_strategy'])
            ]

            missing_config['associated_vars'] = associated_vars
            missing_config['associated_vars_range'] = associated_vars_range

        ret[var] = missing_config

    return ret


#####################################################################################################
def select_incomplete_vars_strategy(
        strategy, num_to_select=0, selected_features=None, dataset_config=None
):
    if strategy == 'random':
        features_idx = dataset_config['features_idx']
        selected_features = random.sample(features_idx, num_to_select)
        return selected_features
    elif strategy == 'exploration':
        return selected_features
    else:
        raise ValueError('Invalid strategy name: {}'.format(strategy))


def select_associated_vars(
        strategy, current_feature_idx, dataset_config, num_to_select=0, include_target=False,
        associated_features=None, seed = 0
):
    features_idx = dataset_config['features_idx'].copy()
    target_index = len(features_idx)
    features_idx.remove(current_feature_idx)
    random.seed(seed)
    if strategy == 'random':
        if include_target:
            num_to_select -= 1
            associated_features = random.sample(features_idx, num_to_select) + [target_index]
        else:
            associated_features = random.sample(features_idx, num_to_select)
        return associated_features
    elif strategy == 'exploration':
        return associated_features
    else:
        raise ValueError('Invalid strategy name: {}'.format(strategy))


def select_associated_vars_range(
        strategy, associated_var, data_config, seed=None
):
    random.seed(seed)
    target_index = len(data_config['features_idx'])
    if strategy == 'random_strategy' or strategy == 'random':
        if associated_var == target_index:
            associated_var_range = [random.choice([0, 1])]
        else:
            if strategy == 'random_strategy':
                num_range_strategy = random.choice(['random', 'upper', 'lower'])
            else:
                num_range_strategy = strategy
            associated_var_range = get_percentiles_from_strategy(num_range_strategy)
    elif strategy == 'upper':
        if associated_var == target_index:
            associated_var_range = [1]
        else:
            associated_var_range = (0.5, 1)
    elif strategy == 'lower':
        if associated_var == target_index:
            associated_var_range = [0]
        else:
            associated_var_range = (0, 0.5)
    else:
        raise ValueError('Invalid strategy name: {}'.format(strategy))

    return associated_var_range


def get_percentiles_from_strategy(strategy):
    if strategy == 'random':
        lower_percentile = random.uniform(0, 1)
        upper_percentile = lower_percentile + 0.5
        if upper_percentile > 1:
            upper_percentile = upper_percentile - 1
        return lower_percentile, upper_percentile
    elif strategy == 'upper':
        return 0.5, 1
    elif strategy == 'lower':
        return 0, 0.5
