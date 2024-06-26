def test_add_missing():

    # missing config
    # MCAR
    features = list(range(10))
    missing_ratio = 0.1
    vars_missing_config = {}
    for feature_idx in features:
        vars_missing_config[feature_idx] = {
            'missing_mechanism': 'mcar',
            'missing_ratio': missing_ratio,
            'associated_vars': None,
            'associated_percentiles': None
        }

    # MAR
    features = list(range(10))
    missing_ratio = 0.1
    vars_missing_config = {}
    for feature_idx in features:
        vars_missing_config[feature_idx] = {
            'missing_mechanism': 'mar',
            'missing_ratio': missing_ratio,
            'associated_vars': None,
            'associated_percentiles': None
        }

    # MARY
    features = list(range(10))
    target_index = 11
    missing_ratio = 0.1
    vars_missing_config = {}
    for feature_idx in features:
        vars_missing_config[feature_idx] = {
            'missing_mechanism': 'mary',
            'missing_ratio': missing_ratio,
            'associated_vars': [target_index],
            'associated_percentiles': [[0]]
        }

    # MNAR
    features = list(range(10))
    missing_ratio = 0.1
    vars_missing_config = {}
    for feature_idx in features:
        vars_missing_config[feature_idx] = {
            'missing_mechanism': 'mnar',
            'missing_ratio': missing_ratio,
            'associated_vars': [feature_idx],
            'associated_percentiles': [(0, 0.5)]
        }

    assert False


def test_get_percentiles_from_strategy():
    assert False
