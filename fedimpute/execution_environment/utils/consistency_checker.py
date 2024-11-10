def check_consistency(imputer_name, federated_strategy_name, workflow_name):
    """
    Check constraints of the configuration for imputer, federated server strategy, federated client strategy and workflow
    :param imputer_name: imputation model name
    :param federated_strategy_name: federated strategy name
    :param workflow_name: workflow name
    :return:
    """

    if imputer_name == 'missforest':
        assert (
                (federated_strategy_name in ['local', 'fedtree', 'central']) and
                (workflow_name in ['ice'])
        ), "Inconsistent configuration for {} imputer".format(imputer_name)
    elif imputer_name == 'linear_ice':
        assert (
                (federated_strategy_name in ['local', 'fedavg', 'central']) and
                (workflow_name in ['ice'])
        ), "Inconsistent configuration for {} imputer".format(imputer_name)
    elif imputer_name == 'simple':
        assert (
                (federated_strategy_name in ['local', 'fedavg', 'central']) and
                (workflow_name in ['simple'])
        ), "Inconsistent configuration for {} imputer".format(imputer_name)
    elif imputer_name == 'simple':
        assert (
                (federated_strategy_name in ['local', 'fedavg', 'central']) and
                (workflow_name in ['em'])
        ), "Inconsistent configuration for {} imputer".format(imputer_name)
    else:
        pass
