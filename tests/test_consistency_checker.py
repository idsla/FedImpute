from src.utils.consistency_checker import check_consistency
import pytest


# Parametrized test for valid configurations
@pytest.mark.parametrize("imputer_name, federated_strategy_name, workflow_name", [
    ("missforest", "local", "ice"),
    ("missforest", "fedtree", "ice"),
    ("missforest", "central", "ice"),
    ("linear_ice", "local", "ice"),
    ("linear_ice", "fedavg", "ice"),
    ("linear_ice", "central", "ice"),
    ("simple", "local", "simple"),
    ("simple", "fedavg", "simple"),
    ("simple", "central", "simple"),
])
def test_valid_configurations(imputer_name, federated_strategy_name, workflow_name):
    check_consistency(imputer_name, federated_strategy_name, workflow_name)


# Parametrized test for invalid configurations
@pytest.mark.parametrize("imputer_name, federated_strategy_name, workflow_name", [
    ("missforest", "fedavg", "ice"),
    ("missforest", "local", "simple"),
    ("linear_ice", "fedtree", "ice"),
    ("linear_ice", "local", "simple"),
    ("simple", "local", "ice"),
    ("simple", "fedavg", "em"),  # Assuming 'em' is a typo or logic error
])
def test_invalid_configurations(imputer_name, federated_strategy_name, workflow_name):
    with pytest.raises(AssertionError):
        check_consistency(imputer_name, federated_strategy_name, workflow_name)
