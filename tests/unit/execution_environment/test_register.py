import pytest

from fedimpute.execution_environment.loaders.register import Register

pytestmark = pytest.mark.unit


def test_default_registry_contains_core_imputers_strategies_and_workflows():
    register = Register()

    assert {"mean", "mice", "em", "missforest"}.issubset(register.get_imputer_mapping())
    assert {"local", "central", "fedmean", "fedmice", "fedem"}.issubset(register.get_strategy_mapping())
    assert {"mean", "ice", "em", "jm"}.issubset(register.get_workflow_mapping())
    assert register.get_imputer_workflow_mapping()["mean"] == "mean"
    assert "fedmean" in register.get_imputer_strategy_mapping()["mean"]


def test_registry_rejects_duplicate_imputer_registration():
    register = Register()

    with pytest.raises(ValueError):
        register.register_imputer("mean", object, "mean", ["local"])
