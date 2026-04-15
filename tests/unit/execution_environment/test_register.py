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


def test_register_can_add_and_initialize_custom_components():
    class DummyImputer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyWorkflow:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyClientStrategy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyServerStrategy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    register = Register()
    register.register_workflow("dummy_workflow", DummyWorkflow)
    register.register_strategy("dummy_strategy", DummyClientStrategy, DummyServerStrategy)
    register.register_imputer("dummy_imputer", DummyImputer, "dummy_workflow", ["dummy_strategy"])

    imputer = register.initialize_imputer("dummy_imputer", {"alpha": 1})
    workflow = register.initialize_workflow("dummy_workflow", {"rounds": 2})
    client_strategy = register.initialize_strategy("dummy_strategy", {"lr": 0.1}, "client")
    server_strategy = register.initialize_strategy("dummy_strategy", {"momentum": 0.9}, "server")

    assert isinstance(imputer, DummyImputer)
    assert imputer.kwargs == {"alpha": 1}
    assert isinstance(workflow, DummyWorkflow)
    assert workflow.kwargs == {"rounds": 2}
    assert isinstance(client_strategy, DummyClientStrategy)
    assert client_strategy.kwargs == {"lr": 0.1}
    assert isinstance(server_strategy, DummyServerStrategy)
    assert server_strategy.kwargs == {"momentum": 0.9}
    assert register.get_imputer_workflow_mapping()["dummy_imputer"] == "dummy_workflow"
    assert register.get_imputer_strategy_mapping()["dummy_imputer"] == ["dummy_strategy"]


def test_registry_rejects_duplicate_workflow_and_strategy_registration():
    register = Register()

    with pytest.raises(ValueError, match="Workflow mean already registered"):
        register.register_workflow("mean", object)

    with pytest.raises(ValueError, match="Strategy local already registered"):
        register.register_strategy("local", object, object)


def test_registry_initializers_report_unknown_names_and_invalid_strategy_side():
    register = Register()

    with pytest.raises(ValueError, match="Imputer missing not registered"):
        register.initialize_imputer("missing", {})

    with pytest.raises(ValueError, match="Strategy missing not registered"):
        register.initialize_strategy("missing", {}, "client")

    with pytest.raises(ValueError, match="Workflow missing not registered"):
        register.initialize_workflow("missing", {})

    with pytest.raises(ValueError, match="Invalid client_or_server"):
        register.initialize_strategy("local", {}, "worker")


def test_clean_registration_removes_custom_entries():
    register = Register()
    register.register_imputer("custom", object, "mean", ["local"])
    register.register_workflow("custom_workflow", object)
    register.register_strategy("custom_strategy", object, object)

    register.clean_registration()

    assert "custom" not in register.get_imputer_mapping()
    assert "custom" not in register.get_imputer_workflow_mapping()
    assert "custom" not in register.get_imputer_strategy_mapping()
    assert "custom_workflow" not in register.get_workflow_mapping()
    assert "custom_strategy" not in register.get_strategy_mapping()
