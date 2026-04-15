import pytest

from fedimpute.execution_environment.loaders import load_imputer as load_imputer_module
from fedimpute.execution_environment.loaders import load_strategy as load_strategy_module
from fedimpute.execution_environment.loaders import load_workflow as load_workflow_module

pytestmark = pytest.mark.unit


class Recorder:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _named_recorder(name):
    return type(name, (Recorder,), {})


@pytest.mark.parametrize(
    ("imputer_name", "class_name", "expected_kwargs"),
    [
        ("mean", "SimpleImputer", {"seed": 1}),
        ("mice", "LinearICEImputer", {"seed": 1}),
        ("em", "EMImputer", {"seed": 1}),
        ("missforest", "MissForestImputer", {"seed": 1}),
        ("gain", "GAINImputer", {"seed": 1}),
        ("miwae", "MIWAEImputer", {"name": "miwae", "seed": 1}),
        ("notmiwae", "NotMIWAEImputer", {"seed": 1}),
        ("gnr", "GNRImputer", {"seed": 1}),
    ],
)
def test_load_imputer_routes_supported_names(monkeypatch, imputer_name, class_name, expected_kwargs):
    dummy_class = _named_recorder(class_name)
    monkeypatch.setattr(load_imputer_module, class_name, dummy_class)

    imputer = load_imputer_module.load_imputer(imputer_name, {"seed": 1})

    assert isinstance(imputer, dummy_class)
    assert imputer.kwargs == expected_kwargs


def test_load_imputer_rejects_unknown_name():
    with pytest.raises(NotImplementedError):
        load_imputer_module.load_imputer("unknown", {})


@pytest.mark.parametrize(
    ("workflow_name", "class_name", "expected_kwargs"),
    [
        ("mean", "WorkflowSimple", {}),
        ("em", "WorkflowEM", {"max_iter": 3}),
        ("ice", "WorkflowICE", {"max_iter": 3}),
        ("jm", "WorkflowJM", {"max_iter": 3}),
    ],
)
def test_load_workflow_routes_supported_names(monkeypatch, workflow_name, class_name, expected_kwargs):
    dummy_class = _named_recorder(class_name)
    monkeypatch.setattr(load_workflow_module, class_name, dummy_class)

    workflow = load_workflow_module.load_workflow(workflow_name, {"max_iter": 3})

    assert isinstance(workflow, dummy_class)
    assert workflow.kwargs == expected_kwargs


def test_load_workflow_rejects_unknown_name():
    with pytest.raises(ValueError, match="Workflow unknown not supported"):
        load_workflow_module.load_workflow("unknown", {})


@pytest.mark.parametrize(
    ("strategy_name", "class_name", "expected_kwargs"),
    [
        ("local", "LocalStrategyClient", {}),
        ("central", "CentralStrategyClient", {}),
        ("fedmice", "FedMICEStrategyClient", {}),
        ("fedem", "FedEMStrategyClient", {}),
        ("fedmean", "FedMeanStrategyClient", {}),
        ("fedtree", "FedTreeStrategyClient", {}),
        ("fedavg", "FedAvgStrategyClient", {"global_initialize": False}),
        ("local_nn", "LocalNNStrategyClient", {}),
        ("central_nn", "CentralNNStrategyClient", {}),
        ("fedadam", "FedAdamStrategyClient", {}),
        ("fedadagrad", "FedAdagradStrategyClient", {}),
        ("fedyogi", "FedYogiStrategyClient", {}),
        ("fedprox", "FedproxStrategyClient", {"mu": 0.01}),
        ("scaffold", "ScaffoldStrategyClient", {}),
        ("fedavg_ft", "FedAvgStrategyClient", {}),
    ],
)
def test_load_fed_strategy_client_routes_supported_names(
    monkeypatch, strategy_name, class_name, expected_kwargs
):
    dummy_class = _named_recorder(class_name)
    monkeypatch.setattr(load_strategy_module, class_name, dummy_class)

    strategy = load_strategy_module.load_fed_strategy_client(strategy_name, {"mu": 0.01})

    assert isinstance(strategy, dummy_class)
    assert strategy.kwargs == expected_kwargs


@pytest.mark.parametrize(
    ("strategy_name", "class_name", "expected_kwargs"),
    [
        ("local", "LocalStrategyServer", {}),
        ("central", "CentralStrategyServer", {}),
        ("fedtree", "FedTreeStrategyServer", {}),
        ("fedmice", "FedMICEStrategyServer", {}),
        ("fedem", "FedEMStrategyServer", {}),
        ("fedmean", "FedMeanStrategyServer", {}),
        ("local_nn", "LocalNNStrategyServer", {}),
        ("central_nn", "CentralNNStrategyServer", {}),
        ("fedavg", "FedAvgStrategyServer", {}),
        ("fedprox", "FedproxStrategyServer", {"mu": 0.01}),
        ("scaffold", "ScaffoldStrategyServer", {"mu": 0.01}),
        ("fedadam", "FedAdamStrategyServer", {"mu": 0.01}),
        ("fedadagrad", "FedAdagradStrategyServer", {"mu": 0.01}),
        ("fedyogi", "FedYogiStrategyServer", {"mu": 0.01}),
        ("fedavg_ft", "FedAvgFtStrategyServer", {"mu": 0.01}),
    ],
)
def test_load_fed_strategy_server_routes_supported_names(
    monkeypatch, strategy_name, class_name, expected_kwargs
):
    dummy_class = _named_recorder(class_name)
    monkeypatch.setattr(load_strategy_module, class_name, dummy_class)

    strategy = load_strategy_module.load_fed_strategy_server(strategy_name, {"mu": 0.01})

    assert isinstance(strategy, dummy_class)
    assert strategy.kwargs == expected_kwargs


def test_load_fed_strategy_rejects_unknown_names():
    with pytest.raises(ValueError, match="Invalid strategy name"):
        load_strategy_module.load_fed_strategy_client("unknown", {})

    with pytest.raises(ValueError, match="Invalid strategy name"):
        load_strategy_module.load_fed_strategy_server("unknown", {})
