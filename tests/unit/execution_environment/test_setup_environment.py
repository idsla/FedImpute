import numpy as np
import pandas as pd
import pytest

from fedimpute.execution_environment.loaders import load_environment

pytestmark = pytest.mark.unit


class DummyClient:
    calls = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        DummyClient.calls.append((args, kwargs))


class DummyServer:
    calls = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        DummyServer.calls.append((args, kwargs))


def _frame(values):
    return pd.DataFrame(values, columns=["x1", "x2", "y"])


def test_setup_clients_builds_one_client_per_partition(monkeypatch):
    DummyClient.calls = []
    monkeypatch.setattr(load_environment, "Client", DummyClient)
    register = object()
    clients_data = [
        (_frame([[1, 2, 0]]), _frame([[3, 4, 1]]), _frame([[np.nan, 2, 0]])),
        (_frame([[5, 6, 1]]), _frame([[7, 8, 0]]), _frame([[5, np.nan, 1]])),
    ]

    clients = load_environment.setup_clients(
        clients_data=clients_data,
        clients_seeds=[11, 22],
        data_config={"task_type": "classification"},
        imp_model_name="mean",
        imp_model_params={"strategy": "mean"},
        fed_strategy="local",
        fed_strategy_client_params={"client": True},
        client_config={"local_dir_path": "unused"},
        register=register,
    )

    assert len(clients) == 2
    assert all(isinstance(client, DummyClient) for client in clients)
    first_args, first_kwargs = DummyClient.calls[0]
    assert first_args[0] == 0
    assert first_kwargs["imp_model_name"] == "mean"
    assert first_kwargs["imp_model_params"] == {"strategy": "mean"}
    assert first_kwargs["fed_strategy"] == "local"
    assert first_kwargs["fed_strategy_params"] == {"client": True}
    assert first_kwargs["seed"] == 11
    assert first_kwargs["columns"] == ["x1", "x2", "y"]
    assert first_kwargs["register"] is register
    np.testing.assert_array_equal(first_kwargs["train_data"], clients_data[0][0].values)
    np.testing.assert_array_equal(first_kwargs["test_data"], clients_data[0][1].values)
    np.testing.assert_array_equal(first_kwargs["X_train_ms"], clients_data[0][2].values)
    assert DummyClient.calls[1][0][0] == 1
    assert DummyClient.calls[1][1]["seed"] == 22


def test_setup_server_passes_global_test_values_and_columns(monkeypatch):
    DummyServer.calls = []
    monkeypatch.setattr(load_environment, "Server", DummyServer)
    register = object()
    global_test = _frame([[1, 2, 0], [3, 4, 1]])

    server = load_environment.setup_server(
        fed_strategy="fedmean",
        fed_strategy_params={"server": True},
        imputer_name="mean",
        imputer_params={"strategy": "mean"},
        global_test=global_test,
        data_config={"task_type": "classification"},
        server_config={"local_dir_path": "unused"},
        register=register,
    )

    assert isinstance(server, DummyServer)
    args, kwargs = DummyServer.calls[0]
    assert args[0] == "fedmean"
    assert args[1] == {"server": True}
    assert args[2] == "mean"
    assert args[3] == {"strategy": "mean"}
    np.testing.assert_array_equal(args[4], global_test.values)
    assert args[5] == {"task_type": "classification"}
    assert args[6] == {"local_dir_path": "unused"}
    assert kwargs["columns"] == ["x1", "x2", "y"]
    assert kwargs["register"] is register
