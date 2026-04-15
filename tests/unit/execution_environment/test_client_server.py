import numpy as np
import pytest

from fedimpute.execution_environment.client.client import Client
from fedimpute.execution_environment.server.server import Server

pytestmark = pytest.mark.unit


class DummyImputer:
    model_persistable = False

    def __init__(self):
        self.initialized_with = None
        self.set_params_calls = []

    def initialize(self, X, X_mask, data_utils, params, seed):
        self.initialized_with = (X.copy(), X_mask.copy(), data_utils, params, seed)

    def impute(self, X, y, X_mask, params):
        X_imp = X.copy()
        X_imp[X_mask] = params.get("fill_value", 0)
        return X_imp

    def get_fit_res(self, params):
        return {"fit_model": params.get("fit_model")}

    def get_imp_model_params(self, params):
        return {"model": "params"}

    def set_imp_model_params(self, model_params, params):
        self.set_params_calls.append((model_params, params))


class DummyStrategy:
    def get_global_model_params(self):
        return {"global": "params"}


class DummyRegister:
    def __init__(self):
        self.imputers = []
        self.strategies = []

    def initialize_imputer(self, imputer_name, imputer_params):
        imputer = DummyImputer()
        self.imputers.append((imputer_name, imputer_params, imputer))
        return imputer

    def initialize_strategy(self, strategy_name, strategy_params, client_or_server):
        strategy = DummyStrategy()
        self.strategies.append((strategy_name, strategy_params, client_or_server, strategy))
        return strategy


def _data_config(task_type="classification"):
    return {"task_type": task_type, "num_cols": 2}


def test_client_initializes_data_masks_imputer_and_strategy(tmp_path):
    register = DummyRegister()
    train = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]])
    test = np.array([[5.0, np.nan, 1.0], [6.0, 7.0, 0.0]])
    train_missing = np.array([[1.0, np.nan], [np.nan, 4.0]])

    client = Client(
        client_id=7,
        train_data=train,
        test_data=test,
        X_train_ms=train_missing,
        data_config=_data_config(),
        imp_model_name="mean",
        imp_model_params={"fill": "mean"},
        fed_strategy="local",
        fed_strategy_params={"scope": "client"},
        client_config={"local_dir_path": str(tmp_path)},
        columns=["x1", "x2", "y"],
        register=register,
        seed=123,
    )

    assert client.client_id == 7
    np.testing.assert_array_equal(client.X_train, train[:, :-1])
    np.testing.assert_array_equal(client.y_train, train[:, -1])
    np.testing.assert_array_equal(client.X_test, test[:, :-1])
    np.testing.assert_array_equal(client.y_test, test[:, -1])
    np.testing.assert_array_equal(client.X_train_mask, np.isnan(train_missing))
    assert client.no_ground_truth is False
    assert bool(client.test_missing) is True
    assert client.data_utils["sample_size"] == 2
    assert client.data_utils["n_features"] == 2
    assert client.data_utils["label_stats"]["num_class"] == 2
    assert register.imputers[0][:2] == ("mean", {"fill": "mean"})
    assert register.strategies[0][:3] == ("local", {"scope": "client"}, "client")
    assert client.columns == ["x1", "x2", "y"]


def test_client_initial_and_local_imputation_update_missing_values(tmp_path):
    register = DummyRegister()
    client = Client(
        client_id=0,
        train_data=np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]]),
        test_data=np.array([[5.0, np.nan, 1.0], [6.0, 7.0, 0.0]]),
        X_train_ms=np.array([[1.0, np.nan], [np.nan, 4.0]]),
        data_config=_data_config(),
        imp_model_name="mean",
        imp_model_params={},
        fed_strategy="local",
        fed_strategy_params={},
        client_config={"local_dir_path": str(tmp_path)},
        columns=["x1", "x2", "y"],
        register=register,
        seed=9,
    )

    client.initial_impute(np.array([10.0, 20.0]), col_type="num")
    np.testing.assert_array_equal(client.X_train_imp, np.array([[1.0, 20.0], [10.0, 4.0]]))
    np.testing.assert_array_equal(client.X_test_imp, np.array([[5.0, 20.0], [6.0, 7.0]]))
    assert client.imputer.initialized_with[-1] == 9

    client.X_train_imp[client.X_train_mask] = np.nan
    client.local_imputation({"fill_value": -1.0})

    np.testing.assert_array_equal(client.X_train_imp, np.array([[1.0, -1.0], [-1.0, 4.0]]))
    np.testing.assert_array_equal(client.X_test_imp, np.array([[5.0, -1.0], [6.0, 7.0]]))


def test_server_initializes_global_data_and_applies_global_model_params():
    register = DummyRegister()
    global_test = np.array([[1.0, np.nan, 0.0], [3.0, 4.0, 1.0]])

    server = Server(
        fed_strategy_name="fedmean",
        fed_strategy_params={"scope": "server"},
        imputer_name="mean",
        imputer_params={"fill": "mean"},
        global_test=global_test,
        data_config=_data_config(),
        server_config={"unused": True},
        seed=321,
        columns=["x1", "x2", "y"],
        register=register,
    )

    np.testing.assert_array_equal(server.X_test, global_test[:, :-1])
    np.testing.assert_array_equal(server.y_test, global_test[:, -1])
    np.testing.assert_array_equal(server.X_test_mask, np.isnan(global_test[:, :-1]))
    assert bool(server.test_missing) is True
    assert server.data_utils["sample_size"] == 2
    assert register.imputers[0][:2] == ("mean", {"fill": "mean"})
    assert register.imputers[1][:2] == ("mean", {"fill": "mean"})
    assert register.strategies[0][:3] == ("fedmean", {"scope": "server"}, "server")
    assert server.global_imputer.initialized_with[-1] == 321

    server.local_imputation({"fill_value": 99.0})

    assert server.imputer.set_params_calls == [({"global": "params"}, {"fill_value": 99.0})]
    np.testing.assert_array_equal(server.X_test_imp, np.array([[1.0, 99.0], [3.0, 4.0]]))
