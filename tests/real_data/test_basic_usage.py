import os
import random

import numpy as np
import pandas as pd
import pytest

from fedimpute.data_prep import load_data
from fedimpute.evaluation import Evaluator
from fedimpute.execution_environment import FedImputeEnv
from fedimpute.scenario import ScenarioBuilder


pytestmark = [
    pytest.mark.real_data,
    pytest.mark.slow,
    pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning"),
]

GLOBAL_SEED = 100330201
MODEL_RTOL = 5e-3
LOCAL_PREDICTION_RTOL = 1e-6
LOCAL_PREDICTION_ATOL = 1e-8
PREDICTION_RTOL = 2e-2
PREDICTION_ATOL = 1e-2


def _set_reproducible_seeds() -> None:
    random.seed(42)
    np.random.seed(42)
    os.environ["PYTHONHASHSEED"] = "0"


def _assert_codrna_data(data: pd.DataFrame, data_config: dict) -> None:
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (5000, 9)
    assert data.columns.tolist() == [f"X{i}" for i in range(1, 9)] + ["y"]
    assert not data.isna().any().any()
    assert np.isfinite(data.to_numpy()).all()
    assert set(data["y"].unique()).issubset({0.0, 1.0})

    expected_means = pd.Series(
        {
            "X1": 0.783610,
            "X2": 0.313509,
            "X3": 0.305492,
            "X4": 0.380526,
            "X5": 0.645553,
            "X6": 0.313939,
            "X7": 0.373722,
            "X8": 0.648571,
            "y": 0.324200,
        }
    )
    pd.testing.assert_series_equal(
        data.mean(numeric_only=True),
        expected_means,
        check_names=False,
        check_dtype=False,
        rtol=1e-5,
        atol=1e-8,
    )

    assert data_config["target"] == "y"
    assert data_config["task_type"] == "classification"
    assert data_config["natural_partition"] is False
    assert data_config["num_cols"] == 8


def _assert_scenario_data(scenario_data: dict, data_config: dict) -> None:
    assert set(scenario_data) == {
        "clients_train_data",
        "clients_test_data",
        "clients_train_data_ms",
        "clients_seeds",
        "global_test_data",
        "data_config",
        "stats",
    }
    assert len(scenario_data["clients_train_data"]) == 4
    assert len(scenario_data["clients_test_data"]) == 4
    assert len(scenario_data["clients_train_data_ms"]) == 4
    assert len(scenario_data["clients_seeds"]) == 4
    assert isinstance(scenario_data["global_test_data"], pd.DataFrame)
    assert scenario_data["data_config"] == data_config

    expected_seeds = [6077, 577, 7231, 5504]
    expected_missing_ratios = [0.47044444, 0.50944444, 0.46244444, 0.47044444]
    expected_train_means = [0.45426866, 0.45586764, 0.45506127, 0.45423325]
    expected_test_means = [0.45292317, 0.45825866, 0.44715748, 0.45541227]
    expected_train_ms_means = [0.4311517, 0.46201121, 0.50622139, 0.52351541]

    for client_idx in range(4):
        train_data = scenario_data["clients_train_data"][client_idx]
        test_data = scenario_data["clients_test_data"][client_idx]
        train_data_ms = scenario_data["clients_train_data_ms"][client_idx]

        assert isinstance(train_data, pd.DataFrame)
        assert train_data.shape == (1125, 9)
        assert np.isclose(
            train_data.to_numpy().mean(), expected_train_means[client_idx], rtol=1e-8
        )

        assert isinstance(test_data, pd.DataFrame)
        assert test_data.shape == (113, 9)
        assert np.isclose(
            test_data.to_numpy().mean(), expected_test_means[client_idx], rtol=1e-8
        )

        assert isinstance(train_data_ms, pd.DataFrame)
        assert train_data_ms.shape == (1125, 8)
        assert np.isclose(
            train_data_ms.isna().to_numpy().mean(),
            expected_missing_ratios[client_idx],
            rtol=1e-8,
        )
        assert np.isclose(
            np.nanmean(train_data_ms.to_numpy()),
            expected_train_ms_means[client_idx],
            rtol=1e-8,
        )

        assert scenario_data["clients_seeds"][client_idx] == expected_seeds[client_idx]


def _assert_env_configuration(env: FedImputeEnv) -> None:
    assert env.workflow.name == "ICE (Imputation via Chain Equation)"
    for client_idx in range(4):
        assert env.clients[client_idx].imputer.name == "mice"
        assert env.clients[client_idx].fed_strategy.name == "fedmice"
    assert env.server.fed_strategy.name == "fedmice"


def _assert_imputed_data(
    env: FedImputeEnv,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    X_trains = env.get_data(client_ids="all", data_type="train")
    X_train_imps = env.get_data(client_ids="all", data_type="train_imp")
    X_train_masks = env.get_data(client_ids="all", data_type="train_mask")

    expected_means = [
        (0.47049669, 0.45157754),
        (0.47229554, 0.44379585),
        (0.47138838, 0.48238528),
        (0.47045685, 0.47645606),
    ]

    for client_idx in range(4):
        assert X_trains[client_idx].shape == (1125, 8)
        assert X_train_imps[client_idx].shape == (1125, 8)
        assert X_train_masks[client_idx].shape == (1125, 8)
        assert not X_trains[client_idx].isna().any().any()
        assert not X_train_imps[client_idx].isna().any().any()
        assert np.isclose(
            np.nanmean(X_trains[client_idx].to_numpy()),
            expected_means[client_idx][0],
            rtol=1e-6,
        )
        assert np.isclose(
            np.nanmean(X_train_imps[client_idx].to_numpy()),
            expected_means[client_idx][1],
            rtol=MODEL_RTOL,
        )

    return X_trains, X_train_imps, X_train_masks


def _assert_imp_quality(evaluator: Evaluator) -> None:
    expected_results = {
        "rmse": [0.18124601, 0.18765093, 0.11695606, 0.1266346],
        "nrmse": [0.5052173, 0.54373233, 0.32421238, 0.35469661],
        "sliced-ws": [0.08026755, 0.093476, 0.04535191, 0.05584367],
    }
    for metric, expected_values in expected_results.items():
        assert np.allclose(
            evaluator.results["imp_quality"][metric], expected_values, rtol=MODEL_RTOL
        )


def _assert_local_prediction(evaluator: Evaluator) -> None:
    expected_results = {
        "accuracy": [
            0.9026548672566371,
            0.9026548672566371,
            0.8495575221238938,
            0.8230088495575221,
        ],
        "f1": [
            0.8705882352941177,
            0.8493150684931506,
            0.8089887640449438,
            0.7777777777777778,
        ],
        "auc": [
            0.9889758179231863,
            0.9615931721194879,
            0.947724039829303,
            0.9711948790896159,
        ],
        "prc": [
            0.9755904395385435,
            0.9358281990553767,
            0.9056074346749368,
            0.9664705546556438,
        ],
    }
    for metric, expected_values in expected_results.items():
        assert np.allclose(
            evaluator.results["local_pred"][metric],
            expected_values,
            rtol=LOCAL_PREDICTION_RTOL,
            atol=LOCAL_PREDICTION_ATOL,
        )


def _assert_fed_prediction(evaluator: Evaluator) -> None:
    expected_global = {
        "accuracy": 0.908,
        "f1": 0.8736263736263736,
        "auc": 0.978614215793703,
        "prc": 0.952616445725458,
    }
    expected_personalized = {
        "accuracy": [
            0.911504424778761,
            0.9203539823008849,
            0.8938053097345132,
            0.8761061946902655,
        ],
        "f1": [
            0.8780487804878049,
            0.891566265060241,
            0.8571428571428571,
            0.8333333333333334,
        ],
        "auc": [
            0.980796586059744,
            0.9957325746799431,
            0.9669274537695591,
            0.9719061166429588,
        ],
        "prc": [
            0.9531852351975345,
            0.9915268803554507,
            0.9446724031437748,
            0.9599141311832291,
        ],
    }

    for metric in expected_global:
        assert np.isclose(
            evaluator.results["fed_pred"]["global"][metric],
            expected_global[metric],
            rtol=PREDICTION_RTOL,
            atol=PREDICTION_ATOL,
        )
        assert np.allclose(
            evaluator.results["fed_pred"]["personalized"][metric],
            expected_personalized[metric],
            rtol=PREDICTION_RTOL,
            atol=PREDICTION_ATOL,
        )


def test_real_world_basic_usage_codrna_end_to_end(tmp_path) -> None:
    _set_reproducible_seeds()

    data, data_config = load_data("codrna")
    _assert_codrna_data(data, data_config)

    scenario_builder = ScenarioBuilder()
    scenario_data = scenario_builder.create_simulated_scenario(
        data,
        data_config,
        num_clients=4,
        dp_strategy="iid-even",
        ms_scenario="mnar-heter",
        seed=GLOBAL_SEED,
        verbose=0,
    )
    _assert_scenario_data(scenario_data, data_config)

    env = FedImputeEnv(debug_mode=False)
    env.configuration(
        imputer="mice",
        fed_strategy="fedmice",
        seed=GLOBAL_SEED,
        save_dir_path=str(tmp_path / "fedimp"),
    )
    env.setup_from_scenario_builder(scenario_builder=scenario_builder, verbose=0)
    env.run_fed_imputation(verbose=0)
    _assert_env_configuration(env)

    X_trains, X_train_imps, X_train_masks = _assert_imputed_data(env)

    evaluator = Evaluator()
    evaluator.evaluate_imp_quality(
        X_train_imps=X_train_imps,
        X_train_origins=X_trains,
        X_train_masks=X_train_masks,
        metrics=["rmse", "nrmse", "sliced-ws"],
        verbose=0,
    )
    _assert_imp_quality(evaluator)

    X_train_imps, y_trains = env.get_data(
        client_ids="all", data_type="train_imp", include_y=True
    )
    X_tests, y_tests = env.get_data(client_ids="all", data_type="test", include_y=True)
    X_global_test, y_global_test = env.get_data(data_type="global_test", include_y=True)
    data_config = env.get_data(data_type="config")

    X_trains, y_trains = env.get_data(
        client_ids="all", data_type="train", include_y=True
    )
    evaluator.run_local_regression_analysis(
        X_train_imps=X_train_imps,
        y_trains=y_trains,
        data_config=data_config,
        verbose=0,
    )
    assert "local_regression" in evaluator.results
    assert len(evaluator.results["local_regression"]) == 4

    evaluator.run_local_prediction(
        X_train_imps=X_train_imps,
        y_trains=y_trains,
        X_tests=X_tests,
        y_tests=y_tests,
        data_config=data_config,
        model="lr",
        seed=0,
        verbose=0,
    )
    _assert_local_prediction(evaluator)

    evaluator.run_fed_prediction(
        X_train_imps=X_train_imps,
        y_trains=y_trains,
        X_tests=X_tests,
        y_tests=y_tests,
        X_test_global=X_global_test,
        y_test_global=y_global_test,
        data_config=data_config,
        model_name="lr",
        seed=0,
        verbose=0,
    )
    _assert_fed_prediction(evaluator)

    evaluator.run_fed_regression_analysis(
        X_train_imps=X_train_imps,
        y_trains=y_trains,
        data_config=data_config,
        verbose=0,
    )
    assert (
        evaluator.results["fed_regression"]["title"]
        == "Federated Logit Regression Result"
    )
