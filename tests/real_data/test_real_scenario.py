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


def _set_reproducible_seeds() -> None:
    random.seed(42)
    np.random.seed(42)
    os.environ["PYTHONHASHSEED"] = "0"


def _to_float_array(data: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        return data.to_numpy(dtype=float)
    return data.astype(float)


def _assert_heart_disease_data(datas: list[pd.DataFrame], data_config: dict) -> None:
    expected_columns = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
        "slope",
        "sex_1.0",
        "cp_2.0",
        "cp_3.0",
        "cp_4.0",
        "fbs_0.0",
        "fbs_1.0",
        "exang_0.0",
        "exang_1.0",
        "num",
    ]
    expected_shapes = [(303, 15), (294, 15), (123, 15), (200, 15)]
    expected_missing_counts = [0, 215, 26, 274]
    expected_target_means = [
        0.45874587458745875,
        0.36054421768707484,
        0.9349593495934959,
        0.745,
    ]

    assert isinstance(datas, list)
    assert len(datas) == 4
    for client_idx, data in enumerate(datas):
        assert isinstance(data, pd.DataFrame)
        assert data.shape == expected_shapes[client_idx]
        assert data.columns.tolist() == expected_columns
        assert data.isna().sum().sum() == expected_missing_counts[client_idx]
        assert np.isclose(
            data["num"].mean(), expected_target_means[client_idx], rtol=1e-10
        )

    assert data_config == {
        "target": "num",
        "task_type": "classification",
        "natural_partition": True,
        "num_cols": 6,
    }


def _assert_real_scenario_data(scenario_data: dict, data_config: dict) -> None:
    assert set(scenario_data) == {
        "clients_train_data",
        "clients_test_data",
        "clients_train_data_ms",
        "clients_seeds",
        "global_test_data",
        "data_config",
        "stats",
    }
    assert scenario_data["data_config"] == data_config
    assert scenario_data["clients_seeds"] == [6077, 577, 7231, 5504]

    expected_train_shapes = [(244, 15), (237, 15), (99, 15), (162, 15)]
    expected_test_shapes = [(28, 15), (27, 15), (11, 15), (18, 15)]
    expected_train_ms_shapes = [(244, 14), (237, 14), (99, 14), (162, 14)]
    expected_train_missing = [
        0.0,
        0.049789029535864976,
        0.012794612794612794,
        0.0934156378600823,
    ]
    expected_train_ms_missing = [
        0.0,
        0.05334538878842676,
        0.013708513708513708,
        0.10008818342151675,
    ]
    expected_train_means = [
        0.4660166610091586,
        0.4580001283980596,
        0.44186010759905014,
        0.49141634540858675,
    ]
    expected_train_ms_means = [
        0.466516680121054,
        0.46549647683178774,
        0.4058280305341678,
        0.4711368000662012,
    ]

    assert scenario_data["global_test_data"].shape == (94, 15)
    assert np.isclose(
        np.nanmean(_to_float_array(scenario_data["global_test_data"])),
        0.4662014146815018,
        rtol=1e-10,
    )

    for client_idx in range(4):
        train_data = _to_float_array(scenario_data["clients_train_data"][client_idx])
        test_data = _to_float_array(scenario_data["clients_test_data"][client_idx])
        train_data_ms = _to_float_array(
            scenario_data["clients_train_data_ms"][client_idx]
        )

        assert train_data.shape == expected_train_shapes[client_idx]
        assert test_data.shape == expected_test_shapes[client_idx]
        assert train_data_ms.shape == expected_train_ms_shapes[client_idx]

        assert np.isclose(
            np.isnan(train_data).mean(), expected_train_missing[client_idx], rtol=1e-10
        )
        assert np.isclose(
            np.isnan(train_data_ms).mean(),
            expected_train_ms_missing[client_idx],
            rtol=1e-10,
        )
        assert np.isclose(
            np.nanmean(train_data), expected_train_means[client_idx], rtol=1e-10
        )
        assert np.isclose(
            np.nanmean(train_data_ms), expected_train_ms_means[client_idx], rtol=1e-10
        )


def _assert_env_configuration(env: FedImputeEnv) -> None:
    assert env.workflow.name == "ICE (Imputation via Chain Equation)"
    for client_idx in range(4):
        assert env.clients[client_idx].imputer.name == "mice"
        assert env.clients[client_idx].fed_strategy.name == "fedmice"
    assert env.server.fed_strategy.name == "fedmice"


def _assert_imputed_data(
    env: FedImputeEnv,
) -> tuple[list[pd.DataFrame], list[pd.Series]]:
    X_train_imps, y_trains = env.get_data(
        client_ids="all", data_type="train_imp", include_y=True
    )
    X_train_masks = env.get_data(client_ids="all", data_type="train_mask")
    X_tests, y_tests = env.get_data(client_ids="all", data_type="test", include_y=True)
    X_test_imps = env.get_data(client_ids="all", data_type="test_imp")
    X_global_test, y_global_test = env.get_data(data_type="global_test", include_y=True)
    X_global_test_imp = env.get_data(data_type="global_test_imp")

    expected_imp_shapes = [(244, 14), (237, 14), (99, 14), (162, 14)]
    expected_test_shapes = [(28, 14), (27, 14), (11, 14), (18, 14)]
    expected_imp_means = [
        0.466516680121054,
        0.4564782019628209,
        0.40625271492218656,
        0.47160620116246793,
    ]
    expected_mask_means = [
        0.0,
        0.05334538878842676,
        0.013708513708513708,
        0.10008818342151675,
    ]
    expected_y_means = [
        0.45901639344262296,
        0.35864978902953587,
        0.9393939393939394,
        0.7469135802469136,
    ]

    for client_idx in range(4):
        assert X_train_imps[client_idx].shape == expected_imp_shapes[client_idx]
        assert X_train_masks[client_idx].shape == expected_imp_shapes[client_idx]
        assert X_tests[client_idx].shape == expected_test_shapes[client_idx]
        assert X_test_imps[client_idx].shape == expected_test_shapes[client_idx]

        assert not X_train_imps[client_idx].isna().any().any()
        assert not X_test_imps[client_idx].isna().any().any()
        assert np.isclose(
            X_train_masks[client_idx].to_numpy(dtype=float).mean(),
            expected_mask_means[client_idx],
        )
        assert np.isclose(
            np.nanmean(X_train_imps[client_idx].to_numpy(dtype=float)),
            expected_imp_means[client_idx],
            rtol=MODEL_RTOL,
        )
        assert np.isclose(
            y_trains[client_idx].mean(), expected_y_means[client_idx], rtol=1e-10
        )
        assert set(y_tests[client_idx].unique()).issubset({0.0, 1.0})

    assert X_global_test.shape == (94, 14)
    assert y_global_test.shape == (94,)
    assert X_global_test_imp.shape == (94, 14)
    assert not X_global_test_imp.isna().any().any()
    assert np.isclose(
        np.nanmean(X_global_test_imp.to_numpy(dtype=float)),
        0.45685905866190746,
        rtol=MODEL_RTOL,
    )

    return X_train_imps, y_trains


def _assert_fed_regression_results(evaluator: Evaluator) -> None:
    assert (
        evaluator.results["fed_regression"]["title"]
        == "Federated Logit Regression Result"
    )

    result = evaluator.results["fed_regression"]["result"]
    expected_params = {
        "const": -2.013702543280314,
        "age": 1.2148256024270203,
        "trestbps": 0.7583692098559088,
        "chol": -0.8222042186885259,
        "thalach": -1.2272248092033191,
        "oldpeak": 4.190491766817677,
        "slope": 0.9588590516112517,
        "sex_1.0": 1.2921849333703,
        "cp_2.0": -0.884052867685576,
        "cp_3.0": -0.3321566439264118,
        "cp_4.0": 1.17590567749644,
        "fbs_0.0": -1.6317646730768172,
        "fbs_1.0": -0.6695032375559288,
        "exang_0.0": -0.3232615433374842,
        "exang_1.0": 0.7179695510407739,
    }

    assert int(result.nobs) == 742
    assert np.isclose(result.llf, -299.7782719783811, rtol=MODEL_RTOL)
    assert np.isclose(result.prsquared, 0.41219271854370054, rtol=MODEL_RTOL)
    assert set(result.params.index) == set(expected_params)
    for param_name, expected_value in expected_params.items():
        assert np.isclose(result.params[param_name], expected_value, rtol=MODEL_RTOL)


def test_real_world_heart_disease_real_scenario_end_to_end(tmp_path) -> None:
    _set_reproducible_seeds()

    data, data_config = load_data("fed_heart_disease")
    _assert_heart_disease_data(data, data_config)

    scenario_builder = ScenarioBuilder()
    scenario_data = scenario_builder.create_real_scenario(
        data, data_config, seed=GLOBAL_SEED, verbose=0
    )
    _assert_real_scenario_data(scenario_data, data_config)

    env = FedImputeEnv(debug_mode=False)
    env.configuration(
        imputer="mice",
        fed_strategy="fedmice",
        workflow_params={
            "imp_iterations": 10,
            "early_stopping": False,
            "early_stopping_metric": "loss",
        },
        seed=GLOBAL_SEED,
        save_dir_path=str(tmp_path / "fedimp"),
    )
    env.setup_from_scenario_builder(scenario_builder=scenario_builder, verbose=0)
    env.run_fed_imputation(verbose=0)
    _assert_env_configuration(env)

    X_train_imps, y_trains = _assert_imputed_data(env)

    evaluator = Evaluator()
    evaluator.run_fed_regression_analysis(
        X_train_imps=X_train_imps,
        y_trains=y_trains,
        data_config=env.get_data(data_type="config"),
        verbose=0,
    )
    _assert_fed_regression_results(evaluator)
