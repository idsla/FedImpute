import numpy as np
import pandas as pd
import pytest

import fedimpute.evaluation.evaluator as evaluator_module
from fedimpute.evaluation import Evaluator

pytestmark = pytest.mark.unit


def _client_frames():
    X_train_imps = [
        pd.DataFrame({"x1": [0.0, 1.0], "x2": [1.0, 0.0]}),
        pd.DataFrame({"x1": [2.0, 3.0], "x2": [0.5, 1.5]}),
    ]
    y_trains = [pd.Series([0, 1]), pd.Series([1, 0])]
    X_tests = [
        pd.DataFrame({"x1": [0.2, 0.8], "x2": [0.9, 0.1]}),
        pd.DataFrame({"x1": [2.2, 2.8], "x2": [0.7, 1.3]}),
    ]
    y_tests = [pd.Series([0, 1]), pd.Series([1, 0])]
    return X_train_imps, y_trains, X_tests, y_tests


def test_evaluate_imp_quality_accepts_dataframe_inputs_and_stores_results():
    evaluator = Evaluator()
    X_origins = [
        pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0]}),
        pd.DataFrame({"x1": [2.0, 4.0], "x2": [6.0, 8.0]}),
    ]
    X_imps = [
        pd.DataFrame({"x1": [1.0, 4.0], "x2": [3.0, 1.0]}),
        pd.DataFrame({"x1": [0.0, 4.0], "x2": [6.0, 10.0]}),
    ]
    X_masks = [
        pd.DataFrame({"x1": [False, True], "x2": [False, True]}),
        pd.DataFrame({"x1": [True, False], "x2": [False, True]}),
    ]

    result = evaluator.evaluate_imp_quality(
        X_train_imps=X_imps,
        X_train_origins=X_origins,
        X_train_masks=X_masks,
        metrics=["rmse", "mae"],
    )

    assert set(result) == {"imp_quality"}
    assert set(result["imp_quality"]) == {"rmse", "mae"}
    assert result == evaluator.results
    assert np.allclose(result["imp_quality"]["rmse"], [np.sqrt(6.5), 2.0])
    assert np.allclose(result["imp_quality"]["mae"], [1.25, 1.0])


def test_run_local_prediction_converts_inputs_and_returns_prediction_results(monkeypatch):
    captured = {}

    def fake_downstream_prediction(
        model,
        model_params,
        X_train_imps,
        y_trains,
        X_tests,
        y_tests,
        data_config,
        seed,
        verbose,
    ):
        captured["model"] = model
        captured["model_params"] = model_params
        captured["X_train_imps"] = X_train_imps
        captured["y_trains"] = y_trains
        captured["X_tests"] = X_tests
        captured["y_tests"] = y_tests
        captured["data_config"] = data_config.copy()
        captured["seed"] = seed
        captured["verbose"] = verbose
        return {"accuracy": [0.75], "f1": [0.8], "auc": [0.9], "prc": [0.85]}

    monkeypatch.setattr(
        Evaluator,
        "_evaluation_downstream_prediction",
        staticmethod(fake_downstream_prediction),
    )
    X_train_imps, y_trains, X_tests, y_tests = _client_frames()
    evaluator = Evaluator()

    result = evaluator.run_local_prediction(
        X_train_imps=X_train_imps,
        y_trains=y_trains,
        X_tests=X_tests,
        y_tests=y_tests,
        data_config={"task_type": "classification"},
        model="lr",
        model_params={"C": 1.0},
        pred_fairness_metrics=["variance", "jain-index"],
        clients_ids=[1],
        seed=42,
        verbose=0,
    )

    assert set(result) == {"local_pred", "local_pred_fairness"}
    assert result["local_pred"] == {"accuracy": [0.75], "f1": [0.8], "auc": [0.9], "prc": [0.85]}
    assert evaluator.results["local_pred"] == result["local_pred"]
    assert captured["model"] == "lr"
    assert captured["model_params"] == {"C": 1.0}
    assert captured["seed"] == 42
    assert captured["verbose"] == 0
    assert captured["data_config"]["clf_type"] == "binary-class"
    assert len(captured["X_train_imps"]) == 1
    assert isinstance(captured["X_train_imps"][0], np.ndarray)
    assert np.array_equal(captured["X_train_imps"][0], X_train_imps[1].values)
    assert np.array_equal(captured["y_trains"][0], y_trains[1].values)
    assert result["local_pred_fairness"]["variance"]["accuracy"] == 0
    assert result["local_pred_fairness"]["jain-index"]["accuracy"] == 1


def test_run_fed_prediction_converts_inputs_and_averages_round_outputs(monkeypatch):
    calls = []

    def fake_eval_fed_pred_lr(
        model_params,
        train_params,
        X_train_imps,
        y_trains,
        X_tests,
        y_tests,
        X_test_global,
        y_test_global,
        data_config,
        seed,
        verbose,
    ):
        calls.append(
            {
                "model_params": model_params,
                "train_params": train_params,
                "X_train_imps": X_train_imps,
                "y_trains": y_trains,
                "X_tests": X_tests,
                "y_tests": y_tests,
                "X_test_global": X_test_global,
                "y_test_global": y_test_global,
                "data_config": data_config.copy(),
                "seed": seed,
                "verbose": verbose,
            }
        )
        offset = seed - 10
        return {
            "global": {"accuracy": [0.8 + offset * 0.1], "f1": [0.7], "auc": [0.9], "prc": [0.85]},
            "personalized": {
                "accuracy": [0.6 + offset * 0.1, 0.7 + offset * 0.1],
                "f1": [0.5, 0.6],
                "auc": [0.8, 0.9],
                "prc": [0.75, 0.85],
            },
        }

    monkeypatch.setattr(evaluator_module, "eval_fed_pred_lr", fake_eval_fed_pred_lr)
    X_train_imps, y_trains, X_tests, y_tests = _client_frames()
    X_test_global = pd.concat(X_tests, ignore_index=True)
    y_test_global = pd.concat(y_tests, ignore_index=True)
    evaluator = Evaluator()

    result = evaluator.run_fed_prediction(
        X_train_imps=X_train_imps,
        y_trains=y_trains,
        X_tests=X_tests,
        y_tests=y_tests,
        X_test_global=X_test_global,
        y_test_global=y_test_global,
        data_config={"task_type": "classification"},
        model_name="lr",
        model_params={"alpha": 0.1},
        train_params={"global_epoch": 1},
        n_rounds=2,
        seed=10,
        verbose=0,
    )

    assert set(result) == {"fed_pred"}
    assert result["fed_pred"] == evaluator.results["fed_pred"]
    assert np.isclose(result["fed_pred"]["global"]["accuracy"], 0.85)
    assert np.allclose(result["fed_pred"]["personalized"]["accuracy"], [0.65, 0.75])
    assert [call["seed"] for call in calls] == [10, 11]
    assert calls[0]["model_params"] == {"alpha": 0.1}
    assert calls[0]["train_params"] == {"global_epoch": 1}
    assert calls[0]["data_config"]["clf_type"] == "binary-class"
    assert isinstance(calls[0]["X_train_imps"][0], np.ndarray)
    assert np.array_equal(calls[0]["X_test_global"], X_test_global.values)
    assert np.array_equal(calls[0]["y_test_global"], y_test_global.values)


def test_export_results_returns_dataframe_and_dict_outputs():
    evaluator = Evaluator()
    evaluator.results = {
        "imp_quality": {"rmse": [0.1, 0.2]},
        "local_pred": {"accuracy": [0.8, 0.9]},
        "fed_pred": {
            "global": {"accuracy": [0.85]},
            "personalized": {"accuracy": [0.75, 0.95]},
        },
    }

    dataframe = evaluator.export_results(format="dataframe")
    dict_dataframes = evaluator.export_results(format="dict-dataframe")
    raw_dict = evaluator.export_results(format="dict")

    assert isinstance(dataframe.columns, pd.MultiIndex)
    assert dataframe.shape == (2, 4)
    assert np.allclose(dataframe[("fed_pred_global", "global_accuracy")], [0.85, 0.85])
    assert set(dict_dataframes) == {
        "imp_quality",
        "local_pred",
        "fed_pred_personalized",
        "fed_pred_global",
    }
    assert dict_dataframes["fed_pred_global"]["global_accuracy"].tolist() == [[0.85], [0.85]]
    assert raw_dict is evaluator.results


def test_export_results_requires_existing_results():
    evaluator = Evaluator()

    with pytest.raises(ValueError, match="No results to export"):
        evaluator.export_results()
