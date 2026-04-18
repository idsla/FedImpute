#!/usr/bin/env python
"""
Generate deterministic expected values for real-data integration tests.

Usage:
  python tests/real_data/generate_expected_values.py
  python tests/real_data/generate_expected_values.py --output tests/real_data/expected_values.generated.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fedimpute.data_prep import load_data
from fedimpute.evaluation import Evaluator
from fedimpute.execution_environment import FedImputeEnv
from fedimpute.scenario import ScenarioBuilder


GLOBAL_SEED = 100330201


def _set_reproducible_seeds() -> None:
    random.seed(42)
    np.random.seed(42)
    os.environ["PYTHONHASHSEED"] = "0"


def _to_float_array(data: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        return data.to_numpy(dtype=float)
    return data.astype(float)


def _float(v: Any) -> float:
    return float(v)


def _list_float(arr: Any) -> list[float]:
    return [float(x) for x in arr]


def collect_basic_usage_expected_values() -> dict[str, Any]:
    data, data_config = load_data("codrna")

    codrna = {
        "shape": list(data.shape),
        "columns": data.columns.tolist(),
        "means": {k: _float(v) for k, v in data.mean(numeric_only=True).to_dict().items()},
        "data_config": data_config,
    }

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

    scenario = {
        "clients_seeds": [int(x) for x in scenario_data["clients_seeds"]],
        "train_means": [],
        "test_means": [],
        "train_ms_missing_ratios": [],
        "train_ms_means": [],
    }
    for i in range(4):
        train_data = scenario_data["clients_train_data"][i]
        test_data = scenario_data["clients_test_data"][i]
        train_data_ms = scenario_data["clients_train_data_ms"][i]
        scenario["train_means"].append(_float(train_data.to_numpy().mean()))
        scenario["test_means"].append(_float(test_data.to_numpy().mean()))
        scenario["train_ms_missing_ratios"].append(
            _float(train_data_ms.isna().to_numpy().mean())
        )
        scenario["train_ms_means"].append(_float(np.nanmean(train_data_ms.to_numpy())))

    env = FedImputeEnv(debug_mode=False)
    env.configuration(
        imputer="mice",
        fed_strategy="fedmice",
        workflow_params={"imp_iterations": 10, "early_stopping": False},
        seed=GLOBAL_SEED,
        save_dir_path="./.tmp_expected_values_basic",
    )
    env.setup_from_scenario_builder(scenario_builder=scenario_builder, verbose=0)
    env.run_fed_imputation(verbose=0)

    X_trains = env.get_data(client_ids="all", data_type="train")
    X_train_imps = env.get_data(client_ids="all", data_type="train_imp")
    X_train_masks = env.get_data(client_ids="all", data_type="train_mask")

    imputed = {
        "train_means": [],
        "train_imp_means": [],
        "train_mask_means": [],
    }
    for i in range(4):
        imputed["train_means"].append(_float(np.nanmean(X_trains[i].to_numpy())))
        imputed["train_imp_means"].append(_float(np.nanmean(X_train_imps[i].to_numpy())))
        imputed["train_mask_means"].append(_float(X_train_masks[i].to_numpy(dtype=float).mean()))

    evaluator = Evaluator()
    evaluator.evaluate_imp_quality(
        X_train_imps=X_train_imps,
        X_train_origins=X_trains,
        X_train_masks=X_train_masks,
        metrics=["rmse", "nrmse", "sliced-ws"],
        verbose=0,
    )
    imp_quality = {
        "rmse": _list_float(evaluator.results["imp_quality"]["rmse"]),
        "nrmse": _list_float(evaluator.results["imp_quality"]["nrmse"]),
        "sliced-ws": _list_float(evaluator.results["imp_quality"]["sliced-ws"]),
    }

    X_train_imps, y_trains = env.get_data(
        client_ids="all", data_type="train_imp", include_y=True
    )
    X_tests, y_tests = env.get_data(client_ids="all", data_type="test", include_y=True)
    X_global_test, y_global_test = env.get_data(data_type="global_test", include_y=True)
    config = env.get_data(data_type="config")

    evaluator.run_local_prediction(
        X_train_imps=X_train_imps,
        y_trains=y_trains,
        X_tests=X_tests,
        y_tests=y_tests,
        data_config=config,
        model="lr",
        seed=0,
        verbose=0,
    )
    local_pred = {
        k: _list_float(v) for k, v in evaluator.results["local_pred"].items()
    }

    evaluator.run_fed_prediction(
        X_train_imps=X_train_imps,
        y_trains=y_trains,
        X_tests=X_tests,
        y_tests=y_tests,
        X_test_global=X_global_test,
        y_test_global=y_global_test,
        data_config=config,
        model_name="lr",
        seed=0,
        verbose=0,
    )
    fed_pred = {
        "global": {k: _float(v) for k, v in evaluator.results["fed_pred"]["global"].items()},
        "personalized": {
            k: _list_float(v)
            for k, v in evaluator.results["fed_pred"]["personalized"].items()
        },
    }

    return {
        "codrna": codrna,
        "scenario": scenario,
        "imputed": imputed,
        "imp_quality": imp_quality,
        "local_pred": local_pred,
        "fed_pred": fed_pred,
    }


def collect_real_scenario_expected_values() -> dict[str, Any]:
    datas, data_config = load_data("fed_heart_disease")

    heart_data = {
        "shapes": [list(df.shape) for df in datas],
        "columns": datas[0].columns.tolist(),
        "missing_counts": [int(df.isna().sum().sum()) for df in datas],
        "target_means": [_float(df["num"].mean()) for df in datas],
        "data_config": data_config,
    }

    scenario_builder = ScenarioBuilder()
    scenario_data = scenario_builder.create_real_scenario(
        datas, data_config, seed=GLOBAL_SEED, verbose=0
    )

    scenario = {
        "clients_seeds": [int(x) for x in scenario_data["clients_seeds"]],
        "global_test_shape": list(scenario_data["global_test_data"].shape),
        "global_test_nanmean": _float(
            np.nanmean(_to_float_array(scenario_data["global_test_data"]))
        ),
        "train_shapes": [],
        "test_shapes": [],
        "train_ms_shapes": [],
        "train_missing_ratios": [],
        "train_ms_missing_ratios": [],
        "train_nanmeans": [],
        "train_ms_nanmeans": [],
    }
    for i in range(4):
        train_data = _to_float_array(scenario_data["clients_train_data"][i])
        test_data = _to_float_array(scenario_data["clients_test_data"][i])
        train_data_ms = _to_float_array(scenario_data["clients_train_data_ms"][i])
        scenario["train_shapes"].append(list(train_data.shape))
        scenario["test_shapes"].append(list(test_data.shape))
        scenario["train_ms_shapes"].append(list(train_data_ms.shape))
        scenario["train_missing_ratios"].append(_float(np.isnan(train_data).mean()))
        scenario["train_ms_missing_ratios"].append(_float(np.isnan(train_data_ms).mean()))
        scenario["train_nanmeans"].append(_float(np.nanmean(train_data)))
        scenario["train_ms_nanmeans"].append(_float(np.nanmean(train_data_ms)))

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
        save_dir_path="./.tmp_expected_values_real",
    )
    env.setup_from_scenario_builder(scenario_builder=scenario_builder, verbose=0)
    env.run_fed_imputation(verbose=0)

    X_train_imps, y_trains = env.get_data(
        client_ids="all", data_type="train_imp", include_y=True
    )
    X_train_masks = env.get_data(client_ids="all", data_type="train_mask")
    X_global_test_imp = env.get_data(data_type="global_test_imp")

    imputed = {
        "train_imp_nanmeans": [],
        "train_mask_means": [],
        "y_train_means": [_float(y.mean()) for y in y_trains],
        "global_test_imp_shape": list(X_global_test_imp.shape),
        "global_test_imp_nanmean": _float(np.nanmean(X_global_test_imp.to_numpy(dtype=float))),
    }
    for i in range(4):
        imputed["train_imp_nanmeans"].append(
            _float(np.nanmean(X_train_imps[i].to_numpy(dtype=float)))
        )
        imputed["train_mask_means"].append(
            _float(X_train_masks[i].to_numpy(dtype=float).mean())
        )

    evaluator = Evaluator()
    evaluator.run_fed_regression_analysis(
        X_train_imps=X_train_imps,
        y_trains=y_trains,
        data_config=env.get_data(data_type="config"),
        verbose=0,
    )
    fed_regression_result = evaluator.results["fed_regression"]["result"]
    fed_regression = {
        "nobs": int(fed_regression_result.nobs),
        "llf": _float(fed_regression_result.llf),
        "prsquared": _float(fed_regression_result.prsquared),
        "params": {k: _float(v) for k, v in fed_regression_result.params.to_dict().items()},
    }

    return {
        "heart_data": heart_data,
        "scenario": scenario,
        "imputed": imputed,
        "fed_regression": fed_regression,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate expected values for real-data integration tests."
    )
    parser.add_argument(
        "--output",
        default="tests/real_data/expected_values.generated.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    _set_reproducible_seeds()
    payload = {
        "global_seed": GLOBAL_SEED,
        "basic_usage": collect_basic_usage_expected_values(),
        "real_scenario": collect_real_scenario_expected_values(),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote expected values to: {output_path.resolve()}")
    print("")
    print("Quick copy targets:")
    print("basic_usage.scenario.clients_seeds =", payload["basic_usage"]["scenario"]["clients_seeds"])
    print(
        "real_scenario.scenario.clients_seeds =",
        payload["real_scenario"]["scenario"]["clients_seeds"],
    )
    print(
        "real_scenario.fed_regression.llf =",
        payload["real_scenario"]["fed_regression"]["llf"],
    )


if __name__ == "__main__":
    main()
