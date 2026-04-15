import numpy as np
import pandas as pd

from fedimpute.evaluation import Evaluator
from fedimpute.execution_environment import FedImputeEnv
from fedimpute.scenario import ScenarioBuilder


def _make_partitioned_data(n_clients=3, n_per_client=40):
    rng = np.random.default_rng(2028)
    partitions = []
    for client_id in range(n_clients):
        X = rng.normal(loc=client_id * 0.2, size=(n_per_client, 4))
        y = (X[:, 0] + X[:, 1] > np.median(X[:, 0] + X[:, 1])).astype(float)
        X_missing = X.copy()
        X_missing[::5, 0] = np.nan
        X_missing[1::7, 2] = np.nan
        data = pd.DataFrame(X_missing, columns=["x1", "x2", "x3", "x4"])
        data["y"] = y
        partitions.append(data)
    return partitions


def test_real_scenario_smoke_with_naturally_partitioned_data(tmp_path):
    data_config = {
        "target": "y",
        "task_type": "classification",
        "natural_partition": True,
        "num_cols": 4,
    }
    scenario_builder = ScenarioBuilder()
    scenario_data = scenario_builder.create_real_scenario(
        _make_partitioned_data(),
        data_config,
        seed=123,
        verbose=0,
    )

    assert len(scenario_data["clients_train_data"]) == 3
    assert len(scenario_builder.clients_train_data_ms) == 3
    assert any(client_data.isna().any().any() for client_data in scenario_builder.clients_train_data_ms)

    env = FedImputeEnv(debug_mode=False)
    env.configuration(
        imputer="mean",
        fed_strategy="fedmean",
        seed=123,
        save_dir_path=str(tmp_path / "fedimp_real"),
    )
    env.setup_from_scenario_builder(scenario_builder=scenario_builder, verbose=0)
    env.run_fed_imputation(verbose=0)

    X_trains = env.get_data(client_ids="all", data_type="train")
    X_train_imps = env.get_data(client_ids="all", data_type="train_imp")
    X_train_masks = env.get_data(client_ids="all", data_type="train_mask")

    assert len(X_train_imps) == 3
    assert all(not X_imp.isna().any().any() for X_imp in X_train_imps)

    evaluator = Evaluator()
    result = evaluator.evaluate_imp_quality(
        X_train_imps=X_train_imps,
        X_train_origins=X_trains,
        X_train_masks=X_train_masks,
        metrics=["rmse", "nrmse"],
        seed=123,
        verbose=0,
    )

    for values in result["imp_quality"].values():
        assert len(values) == 3
        assert np.all(np.isfinite(values))
        assert np.all(np.asarray(values) >= 0)
