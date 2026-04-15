import numpy as np
import pandas as pd

from fedimpute.evaluation import Evaluator
from fedimpute.execution_environment import FedImputeEnv
from fedimpute.scenario import ScenarioBuilder


def _make_small_classification_data(n_samples=120):
    rng = np.random.default_rng(2026)
    y = np.array([0.0, 1.0] * (n_samples // 2))
    rng.shuffle(y)
    X = rng.normal(size=(n_samples, 4))
    X[:, 0] += y * 0.5
    data = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])
    data["y"] = y
    return data


def test_basic_usage_smoke_end_to_end(tmp_path):
    data = _make_small_classification_data()
    data_config = {
        "target": "y",
        "task_type": "classification",
        "natural_partition": False,
        "num_cols": 4,
    }

    scenario_builder = ScenarioBuilder()
    scenario_data = scenario_builder.create_simulated_scenario(
        data,
        data_config,
        num_clients=3,
        dp_strategy="iid-even",
        dp_min_samples=20,
        dp_max_samples=80,
        dp_local_test_size=0.1,
        dp_global_test_size=0.1,
        dp_local_backup_size=0.05,
        ms_scenario="mcar",
        ms_mr_lower=0.1,
        ms_mr_upper=0.2,
        seed=123,
        verbose=0,
    )

    assert set(scenario_data) == {
        "clients_train_data",
        "clients_test_data",
        "clients_train_data_ms",
        "clients_seeds",
        "global_test_data",
        "data_config",
        "stats",
    }
    assert len(scenario_builder.clients_train_data) == 3
    assert len(scenario_builder.clients_train_data_ms) == 3
    assert any(client_data.isna().any().any() for client_data in scenario_builder.clients_train_data_ms)

    env = FedImputeEnv(debug_mode=False)
    env.configuration(
        imputer="mean",
        fed_strategy="fedmean",
        seed=123,
        save_dir_path=str(tmp_path / "fedimp"),
    )
    env.setup_from_scenario_builder(scenario_builder=scenario_builder, verbose=0)
    env.run_fed_imputation(verbose=0)

    X_trains = env.get_data(client_ids="all", data_type="train")
    X_train_imps = env.get_data(client_ids="all", data_type="train_imp")
    X_train_masks = env.get_data(client_ids="all", data_type="train_mask")

    assert len(X_train_imps) == 3
    for X_imp, X_origin in zip(X_train_imps, X_trains):
        assert X_imp.shape == X_origin.shape
        assert not X_imp.isna().any().any()

    evaluator = Evaluator()
    result = evaluator.evaluate_imp_quality(
        X_train_imps=X_train_imps,
        X_train_origins=X_trains,
        X_train_masks=X_train_masks,
        metrics=["rmse", "nrmse", "sliced-ws"],
        seed=123,
        verbose=0,
    )

    metrics = result["imp_quality"]
    assert set(metrics) == {"rmse", "nrmse", "sliced-ws"}
    for values in metrics.values():
        assert len(values) == 3
        assert np.all(np.isfinite(values))
        assert np.all(np.asarray(values) >= 0)
