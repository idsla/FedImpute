import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from fedimpute.execution_environment import FedImputeEnv


def make_small_regression_data():
    rng = np.random.default_rng(20260415)
    n_rows = 40
    x0 = np.linspace(-1.0, 1.0, n_rows)
    x1 = np.sin(np.linspace(0.0, 2.5, n_rows))
    x2 = np.cos(np.linspace(0.2, 2.2, n_rows))
    x3 = rng.normal(0.0, 0.2, n_rows)
    y = 1.4 * x0 - 0.7 * x1 + 0.3 * x2 + 0.1 * x3
    data = pd.DataFrame(
        np.column_stack([x0, x1, x2, x3, y]),
        columns=["x0", "x1", "x2", "x3", "y"],
    )

    clients_train_data = [
        data.iloc[:16].reset_index(drop=True),
        data.iloc[16:32].reset_index(drop=True),
    ]
    clients_test_data = [
        data.iloc[32:36].reset_index(drop=True),
        data.iloc[36:40].reset_index(drop=True),
    ]
    global_test = data.iloc[30:40].reset_index(drop=True)

    clients_train_data_ms = []
    missing_positions = [
        [(1, 0), (5, 0), (3, 1), (7, 1), (9, 2), (13, 2), (11, 3), (15, 3)],
        [(0, 0), (6, 0), (2, 1), (8, 1), (4, 2), (10, 2), (12, 3), (14, 3)],
    ]
    for client_df, positions in zip(clients_train_data, missing_positions):
        features = client_df.drop(columns=["y"]).copy()
        for row_idx, col_idx in positions:
            features.iat[row_idx, col_idx] = np.nan
        clients_train_data_ms.append(features)

    data_config = {
        "target": "y",
        "task_type": "regression",
        "natural_partition": True,
        "num_cols": 4,
    }
    return clients_train_data, clients_test_data, clients_train_data_ms, global_test, data_config


def run_env_for_imputer(
    tmp_path,
    imputer,
    fed_strategy,
    imputer_params=None,
    fed_strategy_params=None,
    workflow_params=None,
    expected_strategy=None,
):
    clients_train_data, clients_test_data, clients_train_data_ms, global_test, data_config = (
        make_small_regression_data()
    )
    env = FedImputeEnv(debug_mode=False)
    env.configuration(
        imputer=imputer,
        fed_strategy=fed_strategy,
        imputer_params=imputer_params or {},
        fed_strategy_params=fed_strategy_params or {},
        workflow_params=workflow_params or {},
        seed=20260415,
        save_dir_path=str(tmp_path / imputer),
    )
    env.setup_from_data(
        clients_train_data=clients_train_data,
        clients_test_data=clients_test_data,
        clients_train_data_ms=clients_train_data_ms,
        global_test=global_test,
        data_config=data_config,
        verbose=0,
    )

    assert env.imputer_name == imputer
    assert env.fed_strategy_name == (expected_strategy or fed_strategy)

    env.run_fed_imputation(run_type="sequential", verbose=0)
    assert_basic_imputation_correctness(env)
    return env


def assert_basic_imputation_correctness(env):
    X_trains = env.get_data(client_ids="all", data_type="train")
    X_train_imps = env.get_data(client_ids="all", data_type="train_imp")
    X_train_masks = env.get_data(client_ids="all", data_type="train_mask")
    X_global_test_imp = env.get_data(data_type="global_test_imp")

    assert len(X_train_imps) == 2
    for X_origin, X_imp, X_mask in zip(X_trains, X_train_imps, X_train_masks):
        origin_values = X_origin.to_numpy()
        imp_values = X_imp.to_numpy()
        mask_values = X_mask.to_numpy(dtype=bool)

        assert imp_values.shape == origin_values.shape
        assert not np.isnan(imp_values).any()
        assert np.isfinite(imp_values).all()
        np.testing.assert_allclose(
            imp_values[~mask_values],
            origin_values[~mask_values],
            rtol=1e-5,
            atol=1e-5,
        )

    global_values = X_global_test_imp.to_numpy()
    assert not np.isnan(global_values).any()
    assert np.isfinite(global_values).all()

    final_quality = env.tracker.imp_quality[-1]
    assert set(final_quality) == {"imp_rmse", "imp_ws"}
    for metric_values in final_quality.values():
        values = np.asarray(list(metric_values.values()), dtype=float)
        assert values.shape == (2,)
        assert np.isfinite(values).all()
        assert (values >= 0).all()


ICE_WORKFLOW_SMOKE_PARAMS = {
    "imp_iterations": 1,
    "early_stopping": False,
    "save_model_interval": 100,
}

EM_WORKFLOW_SMOKE_PARAMS = {
    "max_iterations": 2,
    "local_epoch": 1,
    "evaluation_interval": 1,
    "save_model_interval": 100,
}

JM_WORKFLOW_SMOKE_PARAMS = {
    "global_epoch": 1,
    "local_epoch": 1,
    "use_early_stopping": False,
    "imp_interval": 100,
    "save_model_interval": 100,
}

SMALL_GAIN_PARAMS = {
    "h_dim": 4,
    "n_layers": 1,
    "batch_size": 8,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "scheduler": None,
    "optimizer": "adam",
}

SMALL_VAE_PARAMS = {
    "latent_size": 2,
    "n_hidden": 4,
    "n_hidden_layers": 2,
    "K": 2,
    "L": 2,
    "batch_size": 8,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "scheduler": None,
    "optimizer": "adam",
}
