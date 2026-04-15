import numpy as np
import pandas as pd

from fedimpute.scenario import ScenarioBuilder
from fedimpute.execution_environment import FedImputeEnv
from fedimpute.evaluation import Evaluator


def test_end_to_end_small_smoke():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    data = pd.DataFrame(
        np.column_stack([X, y]),
        columns=["x1", "x2", "x3", "x4", "y"],
    )
    data_config = {
        "target": "y",
        "task_type": "classification",
        "clf_type": "binary",
        "num_cols": 4,
    }

    scenario_builder = ScenarioBuilder()
    scenario_builder.create_simulated_scenario(
        data,
        data_config,
        num_clients=3,
        dp_strategy="iid-even",
        ms_scenario="mcar",
        ms_mr_lower=0.1,
        ms_mr_upper=0.2,
        seed=0,
        verbose=0,
    )

    env = FedImputeEnv(debug_mode=False)
    env.configuration(imputer="mean", fed_strategy="fedmean", seed=0)
    env.setup_from_scenario_builder(scenario_builder, verbose=0)
    env.run_fed_imputation(verbose=0)

    X_train_imps = env.get_data(client_ids="all", data_type="train_imp")
    X_trains = env.get_data(client_ids="all", data_type="train")
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
        seed=0,
    )

    metrics = result["imp_quality"]
    assert set(metrics) == {"rmse", "nrmse", "sliced-ws"}
    for values in metrics.values():
        assert len(values) == 3
        assert np.all(np.isfinite(values))
        assert np.all(np.asarray(values) >= 0)
