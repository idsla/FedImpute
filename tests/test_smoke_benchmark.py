import numpy as np
import pandas as pd

from fedimpute.pipeline import FedImputePipeline
from fedimpute.scenario import ScenarioBuilder


def _make_small_classification_data(n_samples=120):
    rng = np.random.default_rng(2027)
    y = np.array([0.0, 1.0] * (n_samples // 2))
    rng.shuffle(y)
    X = rng.normal(size=(n_samples, 4))
    X[:, 0] += y * 0.5
    data = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])
    data["y"] = y
    return data


def test_small_benchmark_smoke_runs_pipeline():
    data = _make_small_classification_data()
    data_config = {
        "target": "y",
        "task_type": "classification",
        "natural_partition": False,
        "num_cols": 4,
    }

    scenario_builder = ScenarioBuilder()
    scenario_builder.create_simulated_scenario(
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

    pipeline = FedImputePipeline()
    pipeline.setup(
        id="small_benchmark_smoke",
        fed_imp_configs=[
            ("mean", ["local", "fedmean"], {}, [{}, {}]),
        ],
        evaluation_params={
            "metrics": ["imp_quality"],
            "model": "lr",
        },
        persist_data=False,
        seed=123,
        description="Small CI benchmark smoke test",
    )
    pipeline.run_pipeline(scenario_builder, repeats=1, seed=123, verbose=0)

    assert len(pipeline.results) == 2
    assert pipeline.tidy_results is not None
    assert set(pipeline.tidy_results["fed_strategy"]) == {"local", "fedmean"}
    assert "imp_quality" in set(pipeline.tidy_results["metric_type"])
    assert np.all(np.isfinite(pipeline.tidy_results["value"].astype(float)))
