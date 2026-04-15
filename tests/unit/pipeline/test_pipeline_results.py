import numpy as np
import pandas as pd
import pytest

from fedimpute.pipeline import FedImputePipeline, FedImputeResult

pytestmark = pytest.mark.unit


def test_pipeline_setup_expands_imputer_strategy_configs():
    pipeline = FedImputePipeline()

    pipeline.setup(
        id="unit",
        fed_imp_configs=[("mean", ["local", "fedmean"], {}, [{}, {}])],
        evaluation_params={"metrics": ["imp_quality"], "model": "lr"},
    )

    assert len(pipeline.fed_imp_configs) == 2
    assert {config["fed_strategy"] for config in pipeline.fed_imp_configs} == {"local", "fedmean"}


def test_pipeline_converts_results_to_tidy_dataframe():
    result = FedImputeResult(
        imputer="mean",
        fed_strategy="fedmean",
        imputer_params={},
        strategy_params={},
        round_id=0,
        results={"imp_quality": pd.DataFrame({"rmse": [0.1, 0.2]})},
        run_time_imp=1.0,
        run_time_eval=2.0,
    )

    tidy = FedImputePipeline._convert_results_to_tidy_dataframe([result])

    assert {"imputer", "fed_strategy", "metric_type", "metric_name", "client_id", "value"}.issubset(tidy.columns)
    assert "imp_quality" in set(tidy["metric_type"])
    assert np.all(np.isfinite(tidy["value"].astype(float)))
