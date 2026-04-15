from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from fedimpute.execution_environment import FedImputeEnv
from fedimpute.scenario import ScenarioBuilder

pytestmark = [pytest.mark.real_data]


def _heart_disease_dir():
    candidates = [
        Path("replication_material/data/heart_disease"),
        Path("data/heart_disease"),
    ]
    for candidate in candidates:
        if all((candidate / f"processed.{site}.data").exists() for site in ["cleveland", "hungarian", "switzerland", "va"]):
            return candidate
    pytest.skip("heart disease real data is not available")


def _load_local_heart_disease_partitions():
    data_dir = _heart_disease_dir()
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
    ]
    raw_frames = [
        pd.read_csv(data_dir / f"processed.{site}.data", header=None, na_values="?")
        for site in ["cleveland", "hungarian", "switzerland", "va"]
    ]
    for frame in raw_frames:
        frame.columns = columns
    split_indices = np.cumsum([0] + [frame.shape[0] for frame in raw_frames])
    frame = pd.concat(raw_frames, axis=0).reset_index(drop=True)

    cat_cols = ["sex", "cp", "fbs", "exang"]
    num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "slope"]
    frame = frame.drop(columns=["ca"])
    features = frame[num_cols + cat_cols].copy()
    for col in cat_cols:
        features[col] = features[col].fillna(-1)
    features = pd.get_dummies(features, columns=cat_cols, drop_first=True)
    features[num_cols] = StandardScaler().fit_transform(features[num_cols])
    features[num_cols] = MinMaxScaler().fit_transform(features[num_cols])
    target = frame["num"].apply(lambda value: 0 if value == 0 else 1)
    data = pd.concat([features, target], axis=1)
    partitions = [
        data.iloc[split_indices[idx]:split_indices[idx + 1]].reset_index(drop=True).copy()
        for idx in range(len(split_indices) - 1)
    ]

    return partitions, {
        "target": "num",
        "task_type": "classification",
        "natural_partition": True,
        "num_cols": len(num_cols),
    }


def test_heart_disease_real_scenario_runs_with_local_data(tmp_path):
    partitions, data_config = _load_local_heart_disease_partitions()
    scenario_builder = ScenarioBuilder()
    scenario_builder.create_real_scenario(partitions, data_config, seed=123, verbose=0)

    env = FedImputeEnv(debug_mode=False)
    env.configuration(
        imputer="mean",
        fed_strategy="fedmean",
        seed=123,
        save_dir_path=str(tmp_path / "fedimp_real_data"),
    )
    env.setup_from_scenario_builder(scenario_builder=scenario_builder, verbose=0)
    env.run_fed_imputation(verbose=0)

    X_train_imps = env.get_data(client_ids="all", data_type="train_imp")
    assert len(X_train_imps) == 4
    assert all(np.isfinite(X_imp.to_numpy()).all() for X_imp in X_train_imps)
