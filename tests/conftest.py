import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_classification_data():
    rng = np.random.default_rng(2026)
    n_samples = 120
    y = np.array([0.0, 1.0] * (n_samples // 2))
    rng.shuffle(y)
    X = rng.normal(size=(n_samples, 4))
    X[:, 0] += y * 0.5
    data = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])
    data["y"] = y
    return data


@pytest.fixture
def small_classification_data_config():
    return {
        "target": "y",
        "task_type": "classification",
        "natural_partition": False,
        "num_cols": 4,
    }


@pytest.fixture
def partitioned_classification_data():
    rng = np.random.default_rng(2028)
    partitions = []
    for client_id in range(3):
        X = rng.normal(loc=client_id * 0.2, size=(40, 4))
        y = (X[:, 0] + X[:, 1] > np.median(X[:, 0] + X[:, 1])).astype(float)
        X_missing = X.copy()
        X_missing[::5, 0] = np.nan
        X_missing[1::7, 2] = np.nan
        data = pd.DataFrame(X_missing, columns=["x1", "x2", "x3", "x4"])
        data["y"] = y
        partitions.append(data)
    return partitions


@pytest.fixture
def partitioned_classification_data_config():
    return {
        "target": "y",
        "task_type": "classification",
        "natural_partition": True,
        "num_cols": 4,
    }
