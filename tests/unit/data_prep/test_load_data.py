import importlib

import numpy as np
import pytest

pytestmark = pytest.mark.unit

load_data_module = importlib.import_module("fedimpute.data_prep.load_data")


def test_load_fed_heart_disease_uses_bundled_package_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    data, data_config = load_data_module.load_data("fed_heart_disease")

    assert [frame.shape[0] for frame in data] == [303, 294, 123, 200]
    assert all(frame.columns[-1] == "num" for frame in data)
    assert data_config == {
        "target": "num",
        "task_type": "classification",
        "natural_partition": True,
        "num_cols": 6,
    }
    assert not (tmp_path / "data" / "heart_disease").exists()


@pytest.mark.real_data
def test_load_california_returns_processed_dataframe():
    data, data_config = load_data_module.load_data("california")

    assert data.shape == (5000, 9)
    assert data.columns[-1] == "MedHouseVal"
    assert np.isfinite(data.to_numpy()).all()
    assert data_config == {
        "target": "MedHouseVal",
        "task_type": "regression",
        "natural_partition": False,
        "num_cols": 8,
    }


@pytest.mark.real_data
def test_load_codrna_returns_processed_dataframe():
    data, data_config = load_data_module.load_data("codrna")

    assert data.shape == (5000, 9)
    assert data.columns[-1] == "y"
    assert np.isfinite(data.to_numpy()).all()
    assert data_config == {
        "target": "y",
        "task_type": "classification",
        "natural_partition": False,
        "num_cols": 8,
    }
