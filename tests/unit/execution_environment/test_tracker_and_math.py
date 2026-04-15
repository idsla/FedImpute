import numpy as np
import pandas as pd
import pytest

from fedimpute.execution_environment.utils import tracker as tracker_module
from fedimpute.execution_environment.utils.math_util import max_squared_sum

pytestmark = pytest.mark.unit


class DummySummaryWriter:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.scalars = []
        self.closed = False

    def add_scalars(self, tag, scalars, round_num):
        self.scalars.append((tag, scalars, round_num))

    def close(self):
        self.closed = True


@pytest.fixture
def tracker(monkeypatch):
    monkeypatch.setattr(tracker_module, "SummaryWriter", DummySummaryWriter)
    return tracker_module.Tracker({"track_data": False, "track_misc": False})


def test_tracker_records_initial_round_and_final_results(tracker):
    tracker.record_initial(
        data=[np.array([[1.0]]), np.array([[2.0]])],
        mask=[np.array([[False]]), np.array([[True]])],
        imp_quality={"rmse": {0: 0.1, 1: 0.2}},
    )
    tracker.record_round(
        round_num=1,
        imp_quality={"rmse": {0: 0.05, 1: 0.15}},
        data=[],
        model_params=[],
        other_info={"loss": {0: 1.0, 1: 2.0}},
    )
    tracker.record_final(
        imp_quality={"rmse": {0: 0.01, 1: 0.02}},
        data=[],
        model_params=[],
        other_info=None,
    )

    assert tracker.num_clients == 2
    assert tracker.rounds == [0, 1, 3]
    assert tracker.imp_quality == [
        {"rmse": {0: 0.1, 1: 0.2}},
        {"rmse": {0: 0.05, 1: 0.15}},
        {"rmse": {0: 0.01, 1: 0.02}},
    ]
    assert tracker.other_info == [None, {"loss": {0: 1.0, 1: 2.0}}, None]
    assert tracker.writer.scalars == [
        ("imp_quality/rmse", {"client_0": 0.05, "client_1": 0.15}, 1),
        ("other_info/loss", {"client_0": 1.0, "client_1": 2.0}, 1),
    ]
    assert tracker.writer.closed is True
    assert tracker.to_dict() == {
        "results": {
            "rounds": [0, 1, 3],
            "imp_quality": tracker.imp_quality,
            "other_info": tracker.other_info,
        },
        "persist": {},
    }


def test_tracker_to_dict_rejects_persist_mode(monkeypatch):
    monkeypatch.setattr(tracker_module, "SummaryWriter", DummySummaryWriter)
    tracker = tracker_module.Tracker({"track_data": False, "track_misc": False, "persist": "final"})

    with pytest.raises(NotImplementedError, match="Final persist is not implemented yet"):
        tracker.to_dict()


def test_tracker_extracts_metrics_while_ignoring_none_entries():
    metrics = tracker_module.Tracker.get_metrics_from_results(
        [None, {"rmse": {0: 1.0}}, {"ws": {0: 2.0}, "rmse": {0: 0.5}}]
    )

    assert set(metrics) == {"rmse", "ws"}


def test_tracker_processing_tracking_metric_returns_long_dataframe():
    df = tracker_module.Tracker.processing_tracking_metric(
        metric_value_array=[
            {"rmse": {0: 1.0, 1: 2.0}},
            {"rmse": {0: 0.5}, "ws": {1: 0.25}},
        ],
        num_clients=2,
        rounds=[0, 1, 2],
    )

    assert set(df.columns) == {"round", "client", "metric", "value"}
    assert len(df) == 8
    rmse_client0_round0 = df[
        (df["round"] == 0) & (df["client"] == "client_0") & (df["metric"] == "rmse")
    ]["value"].iloc[0]
    ws_client0_round1 = df[
        (df["round"] == 1) & (df["client"] == "client_0") & (df["metric"] == "ws")
    ]["value"].iloc[0]

    assert rmse_client0_round0 == 1.0
    assert pd.isna(ws_client0_round1)


def test_max_squared_sum_returns_largest_row_norm():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [-5.0, 0.0]])

    assert max_squared_sum(X) == 25.0
