import numpy as np
import pandas as pd
import pytest

import fedimpute.scenario.scenario_builder as scenario_builder_module
from fedimpute.scenario import ScenarioBuilder

pytestmark = pytest.mark.unit


def _patch_simulation_dependencies(monkeypatch):
    calls = {}

    def fake_setup_clients_seed(num_clients, rng):
        return list(range(100, 100 + num_clients))

    def fake_load_data_partition(
        data,
        data_config,
        num_clients,
        split_cols_option,
        partition_strategy,
        seeds,
        **kwargs,
    ):
        calls["partition"] = {
            "data": data,
            "data_config": data_config,
            "num_clients": num_clients,
            "split_cols_option": split_cols_option,
            "partition_strategy": partition_strategy,
            "seeds": seeds,
            "kwargs": kwargs,
        }
        clients_train = [
            np.array(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 2.0, 1.0],
                    [2.0, 3.0, 0.0],
                    [3.0, 4.0, 1.0],
                ]
            )
            for _ in range(num_clients)
        ]
        clients_backup = [np.array([[9.0, 10.0, 1.0]]) for _ in range(num_clients)]
        clients_test = [np.array([[5.0, 6.0, 0.0], [7.0, 8.0, 1.0]]) for _ in range(num_clients)]
        global_test = np.array([[11.0, 12.0, 0.0], [13.0, 14.0, 1.0]])
        return clients_train, clients_backup, clients_test, global_test, {"partition": "ok"}

    def fake_add_missing(clients_data, cols, rngs, obs_cols, **kwargs):
        calls["missing"] = {
            "clients_data": clients_data,
            "cols": cols,
            "rngs": rngs,
            "obs_cols": obs_cols,
            "kwargs": kwargs,
        }
        clients_missing = []
        for client_data in clients_data:
            X_missing = client_data[:, :-1].copy()
            X_missing[0, 0] = np.nan
            clients_missing.append(X_missing)
        return clients_missing

    monkeypatch.setattr(scenario_builder_module, "setup_clients_seed", fake_setup_clients_seed)
    monkeypatch.setattr(scenario_builder_module, "load_data_partition", fake_load_data_partition)
    monkeypatch.setattr(scenario_builder_module, "add_missing", fake_add_missing)
    return calls


def test_summarize_scenario_returns_message_before_initialization():
    scenario_builder = ScenarioBuilder()

    assert scenario_builder.summarize_scenario(return_summary=True) == "Scenario is not initialized."


def test_create_simulated_scenario_forwards_missing_ratio_configuration(monkeypatch):
    calls = _patch_simulation_dependencies(monkeypatch)
    data = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0],
            "x2": [4.0, 5.0, 6.0, 7.0],
            "y": [0.0, 1.0, 0.0, 1.0],
        }
    )
    data_config = {"target": "y", "task_type": "classification", "num_cols": 2}
    scenario_builder = ScenarioBuilder()

    scenario_data = scenario_builder.create_simulated_scenario(
        data,
        data_config,
        num_clients=2,
        dp_strategy="iid-even",
        dp_min_samples=2,
        dp_max_samples=4,
        ms_cols="all-num",
        ms_mr_dist_clients="randu-int",
        ms_mr_clients=[0.2, (0.4, 0.6)],
        ms_mr_lower=0.1,
        ms_mr_upper=0.8,
        seed=123,
        verbose=0,
    )

    missing_kwargs = calls["missing"]["kwargs"]
    assert missing_kwargs["mr_dist"] == "randu-int"
    assert missing_kwargs["ms_mr_clients"] == [0.2, (0.4, 0.6)]
    assert missing_kwargs["mr_lower"] == 0.1
    assert missing_kwargs["mr_upper"] == 0.8
    assert calls["missing"]["cols"] == [0, 1]
    assert calls["missing"]["obs_cols"].tolist() == [1]
    assert scenario_builder.ms_mr_clients == [0.2, (0.4, 0.6)]
    assert scenario_builder.clients_seeds == [100, 101]
    assert list(scenario_data) == [
        "clients_train_data",
        "clients_test_data",
        "clients_train_data_ms",
        "clients_seeds",
        "global_test_data",
        "data_config",
        "stats",
    ]
    assert [client.shape for client in scenario_data["clients_train_data_ms"]] == [(5, 2), (5, 2)]


def test_simulated_scenario_accepts_mixed_ms_mr_clients(
    small_classification_data, small_classification_data_config
):
    scenario_builder = ScenarioBuilder()
    scenario_data = scenario_builder.create_simulated_scenario(
        small_classification_data,
        small_classification_data_config,
        num_clients=3,
        dp_strategy="iid-even",
        dp_min_samples=20,
        dp_max_samples=80,
        dp_local_test_size=0.1,
        dp_global_test_size=0.1,
        dp_local_backup_size=0.05,
        ms_scenario="mcar",
        ms_mr_clients=[0.5, (0.2, 0.3), "large"],
        seed=123,
        verbose=0,
    )

    assert len(scenario_data["clients_train_data_ms"]) == 3
    assert any(client_data.isna().any().any() for client_data in scenario_data["clients_train_data_ms"])


def test_predefined_scenario_overrides_missing_mechanism_parameters(monkeypatch):
    calls = _patch_simulation_dependencies(monkeypatch)
    data = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0],
            "x2": [4.0, 5.0, 6.0, 7.0],
            "y": [0.0, 1.0, 0.0, 1.0],
        }
    )
    data_config = {"target": "y", "task_type": "classification", "num_cols": 2}
    scenario_builder = ScenarioBuilder()

    scenario_builder.create_simulated_scenario(
        data,
        data_config,
        num_clients=2,
        dp_strategy="iid-even",
        ms_scenario="mar-heter",
        ms_mech_type="mcar",
        ms_global_mechanism=True,
        ms_mr_dist_clients="randu-int",
        ms_mm_dist_clients="identity",
        ms_mm_beta_option=None,
        ms_mm_obs=False,
        seed=123,
        verbose=0,
    )

    missing_kwargs = calls["missing"]["kwargs"]
    assert missing_kwargs["mm_mech"] == "mar_logit"
    assert missing_kwargs["global_missing"] is False
    assert missing_kwargs["mr_dist"] == "randu"
    assert missing_kwargs["mm_funcs_dist"] == "random"
    assert missing_kwargs["mm_beta_option"] == "randu"
    assert missing_kwargs["mm_obs"] is True
    assert scenario_builder.ms_mech_type == "mar_logit"
    assert scenario_builder.ms_global_mechanism is False
    assert scenario_builder.ms_mm_dist_clients == "random"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dp_strategy": "unknown"}, "Invalid data partition strategy"),
        ({"dp_strategy": "iid-dir@bad"}, "Invalid data partition strategy"),
        ({"dp_strategy": "iid-even", "dp_split_cols": "bad"}, "Invalid data partition split columns option"),
    ],
)
def test_create_simulated_scenario_rejects_invalid_partition_options(
    small_classification_data, small_classification_data_config, kwargs, message
):
    scenario_builder = ScenarioBuilder()

    with pytest.raises(ValueError, match=message):
        scenario_builder.create_simulated_scenario(
            small_classification_data,
            small_classification_data_config,
            num_clients=2,
            dp_min_samples=20,
            dp_max_samples=80,
            seed=123,
            verbose=0,
            **kwargs,
        )
