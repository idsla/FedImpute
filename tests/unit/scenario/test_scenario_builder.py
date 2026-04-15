import pytest

from fedimpute.scenario import ScenarioBuilder

pytestmark = pytest.mark.unit


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
