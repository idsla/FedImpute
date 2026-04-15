import pytest

from fedimpute.execution_environment import FedImputeEnv
from fedimpute.scenario import ScenarioBuilder

pytestmark = pytest.mark.smoke


def test_real_scenario_smoke_with_naturally_partitioned_data(
    tmp_path, partitioned_classification_data, partitioned_classification_data_config
):
    scenario_builder = ScenarioBuilder()
    scenario_data = scenario_builder.create_real_scenario(
        partitioned_classification_data,
        partitioned_classification_data_config,
        seed=123,
        verbose=0,
    )

    assert len(scenario_data["clients_train_data"]) == 3
    assert len(scenario_builder.clients_train_data_ms) == 3
    assert any(client_data.isna().any().any() for client_data in scenario_builder.clients_train_data_ms)

    env = FedImputeEnv(debug_mode=False)
    env.configuration(
        imputer="mean",
        fed_strategy="fedmean",
        seed=123,
        save_dir_path=str(tmp_path / "fedimp_real"),
    )
    env.setup_from_scenario_builder(scenario_builder=scenario_builder, verbose=0)
    env.run_fed_imputation(verbose=0)

    X_trains = env.get_data(client_ids="all", data_type="train")
    X_train_imps = env.get_data(client_ids="all", data_type="train_imp")

    assert len(X_train_imps) == 3
    assert all(not X_imp.isna().any().any() for X_imp in X_train_imps)
    for X_imp, X_train in zip(X_train_imps, X_trains):
        assert X_imp.shape == X_train.shape
