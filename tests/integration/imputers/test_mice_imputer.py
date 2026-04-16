import pytest

from tests.integration.imputers._helpers import ICE_WORKFLOW_SMOKE_PARAMS, run_env_for_imputer

pytestmark = pytest.mark.smoke


def test_mice_imputer_runs_through_fedimpute_environment(tmp_path):
    run_env_for_imputer(
        tmp_path=tmp_path,
        imputer="mice",
        fed_strategy="fedmice",
        workflow_params=ICE_WORKFLOW_SMOKE_PARAMS,
    )
