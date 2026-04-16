import pytest

from tests.integration.imputers._helpers import EM_WORKFLOW_SMOKE_PARAMS, run_env_for_imputer

pytestmark = pytest.mark.smoke


def test_em_imputer_runs_through_fedimpute_environment(tmp_path):
    run_env_for_imputer(
        tmp_path=tmp_path,
        imputer="em",
        fed_strategy="fedem",
        workflow_params=EM_WORKFLOW_SMOKE_PARAMS,
    )
