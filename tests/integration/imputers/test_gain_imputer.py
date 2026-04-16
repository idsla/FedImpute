import pytest

from tests.integration.imputers._helpers import (
    JM_WORKFLOW_SMOKE_PARAMS,
    SMALL_GAIN_PARAMS,
    run_env_for_imputer,
)

pytestmark = pytest.mark.smoke


def test_gain_imputer_runs_through_fedimpute_environment(tmp_path):
    run_env_for_imputer(
        tmp_path=tmp_path,
        imputer="gain",
        fed_strategy="local",
        imputer_params=SMALL_GAIN_PARAMS,
        workflow_params=JM_WORKFLOW_SMOKE_PARAMS,
        expected_strategy="local_nn",
    )
