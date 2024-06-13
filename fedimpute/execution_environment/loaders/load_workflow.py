from typing import Union, List

from ..workflows.workflow import BaseWorkflow
from ..workflows.workflow_ice import WorkflowICE
from ..workflows.workflow_icegrad import WorkflowICEGrad
from ..workflows.workflow_jm import WorkflowJM
from ..workflows.workflow_simple import WorkflowSimple
from ..workflows.workflow_em import WorkflowEM


def load_workflow(
        workflow_name: str,
        workflow_params: dict,
) -> Union[BaseWorkflow]:
    """
    Load the workflow based on the workflow name
    """
    if workflow_name == 'simple':
        return WorkflowSimple()
    elif workflow_name == 'em':
        return WorkflowEM(**workflow_params)
    elif workflow_name == 'ice':
        return WorkflowICE(**workflow_params)
    elif workflow_name == 'icegrad':
        return WorkflowICEGrad(**workflow_params)
    elif workflow_name == 'jm':
        return WorkflowJM(**workflow_params)
    else:
        raise ValueError(f"Workflow {workflow_name} not supported")