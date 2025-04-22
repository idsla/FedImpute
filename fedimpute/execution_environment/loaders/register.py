from typing import List, Union, Tuple, Dict

from fedimpute.execution_environment.imputation.imputers import (
    LinearICEImputer, SimpleImputer, EMImputer,
    GAINImputer, MIWAEImputer, MissForestImputer, NotMIWAEImputer, GNRImputer
)
from fedimpute.execution_environment.imputation.base import BaseMLImputer, BaseNNImputer
from fedimpute.execution_environment.workflows import (
    BaseWorkflow, WorkflowICE, WorkflowJM, WorkflowSimple, WorkflowEM
)

from fedimpute.execution_environment.fed_strategy.fed_strategy_client import (
    # NN Strategy Client
    FedAvgStrategyClient,
    FedproxStrategyClient,
    ScaffoldStrategyClient,
    LocalNNStrategyClient,
    CentralNNStrategyClient,
    FedAdamStrategyClient,
    FedAdagradStrategyClient,
    FedYogiStrategyClient,
    
    # Traditional Strategy Client
    CentralStrategyClient,
    LocalStrategyClient,
    FedMeanStrategyClient,
    FedMICEStrategyClient,
    FedEMStrategyClient,
    FedTreeStrategyClient,

    # Base Strategy
    StrategyBaseClient
)

from fedimpute.execution_environment.fed_strategy.fed_strategy_server import (
    # NN Strategy Server
    FedAvgStrategyServer,
    FedAvgFtStrategyServer,
    FedproxStrategyServer,
    ScaffoldStrategyServer,
    FedAdamStrategyServer,
    FedAdagradStrategyServer,
    FedYogiStrategyServer,
    LocalNNStrategyServer,
    CentralNNStrategyServer,
    
    # Traditional Strategy Server
    LocalStrategyServer,
    CentralStrategyServer,
    FedTreeStrategyServer,
    FedMeanStrategyServer,
    FedMICEStrategyServer,
    FedEMStrategyServer,

    # Base Strategy
    NNStrategyBaseServer,
    RawBaseStrategyServer
)


class Register:
    """
    Register for the FedImpute environment, contains the mapping of workflows, imputers and imputer strategies
    """
    def __init__(self):
        self.workflow_mapping = self._workflow_mapping()
        self.imputer_mapping = self._imputer_mapping()
        self.imputer_strategy_mapping = self._imputer_strategy_mapping()
        self.imputer_workflow_mapping = self._imputer_workflow_mapping()
        self.strategy_mapping = self._strategy_mapping()

    def clean_registration(self):
        """
        Clean the registration
        """
        self.workflow_mapping = self._workflow_mapping()
        self.imputer_mapping = self._imputer_mapping()
        self.imputer_workflow_mapping = self._imputer_workflow_mapping()
        self.imputer_strategy_mapping = self._imputer_strategy_mapping()
        self.strategy_mapping = self._strategy_mapping()

    def register_imputer(
        self,
        imputer_name: str,
        imputer_class: Union[BaseMLImputer, BaseNNImputer],
        workflow_name: str,
        supported_fed_strategies: List[str]
    ):
        """
        Register a new imputer to the environment

        Args:
            imputer_name: str, name of the imputer
            imputer_class: Union[BaseMLImputer, BaseNNImputer], class of the imputer
            workflow_name: str, name of the workflow
            supported_fed_strategies: List[str], list of supported federated strategies
        
        Note:
            This function will register the imputer to the environment.
        """
        if imputer_name in self.imputer_mapping:
            raise ValueError(f"Imputer {imputer_name} already registered")
        self.imputer_mapping[imputer_name] = imputer_class
        self.imputer_workflow_mapping[imputer_name] = workflow_name
        self.imputer_strategy_mapping[imputer_name] = supported_fed_strategies

    def register_workflow(
        self,
        workflow_name: str,
        workflow_class: BaseWorkflow
    ):
        """
        Register a new workflow to the environment

        Args:
            workflow_name: str, name of the workflow
            workflow_class: BaseWorkflow, class of the workflow
        
        Note:
            This function will register the workflow to the environment. 
        """
        if workflow_name in self.workflow_mapping:
            raise ValueError(f"Workflow {workflow_name} already registered")
        self.workflow_mapping[workflow_name] = workflow_class

    def register_strategy(
        self,
        strategy_name: str,
        strategy_client: StrategyBaseClient,
        strategy_server: Union[RawBaseStrategyServer, NNStrategyBaseServer]
    ):
        """
        Register a new strategy to the environment

        Args:
            strategy_name: str, name of the strategy
            strategy_client: StrategyBaseClient, class of the strategy client
            strategy_server: Union[RawBaseStrategyServer, NNStrategyBaseServer], class of the strategy server
        
        Note:
            This function will register the strategy to the environment.
        """
        if strategy_name in self.strategy_mapping:
            raise ValueError(f"Strategy {strategy_name} already registered")
        self.strategy_mapping[strategy_name] = (strategy_client, strategy_server)

    def _workflow_mapping(self):
        """
        Load the workflow mapping
        """

        workflow_mapping = {
            'mean': WorkflowSimple,
            'ice': WorkflowICE,
            'em': WorkflowEM,
            'jm': WorkflowJM
        }

        return workflow_mapping

    def _strategy_mapping(self):
        """
        Load the strategy mapping
        """

        strategy_mapping = {
            'local': (LocalStrategyClient, LocalStrategyServer),
            'central': (CentralStrategyClient, CentralStrategyServer),
            'fedmean': (FedMeanStrategyClient, FedMeanStrategyServer),
            'fedmice': (FedMICEStrategyClient, FedMICEStrategyServer),
            'fedem': (FedEMStrategyClient, FedEMStrategyServer),
            'fedtree': (FedTreeStrategyClient, FedTreeStrategyServer),
            'local_nn': (LocalNNStrategyClient, LocalNNStrategyServer),
            'central_nn': (CentralNNStrategyClient, CentralNNStrategyServer),
            'fedavg': (FedAvgStrategyClient, FedAvgStrategyServer),
            'fedadam': (FedAdamStrategyClient, FedAdamStrategyServer),
            'fedadagrad': (FedAdagradStrategyClient, FedAdagradStrategyServer),
            'fedyogi': (FedYogiStrategyClient, FedYogiStrategyServer),
            'fedprox': (FedproxStrategyClient, FedproxStrategyServer),
            'scaffold': (ScaffoldStrategyClient, ScaffoldStrategyServer),
            'fedavg_ft': (FedAvgStrategyClient, FedAvgFtStrategyServer)
        }

        return strategy_mapping

    def _imputer_mapping(self):
        """
        Load the imputer mapping
        """

        imputer_mapping = {
            'mean': SimpleImputer,
            'mice': LinearICEImputer,
            'em': EMImputer,
            'missforest': MissForestImputer,
            'gain': GAINImputer,
            'miwae': MIWAEImputer,
            'notmiwae': NotMIWAEImputer,
            'gnr': GNRImputer
        }

        return imputer_mapping
    
    def _imputer_workflow_mapping(self):
        """
        Load the imputer workflow mapping
        """
        imputer_workflow_mapping = {
            'mean': 'mean',
            'mice': 'ice',
            'em': 'em',
            'missforest': 'ice',
            'gain': 'jm',
            'miwae': 'jm',
            'notmiwae': 'jm',
            'gnr': 'jm'
        }

        return imputer_workflow_mapping
    
    def _imputer_strategy_mapping(self):
        """
        Load the imputer strategy mapping
        """

        imputer_strategy_mapping = {
            'mean': ['local', 'central', 'fedmean'],
            'mice': ['local', 'central', 'fedmice'],
            'em': ['local', 'central', 'fedem'],
            'missforest': ['fedtree', 'local', 'central'],
            'gain': [
                'fedavg', 'fedavg_ft', 'fedprox', 'scaffold', 'fedadam', 'fedadagrad', 
                'fedyogi', 'local_nn', 'central_nn'
            ],
            'miwae': [
                'fedavg', 'fedavg_ft', 'fedprox', 'scaffold', 'fedadam', 'fedadagrad', 
                'fedyogi', 'local_nn', 'central_nn'
            ],
            'notmiwae': [
                'fedavg', 'fedavg_ft', 'fedprox', 'scaffold', 'fedadam', 'fedadagrad', 
                'fedyogi', 'local_nn', 'central_nn'
            ],
            'gnr': [
                'fedavg', 'fedavg_ft', 'fedprox', 'scaffold', 'fedadam', 'fedadagrad', 
                'fedyogi', 'local_nn', 'central_nn'
            ]
        }

        return imputer_strategy_mapping

    def get_workflow_mapping(self):
        """
        Get the workflow mapping
        """
        return self.workflow_mapping
    
    def get_imputer_mapping(self):
        """
        Get the imputer mapping
        """
        return self.imputer_mapping
    
    def get_imputer_strategy_mapping(self):
        """
        Get the imputer strategy mapping
        """
        return self.imputer_strategy_mapping
    
    def get_imputer_workflow_mapping(self):
        """
        Get the imputer workflow mapping
        """
        return self.imputer_workflow_mapping
    
    def get_strategy_mapping(self):
        """
        Get the strategy mapping
        """
        return self.strategy_mapping
    
    def initialize_imputer(self, imputer_name: str, imputer_params: dict):
        """
        Initialize an imputer
        """
        if imputer_name not in self.imputer_mapping:
            raise ValueError(f"Imputer {imputer_name} not registered")
        return self.imputer_mapping[imputer_name](**imputer_params)
    
    def initialize_strategy(self, strategy_name: str, strategy_params: dict, client_or_server: str):
        """
        Initialize a strategy
        """
        if strategy_name not in self.strategy_mapping:
            raise ValueError(f"Strategy {strategy_name} not registered")
        
        if client_or_server == 'client':
            return self.strategy_mapping[strategy_name][0](**strategy_params)
        elif client_or_server == 'server':
            return self.strategy_mapping[strategy_name][1](**strategy_params)
        else:
            raise ValueError(f"Invalid client_or_server: {client_or_server}")

    def initialize_workflow(self, workflow_name: str, workflow_params: dict):
        """
        Initialize a workflow
        """
        if workflow_name not in self.workflow_mapping:
            raise ValueError(f"Workflow {workflow_name} not registered")
        return self.workflow_mapping[workflow_name](**workflow_params)
