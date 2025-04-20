from ..fed_strategy.fed_strategy_client import (
    # NN Strategy Client
    StrategyBaseClient,
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
    FedTreeStrategyClient
)

from ..fed_strategy.fed_strategy_server import (
    # NN Strategy Server
    NNStrategyBaseServer,
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
)

from typing import Union


def load_fed_strategy_client(strategy_name: str, strategy_params: dict) -> Union[
    StrategyBaseClient, FedMeanStrategyClient, FedMICEStrategyClient, FedEMStrategyClient, 
    FedTreeStrategyClient, LocalStrategyClient, CentralStrategyClient
]:

    # Strategies for traditional imputation
    if strategy_name == 'local':
        return LocalStrategyClient()
    elif strategy_name == 'central':
        return CentralStrategyClient()
    elif strategy_name == 'fedmice':
        return FedMICEStrategyClient()
    elif strategy_name == 'fedem':
        return FedEMStrategyClient()
    elif strategy_name == 'fedmean':
        return FedMeanStrategyClient()
    elif strategy_name == 'fedtree':
        return FedTreeStrategyClient()
    # Strategies for deep learning imputation
    elif strategy_name == 'fedavg':
        return FedAvgStrategyClient(global_initialize=False)
    elif strategy_name == 'local_nn':
        return LocalNNStrategyClient()
    elif strategy_name == 'central_nn':
        return CentralNNStrategyClient()
    elif strategy_name == 'fedadam':
        return FedAdamStrategyClient()
    elif strategy_name == 'fedadagrad':
        return FedAdagradStrategyClient()
    elif strategy_name == 'fedyogi':
        return FedYogiStrategyClient()
    elif strategy_name == 'fedprox':
        return FedproxStrategyClient(**strategy_params)
    elif strategy_name == 'scaffold':
        return ScaffoldStrategyClient()
    elif strategy_name == 'fedavg_ft':
        return FedAvgStrategyClient()  # client side fedavg_ft is same as fedavg local training
    else:
        raise ValueError(f"Invalid strategy name: {strategy_name}")


def load_fed_strategy_server(strategy_name: str, strategy_params: dict) -> Union[
    NNStrategyBaseServer, FedMeanStrategyServer, FedMICEStrategyServer, FedEMStrategyServer, 
    FedTreeStrategyServer, LocalStrategyServer, CentralStrategyServer
]:

    # Strategies for traditional imputation
    if strategy_name == 'local':
        return LocalStrategyServer()
    elif strategy_name == 'central':
        return CentralStrategyServer()
    elif strategy_name == 'fedtree':
        return FedTreeStrategyServer()
    elif strategy_name == 'fedmice':
        return FedMICEStrategyServer()
    elif strategy_name == 'fedem':
        return FedEMStrategyServer()
    elif strategy_name == 'fedmean':
        return FedMeanStrategyServer()
    # Strategies for deep learning imputation
    elif strategy_name == 'local_nn':
        return LocalNNStrategyServer()
    elif strategy_name == 'central_nn':
        return CentralNNStrategyServer()
    elif strategy_name == 'fedavg':
        return FedAvgStrategyServer()
    elif strategy_name == 'fedprox':
        return FedproxStrategyServer(**strategy_params)
    elif strategy_name == 'scaffold':
        return ScaffoldStrategyServer(**strategy_params)
    elif strategy_name == 'fedadam':
        return FedAdamStrategyServer(**strategy_params)
    elif strategy_name == 'fedadagrad':
        return FedAdagradStrategyServer(**strategy_params)
    elif strategy_name == 'fedyogi':
        return FedYogiStrategyServer(**strategy_params)
    elif strategy_name == 'fedavg_ft':
        return FedAvgFtStrategyServer(**strategy_params)
    else:
        raise ValueError(f"Invalid strategy name: {strategy_name}")


