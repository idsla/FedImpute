from ..fed_strategy.fed_strategy_client import (
    # NN Strategy Client
    StrategyBaseClient,
    FedAvgStrategyClient,
    FedproxStrategyClient,
    ScaffoldStrategyClient,
    # Traditional Strategy Client
    CentralStrategyClient,
    LocalStrategyClient,
    SimpleAvgStrategyClient,
    FedTreeStrategyClient
)

from ..fed_strategy.fed_strategy_server import (
    # NN Strategy Server
    StrategyBaseServer,
    FedAvgStrategyServer,
    FedAvgFtStrategyServer,
    FedproxStrategyServer,
    ScaffoldStrategyServer,
    FedAdamStrategyServer,
    FedAdagradStrategyServer,
    FedYogiStrategyServer,
    # Traditional Strategy Server
    LocalStrategyServer,
    CentralStrategyServer,
    FedTreeStrategyServer,
    SimpleAvgStrategyServer,
)

from typing import Union


def load_fed_strategy_client(strategy_name: str, strategy_params: dict) -> Union[
    StrategyBaseClient, SimpleAvgStrategyClient, FedTreeStrategyClient, LocalStrategyClient, CentralStrategyClient
]:

    # Traditional strategies
    if strategy_name == 'local':
        return LocalStrategyClient()
    elif strategy_name == 'central':
        return CentralStrategyClient()
    elif strategy_name == 'simple_avg':
        return SimpleAvgStrategyClient()
    elif strategy_name == 'fedtree':
        return FedTreeStrategyClient()
    # NN based strategies
    elif strategy_name == 'fedavg':
        return FedAvgStrategyClient(global_initialize=False)
    elif strategy_name == 'fedadam' or strategy_name == 'fedadagrad' or strategy_name == 'fedyogi':
        return FedAvgStrategyClient(global_initialize=True)
    elif strategy_name == 'fedprox':
        return FedproxStrategyClient(**strategy_params)
    elif strategy_name == 'scaffold':
        return ScaffoldStrategyClient()
    elif strategy_name == 'fedavg_ft':
        return FedAvgStrategyClient()
    else:
        raise ValueError(f"Invalid strategy name: {strategy_name}")


def load_fed_strategy_server(strategy_name: str, strategy_params: dict) -> Union[
    StrategyBaseServer, SimpleAvgStrategyServer, FedTreeStrategyServer, LocalStrategyServer, CentralStrategyServer
]:

    # Traditional strategies
    if strategy_name == 'local':
        return LocalStrategyServer()
    elif strategy_name == 'central':
        return CentralStrategyServer()
    elif strategy_name == 'fedtree':
        return FedTreeStrategyServer()
    elif strategy_name == 'simple_avg':
        return SimpleAvgStrategyServer()
    # NN based strategies
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


