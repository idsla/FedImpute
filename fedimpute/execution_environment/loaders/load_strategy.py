from ..fed_strategy.fed_strategy_client import (
    FedAvgStrategyClient,
    CentralStrategyClient,
    LocalStrategyClient,
    FedProxStrategyClient,
    FedAvgFtStrategyClient,
    StrategyClient,
)

from ..fed_strategy.fed_strategy_server import (
    LocalStrategyServer,
    FedAvgStrategyServer,
    CentralStrategyServer,
    FedProxStrategyServer,
    FedAvgFtStrategyServer,
    FedTreeStrategyServer,
    StrategyServer,
)


def load_fed_strategy_client(strategy_name: str, strategy_params: dict) -> StrategyClient:

    if strategy_name == 'local':
        return LocalStrategyClient(strategy_params)
    elif strategy_name == 'central':
        return CentralStrategyClient(strategy_params)
    elif strategy_name == 'fedavg':
        return FedAvgStrategyClient(strategy_params)
    elif strategy_name == 'fedtree':
        return FedAvgStrategyClient(strategy_params)
    elif strategy_name == 'fedavg_ft':
        return FedAvgFtStrategyClient(strategy_params)
    elif strategy_name == 'fedprox':
        return FedProxStrategyClient(strategy_params)
    elif strategy_name == 'fedavg_ft':
        return FedProxStrategyClient(strategy_params)
    else:
        raise ValueError(f"Invalid strategy name: {strategy_name}")


def load_fed_strategy_server(strategy_name: str, strategy_params: dict) -> StrategyServer:

    if strategy_name == 'local':
        return LocalStrategyServer(strategy_params)
    elif strategy_name == 'central':
        return CentralStrategyServer(strategy_params)
    elif strategy_name == 'fedavg':
        return FedAvgStrategyServer(strategy_params)
    elif strategy_name == 'fedtree':
        return FedTreeStrategyServer(strategy_params)
    elif strategy_name == 'fedprox':
        return FedProxStrategyServer(strategy_params)
    elif strategy_name == 'fedavg_ft':
        return FedAvgFtStrategyServer(strategy_params)
    elif strategy_name == 'fedprox_ft':
        return FedAvgFtStrategyServer(strategy_params)
    else:
        raise ValueError(f"Invalid strategy name: {strategy_name}")


