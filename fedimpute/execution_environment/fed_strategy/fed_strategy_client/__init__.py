# nn based strategy
from .strategy_base import StrategyBaseClient
from .fedavg import FedAvgStrategyClient
from .fedprox import FedproxStrategyClient
from .scaffold import ScaffoldStrategyClient

# traditional strategy
from .basic_strategy import LocalStrategyClient, CentralStrategyClient, SimpleAvgStrategyClient, FedTreeStrategyClient
