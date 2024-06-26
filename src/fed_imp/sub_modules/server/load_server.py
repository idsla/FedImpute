from .server_fedavg import ServerFedAvg
from .server_central import ServerCentral
from .server_central_pytorch import ServerCentralPytorch
from .pred_server_central_pytorch import PredServerCentralPytorch
from .pred_server_central_sklearn import PredServerCentralSklearn
from .pred_server_fedavg_pytorch import PredServerFedAvgPytorch
from .server_miwae import ServerVAE
from .server_gain import ServerGAIN


def load_server(server_type: str, **kwargs):
    # whole flow server
    if server_type == 'central_mlp_sklearn':
        return ServerCentral(**kwargs)
    elif server_type == 'central_rf_sklearn':
        return ServerCentral(base_model='rf', **kwargs)
    elif server_type == 'central_lr_sklearn':
        return ServerCentral(base_model='lr', **kwargs)
    elif server_type == 'central_mlp_pytorch':
        return ServerCentralPytorch(**kwargs)
    elif server_type == 'central_lr_pytorch':
        return ServerCentralPytorch(base_model='lr', **kwargs)
    elif server_type == 'fedavg_pytorch':
        return ServerFedAvg(**kwargs)
    
     # vae imputation server
    elif server_type == 'vae':
        return ServerVAE(**kwargs)
    elif server_type == 'gain':
        return ServerGAIN(**kwargs)

    # only prediction server
    elif server_type == 'central_mlp_sklearn_pred':
        return PredServerCentralSklearn(**kwargs)
    elif server_type == 'central_lr_sklearn_pred':
        return PredServerCentralSklearn(base_model='lr', **kwargs)
    elif server_type == 'central_mlp_pytorch_pred':
        return PredServerCentralPytorch(**kwargs)
    elif server_type == 'central_lr_pytorch_pred':
        return PredServerCentralPytorch(base_model='lr', **kwargs)
    elif server_type == 'fedavg_mlp_pytorch_pred':
        return PredServerFedAvgPytorch(**kwargs)
    elif server_type == 'fedavg_lr_pytorch_pred':
        return PredServerFedAvgPytorch(base_model='lr', **kwargs)
    else:
        raise ValueError(f'Invalid server type: {server_type}')

