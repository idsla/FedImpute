import gc
import numpy as np
import torch
from typing import Tuple, Union, List, Dict

from ...imputation.base import BaseNNImputer
from .strategy_base import StrategyBaseClient
from ..utils import get_parameters, convert_params_format

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ScaffoldStrategyClient(StrategyBaseClient):

    def __init__(self):

        super().__init__(name='scaffold')
        self.name = 'scaffold'
        self.local_model = None
        self.global_model_dict: Union[None, Dict[str, np.ndarray]] = None
        self.global_c = None
        self.client_c = None
        self.delta_y = None
        self.delta_c = None
        self.loss = None

    def get_parameters(self, local_model: torch.nn.Module, params: dict) -> dict:
        return get_parameters(local_model, trainable_only=True, return_type='numpy_dict', copy=True)

    def set_parameters(self, updated_model_params: dict, local_model: torch.nn.Module, params: dict):
        state_dict = {k: torch.from_numpy(v.copy()) for k, v in updated_model_params.items()}
        local_model.load_state_dict(state_dict)

    def pre_training_setup(self, params: dict):

        # update local model with global model and global c
        local_model, global_model_dict, global_c = (
            params['local_model'], params['global_model_dict'], params['global_c']
        )

        # for name, param in local_model.named_parameters():
        #     if param.requires_grad and name in global_model_dict:
        #         param.data = torch.from_numpy(global_model_dict[name]).to(param.device)

        self.global_c = global_c
        self.global_model_dict = global_model_dict
        self.local_model = local_model

        # First time setup of client c
        if self.client_c is None:
            self.client_c = [torch.zeros_like(param) for name, param in get_parameters(
                local_model, trainable_only=True, return_type='named_parameters')]

    def post_training_setup(self, params: dict):

        local_epochs = params['local_epoch']
        num_batches = params['num_batches']
        learning_rate = params['learning_rate']
        self.update_yc(local_epochs, num_batches, learning_rate)
        delta_y, delta_c = self.delta_yc(local_epochs, num_batches, learning_rate)
        return {
            'delta_y': delta_y, 'delta_c': delta_c
        }

    def train_local_nn_model(
            self, imputer: BaseNNImputer, training_params: dict, X_train_imp: np.ndarray,
            y_train: np.ndarray, X_train_mask: np.ndarray
    ) -> Tuple[dict, dict]:

        ################################################################################################################
        # training params
        try:
            local_epochs = training_params['local_epoch']
            global_model_dict = training_params['global_model_dict']
            global_c = training_params['global_c']
        except KeyError as e:
            raise ValueError(f"Parameter {str(e)} not found in params")

        ################################################################################################################
        # model and dataloader
        local_model, train_dataloader = imputer.configure_model(training_params, X_train_imp, y_train, X_train_mask)

        # optimizer and scheduler
        training_params['optimizer_name'] = 'sgd'
        optimizers, lr_schedulers, optim_params = imputer.configure_optimizer(training_params, local_model)
        learning_rate = optim_params['learning_rate']

        ################################################################################################################
        # pre-training setup - set global_c, global_model, local_model
        pre_training_params = {
            'local_model': local_model, 'global_model_dict': global_model_dict, 'global_c': global_c
        }
        self.pre_training_setup(pre_training_params)

        ################################################################################################################
        # training loop
        local_model.to(DEVICE)
        ################################################################################################################
        # training loop
        total_loss, total_iters = 0, 0
        # for ep in trange(local_epochs, desc='Local Epoch', colour='blue'):
        for ep in range(local_epochs):
            #################################################################################
            # training one epoch
            losses_epoch, ep_iters = 0, 0

            for batch_idx, batch in enumerate(train_dataloader):
                loss_opt = 0
                for optimizer_idx, optimizer in enumerate(optimizers):
                    #########################################################################
                    # training step
                    local_model.train()
                    optimizer.zero_grad()
                    loss, res = local_model.train_step(batch, batch_idx, optimizers, optimizer_idx=optimizer_idx)
                    loss_opt += loss

                    #########################################################################
                    # scaffold updates
                    for p, sc, cc in zip(
                            get_parameters(local_model, trainable_only=True, return_type='parameters', copy=False),
                            self.global_c, self.client_c
                    ):
                        p.data.add_(sc - cc, alpha=-learning_rate)

                    #########################################################################
                    # update loss
                    optimizer.step()

                loss_opt /= len(optimizers)
                losses_epoch += loss_opt
                ep_iters += 1

            #################################################################################
            # epoch end - update loss, early stopping, evaluation, garbage collection etc.
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            epoch_loss = losses_epoch / len(train_dataloader)

            total_loss += epoch_loss  # average loss
            total_iters += 1

        final_loss = total_loss / total_iters
        gc.collect()
        local_model.to('cpu')

        #########################################################################################
        # post-training setup
        post_training_params = {
            'local_epoch': local_epochs, 'num_batches': len(train_dataloader), 'learning_rate': learning_rate
        }

        post_training_ret = self.post_training_setup(post_training_params)
        delta_y, delta_c = post_training_ret['delta_y'], post_training_ret['delta_c']
        self.loss = final_loss
        return self.get_parameters(local_model, {}), {
            'loss': final_loss, 'sample_size': len(train_dataloader.dataset),
            'delta_y': delta_y, 'delta_c': delta_c
        }

    def update_yc(self, local_epochs, num_batches, learning_rate):
        for ci, c, x, yi in zip(
                self.client_c, self.global_c,
                convert_params_format(self.global_model_dict, 'parameters'),
                get_parameters(self.local_model, trainable_only=True, return_type='parameters')
        ):
            ci.data = ci - c + 1 / num_batches / local_epochs / learning_rate * (x - yi)

    def delta_yc(self, local_epochs, num_batches, learning_rate):
        delta_y = []
        delta_c = []
        for c, x, yi in zip(
                self.global_c,
                convert_params_format(self.global_model_dict, 'parameters'),
                get_parameters(self.local_model, trainable_only=True, return_type='parameters')
        ):
            delta_y.append(yi - x)
            delta_c.append(- c + 1 / num_batches / local_epochs / learning_rate * (x - yi))

        self.delta_y = delta_y
        self.delta_c = delta_c

        return delta_y, delta_c

    def get_fit_res(self, local_model: torch.nn.Module, params: dict) -> dict:
        return {
            'loss': self.loss, 'delta_y': self.delta_y, 'delta_c': self.delta_c
        }
        
    def __str__(self):
        return f"Scaffold Strategy Client"

    def __repr__(self):
        return f"Scaffold Strategy Client"


