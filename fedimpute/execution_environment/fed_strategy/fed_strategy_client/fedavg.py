import numpy as np

from fedimpute.execution_environment.fed_strategy.fed_strategy_client.strategy_base import StrategyBaseClient
import torch
from typing import Tuple
import gc

from ...imputation.base import BaseNNImputer
from ..utils import get_parameters, convert_params_format

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FedAvgStrategyClient(StrategyBaseClient):  # client side fedavg is same as local nn

    def __init__(self, global_initialize=False):
        super().__init__(name='fedavg')
        self.loss = None
        self.initial_aligned = False
        self.global_initialize = global_initialize

    def pre_training_setup(self, params: dict):
        # initialize local model with global model
        return {}

    def post_training_setup(self, params: dict):
        return {}

    def set_parameters(self, updated_model_params: dict, local_model: torch.nn.Module, params: dict):
        state_dict = {k: torch.from_numpy(v.copy()) for k, v in updated_model_params.items()}
        local_model.load_state_dict(state_dict)

    def get_parameters(self, local_model: torch.nn.Module, params: dict) -> dict:
        return get_parameters(local_model, trainable_only=True, return_type='numpy_dict')

    def train_local_nn_model(
            self, imputer: BaseNNImputer, training_params: dict, X_train_imp: np.ndarray,
            y_train: np.ndarray, X_train_mask: np.ndarray
    ) -> Tuple[dict, dict]:

        ################################################################################################################
        # training params
        try:
            local_epochs = training_params['local_epoch']
        except KeyError as e:
            raise ValueError(f"Parameter {str(e)} not found in params")

        ################################################################################################################
        # model and dataloader
        local_model, train_dataloader = imputer.configure_model(training_params, X_train_imp, y_train, X_train_mask)

        # optimizer and scheduler
        optimizers, lr_schedulers, optim_params = imputer.configure_optimizer(training_params, local_model)
        local_model.to(DEVICE)

        ################################################################################################################
        # pre-training setup - set global_c, global_model, local_model
        if self.global_initialize and self.initial_aligned == False:
            try:
                global_model_dict = training_params['global_model_dict']
            except KeyError as e:
                raise ValueError(f"When initializing global model, parameter {str(e)} not found in params")
            
            global_dict = convert_params_format(global_model_dict, output_type='state_dict')
            local_model.load_state_dict(global_dict)
            self.initial_aligned = True

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

            # update lr scheduler
            # for scheduler in lr_schedulers:
            #     scheduler.step()

            total_loss += epoch_loss  # average loss
            total_iters += 1

        final_loss = total_loss / total_iters
        gc.collect()
        local_model.to('cpu')

        #########################################################################################
        # post-training setup
        self.post_training_setup({})
        uploaded_params = self.get_parameters(local_model, {})
        self.loss = final_loss

        return uploaded_params, {
            'loss': final_loss, 'sample_size': len(train_dataloader.dataset),
        }

    def get_fit_res(self, local_model: torch.nn.Module, params: dict) -> dict:
        return {'loss': self.loss}
    
    def __str__(self):
        return f"FedAvg Strategy Client"

    def __repr__(self):
        return f"FedAvg Strategy Client"    
