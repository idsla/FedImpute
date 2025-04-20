import gc
import numpy as np
import torch
from typing import Tuple
from ...imputation.base import BaseNNImputer
from .strategy_base import StrategyBaseClient
from ..utils import get_parameters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FedproxStrategyClient(StrategyBaseClient):

    def __init__(self, mu: float = 1.0):

        super().__init__(name='fedprox')
        self.name = 'fedprox'
        self.global_model_dict = None
        self.mu = mu
        self.loss = None

    def set_parameters(self, updated_model_params: dict, local_model: torch.nn.Module, params: dict):
        state_dict = {k: torch.from_numpy(v.copy()) for k, v in updated_model_params.items()}
        local_model.load_state_dict(state_dict)

    def get_parameters(self, local_model: torch.nn.Module, params: dict) -> dict:
        return get_parameters(local_model, trainable_only=True, return_type='numpy_dict')

    def pre_training_setup(self, params: dict):

        local_model, global_model_dict = params['local_model'], params['global_model_state_dict']
        self.global_model_dict = global_model_dict

        #update local model with global model
        state_dict = {k: torch.from_numpy(v.copy()) for k, v in global_model_dict.items()}
        local_model.load_state_dict(state_dict)

    def post_training_setup(self, params: dict):
        return {}

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
        optimizers, lr_schedulers, optim_params = imputer.configure_optimizer(training_params, local_model)

        ################################################################################################################
        # pre-training setup - set global_c, global_model, local_model
        pre_training_params = {
            'local_model': local_model,
            'global_model_state_dict': self.get_parameters(local_model, {}),
        }

        self.pre_training_setup(pre_training_params)

        ################################################################################################################
        # training loop
        local_model.to(DEVICE)
        total_loss, total_iters = 0, 0
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
                    # fedprox updates
                    for name, w in local_model.named_parameters():
                        if w.requires_grad:
                            w_t = self.global_model_dict[name]
                            w_t = torch.from_numpy(w_t.copy()).to(DEVICE)
                            w.grad.data += self.mu * (w.data - w_t.data)

                    # for w, w_t in zip(
                    #         trainable_params(local_model.state_dict()), trainable_params(self.global_model_dict)
                    # ):
                    #     w.grad.data += self.mu * (w.data - w_t.data)

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

