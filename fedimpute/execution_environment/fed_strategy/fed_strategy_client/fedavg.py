import numpy as np

from ...fed_strategy.fed_strategy_client import StrategyBaseClient
import torch
from typing import Tuple
import gc

from ...imputation.base import BaseNNImputer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FedAvgStrategyClient(StrategyBaseClient):

    def __init__(self):
        super().__init__('fedavg')

    def pre_training_setup(self, params: dict):
        return {}

    def post_training_setup(self, params: dict):
        return {}

    def set_parameters(self, updated_model_params: dict, local_model: torch.nn.Module, params: dict):
        # for new_param, old_param in zip(global_model.parameters(), local_model.parameters()):
        #     old_param.data = new_param.data.clone()
        state_dict = {k: torch.from_numpy(v.copy()) for k, v in updated_model_params.items()}
        local_model.load_state_dict(state_dict)

    def get_parameters(self, local_model: torch.nn.Module, params: dict) -> dict:
        return {key: val.cpu().numpy() for key, val in local_model.state_dict().items()}

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
        optimizers, lr_schedulers = imputer.configure_optimizer(training_params, local_model)
        local_model.to(DEVICE)

        ################################################################################################################
        # pre-training setup - set global_c, global_model, local_model
        self.pre_training_setup({})

        ################################################################################################################
        # training loop
        local_model.train()
        total_loss, total_iters = 0, 0
        # for ep in trange(local_epochs, desc='Local Epoch', colour='blue'):
        for ep in range(local_epochs):

            #################################################################################
            # training one epoch
            losses_epoch, ep_iters = [0 for _ in range(len(optimizers))], 0
            for batch_idx, batch in enumerate(train_dataloader):
                # for optimizer_idx, optimizer in enumerate(optimizers):
                #########################################################################
                # training step
                loss, res = local_model.train_step(batch, batch_idx, optimizers, optimizer_idx=0)

                #########################################################################
                # update loss
                for optimizer_idx, optimizer in enumerate(optimizers):
                    losses_epoch[optimizer_idx] += loss

                ep_iters += 1

            #################################################################################
            # epoch end - update loss, early stopping, evaluation, garbage collection etc.
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            losses_epoch = np.array(losses_epoch) / len(train_dataloader)
            epoch_loss = losses_epoch.mean()

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

        return uploaded_params, {
            'loss': final_loss, 'sample_size': len(train_dataloader.dataset),
        }
