import gc
import numpy as np
import torch
from typing import Tuple
from .utils import trainable_params

from ...imputation.base import BaseNNImputer
from .strategy_base import StrategyBaseClient

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ScaffoldStrategyClient(StrategyBaseClient):

    def __init__(self):

        super(ScaffoldStrategyClient, self).__init__(name='scaffold')
        self.name = 'scaffold'
        self.local_model = None
        self.global_model = None
        self.global_c = None
        self.client_c = None

    def set_parameters(self, updated_model_params: torch.nn.Module, local_model: torch.nn.Module, params: dict):
        for new_param, old_param in zip(updated_model_params.parameters(), local_model.parameters()):
            old_param.data = new_param.data.clone()

    def pre_training_setup(self, params: dict):
        local_model, global_model, global_c = params['local_model'], params['global_model'], params['global_c']
        for new_param, old_param in zip(global_model.parameters(), local_model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_c = global_c
        self.global_model = global_model
        self.local_model = local_model

        # first time setup of client c
        if self.client_c is None:
            self.client_c = [torch.zeros_like(param) for param in trainable_params(local_model)]

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
    ) -> Tuple[torch.nn.Module, dict]:

        ################################################################################################################
        # training params
        try:
            local_epochs = training_params['local_epoch']
            global_model = training_params['global_model']
            global_c = training_params['global_c']
            learning_rate = training_params['lr']
        except KeyError as e:
            raise ValueError(f"Parameter {str(e)} not found in params")

        ################################################################################################################
        # model and dataloader
        local_model, train_dataloader = imputer.configure_model(training_params, X_train_imp, y_train, X_train_mask)

        # optimizer and scheduler
        training_params['optimizer_name'] = 'scaffold'
        optimizers, lr_schedulers = imputer.configure_optimizer(training_params, local_model)
        local_model.to(DEVICE)

        ################################################################################################################
        # pre-training setup - set global_c, global_model, local_model
        pre_training_params = {
            'local_model': local_model, 'global_model': global_model, 'global_c': global_c
        }
        self.pre_training_setup(pre_training_params)

        ################################################################################################################
        # training loop
        self.local_model.train()
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
                loss, res = self.local_model.train_step(batch, batch_idx, optimizers, optimizer_idx=0)

                #########################################################################
                # scaffold updates
                for p, sc, cc in zip(trainable_params(self.local_model), self.global_c, self.client_c):
                    p.data.add_(sc - cc, alpha=-learning_rate)

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
        self.local_model.to('cpu')

        #########################################################################################
        # post-training setup
        post_training_params = {
            'local_epoch': local_epochs, 'num_batches': len(train_dataloader), 'learning_rate': learning_rate
        }
        post_training_ret = self.post_training_setup(post_training_params)
        delta_y, delta_c = post_training_ret['delta_y'], post_training_ret['delta_c']

        return local_model, {
            'loss': final_loss, 'sample_size': len(train_dataloader.dataset),
            'delta_y': delta_y, 'delta_c': delta_c
        }

    def update_yc(self, local_epochs, num_batches, learning_rate):
        for ci, c, x, yi in zip(
                self.client_c, self.global_c, self.global_model.parameters(), self.local_model.parameters()
        ):
            ci.data = ci - c + 1 / num_batches / local_epochs / learning_rate * (x - yi)

    def delta_yc(self, local_epochs, num_batches, learning_rate):
        delta_y = []
        delta_c = []
        for c, x, yi in zip(self.global_c, self.global_model.parameters(), self.local_model.parameters()):
            delta_y.append(yi - x)
            delta_c.append(- c + 1 / num_batches / local_epochs / learning_rate * (x - yi))

        return delta_y, delta_c
